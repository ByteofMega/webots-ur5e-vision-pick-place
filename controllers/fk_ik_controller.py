# Camera-based pick & place: move a cube from table A to table B.
# UR5e + Robotiq 2F-85, Webots R2025.
#
# A separate overhead_cam node (controllers/camera_node) finds the cube. Pixels are
# converted to meters with a homography fitted at startup from 9 known cube positions.
# Simulator ground truth is used only for calibration and for printing error checks;
# it is never used to decide where to grasp.
import json
import random

import numpy as np
from controller import Supervisor

try:
    import cv2
except ImportError:
    cv2 = None

# ---------- parameters ----------
DH = [
    [0.0000, 0.1625, np.pi / 2, 0],
    [-0.4250, 0.0000, 0, 0],
    [-0.3922, 0.0000, 0, 0],
    [0.0000, 0.1333, np.pi / 2, 0],
    [0.0000, 0.0997, -np.pi / 2, 0],
    [0.0000, 0.0996, 0, 0],
]

JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
JOINT_LIMITS = [(-2 * np.pi, 2 * np.pi), (-np.pi, 0.0), (-np.pi, np.pi),
                (-np.pi, np.pi), (-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi)]
HOME_ANGLES = np.radians([0, -90, 0, -90, 0, 0])  # waiting pose, arm stays clear of table A

GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.5
TOOL_LENGTH = 0.15  # flange to the point between the fingertips, m

BOX_SIZE = 0.07
TABLE_H = 0.20
TABLE_SIZE = (0.36, 0.50)
TABLE_A = np.array([0.47, 0.0])    # pick table
TABLE_B = np.array([-0.45, 0.0])   # place table
PLACE_XY_W = TABLE_B.copy()
SPAWN_X, SPAWN_Y = (0.34, 0.54), (-0.18, 0.18)
AUTO_RESPAWN = True       # if the camera sees no cube for ~3 s, spawn a new one on table A
SPAWN_RANDOM_YAW = True

LIFT_Z = 0.47
MOTOR_VEL = 2.0
V_APPROACH, V_PICK, V_TRANSFER, V_RETREAT, V_HOME = 0.8, 0.3, 0.45, 0.5, 0.9

CAM_POS = (float(TABLE_A[0]), float(TABLE_A[1]), 1.8)
CAM_FOV, CAM_RES = 0.6, 640
# The "forward" direction of a Camera depends on the coordinate convention, so we try a few
CAM_CANDIDATES = [("forward = +X", "0 1 0 1.5708"),
                  ("forward = -Z", "0 0 1 0"),
                  ("forward = +Z", "1 0 0 3.14159")]
CAL_POINTS = [(x, y) for x in (0.34, 0.44, 0.54) for y in (-0.18, 0.0, 0.18)]

R_DOWN = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(float)

HELD_MAX = 0.45  # finger angle below this: cube between fingers; above: closed on nothing

# set after the base calibration below
R_WD = np.eye(3)
base_pos = np.zeros(3)


def world_to_dh(p):
    return R_WD.T @ (np.asarray(p, float) - base_pos)


# ---------- kinematics and planning ----------
def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def yaw_diff(a, b):
    # the cube is symmetric every 90 degrees, so wrap the difference to [-45, 45)
    return (a - b + np.pi / 4) % (np.pi / 2) - np.pi / 4


def dh_matrix(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1]])


def forward_kinematics(q):
    T = np.eye(4)
    pts, axes = [T[:3, 3].copy()], [T[:3, 2].copy()]
    for i, (a, d, al, _) in enumerate(DH):
        T = T @ dh_matrix(a, d, al, q[i])
        pts.append(T[:3, 3].copy())
        axes.append(T[:3, 2].copy())
    return T, pts, axes


def jacobian6(q):
    _, pts, axes = forward_kinematics(q)
    J = np.zeros((6, 6))
    for i in range(6):
        J[:3, i] = np.cross(axes[i], pts[-1] - pts[i])
        J[3:, i] = axes[i]
    return J


def orientation_error(R, Rd):
    return 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))


def ik_single(p, R, q0, mode, iters=600, ptol=1e-3, otol=0.02, step=0.3, damping=0.05):
    """Damped least-squares IK.
    mode='full': position + full orientation.
    mode='axis': position + tool axis direction (rotation about that axis is free)."""
    q = np.array(q0, float)
    for _ in range(iters):
        T, pts, _ = forward_kinematics(q)
        pos_err = p - pts[-1]
        if mode == "full":
            ori_err = orientation_error(T[:3, :3], R)
        else:
            ori_err = np.cross(T[:3, 2], R[:, 2])

        if np.linalg.norm(pos_err) < ptol and np.linalg.norm(ori_err) < otol:
            return q, np.linalg.norm(pos_err), True

        J = jacobian6(q)
        if mode == "axis":
            zd = R[:, 2]
            J[3:] = (np.eye(3) - np.outer(zd, zd)) @ J[3:]

        e = np.concatenate([pos_err, ori_err])
        q = q + step * J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        for j, (lo, hi) in enumerate(JOINT_LIMITS):
            q[j] = np.clip(q[j], lo, hi)

    _, pts, _ = forward_kinematics(q)
    return q, np.linalg.norm(p - pts[-1]), False


def sane(q):
    return (-2.7 < q[1] < -0.45) and (-3.0 < q[2] < -0.3) and abs(q[5]) < 4.6


def ik_global(p, R, seed, mode, n_random=12, want=3):
    pan = np.arctan2(p[1], p[0])
    rng = np.random.default_rng(1)
    seeds = [np.array(seed, float),
             np.array([pan, -1.9, -1.5, 1.5, -np.pi / 2, 0]),
             np.array([pan, -1.5, -1.6, 1.5, -np.pi / 2, 0]),
             np.array([pan, -1.9, 1.9, -1.5, -np.pi / 2, 0]),
             np.array([pan, -1.5, 1.6, -1.7, -np.pi / 2, 0])]
    seeds += [np.array([rng.uniform(lo, hi) for lo, hi in JOINT_LIMITS]) for _ in range(n_random)]

    good, best = [], None
    for s in seeds:
        q, err, ok = ik_single(p, R, s, mode)
        if best is None or err < best[1]:
            best = (q, err)
        if ok and sane(q):
            good.append((q, err, True))
            if len(good) >= want:
                break

    if good:
        return min(good, key=lambda r: np.sum(np.abs(r[0] - seed)))
    return best[0], best[1], False


def cart_path(points, R, q_prev, mode, tag="", verbose=True):
    out, q = [], np.array(q_prev, float)
    for i, p in enumerate(points):
        p = np.array(p)
        q2, err, ok = ik_single(p, R, q, mode)
        if not (ok and sane(q2) and np.max(np.abs(q2 - q)) < 0.35):
            q2, err, ok = ik_global(p, R, q, mode)
            if ok and np.max(np.abs(q2 - q)) >= 0.6:
                ok = False
        if not ok:
            if verbose:
                print(f"  [{tag}] IK failed at point {i + 1}/{len(points)}, "
                      f"p={np.round(p, 3).tolist()}, err {err * 1000:.0f} mm")
            return None
        out.append(q2)
        q = q2
    return out


def line(a, b, step=0.025):
    a, b = np.array(a, float), np.array(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)))
    return [a + (b - a) * k / n for k in range(1, n + 1)]


def arc(p1, p2, z, step=0.08, flip=False):
    a1, a2 = np.arctan2(p1[1], p1[0]), np.arctan2(p2[1], p2[0])
    r1, r2 = np.hypot(p1[0], p1[1]), np.hypot(p2[0], p2[1])
    d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    if flip:
        d = d - 2 * np.pi if d > 0 else d + 2 * np.pi
    n = max(2, int(np.ceil(abs(d) / step)))
    pts = []
    for k in range(1, n + 1):
        r = r1 + (r2 - r1) * k / n
        ang = a1 + d * k / n
        pts.append(np.array([r * np.cos(ang), r * np.sin(ang), z]))
    return pts


def plan_route(box_w, box_yaw, seed, verbose=True):
    """Route for a cube at world pose (box_w, box_yaw). Returns a list of segments or None."""
    z_box = float(box_w[2])
    place_z = TABLE_H + BOX_SIZE / 2 + 0.006 + TOOL_LENGTH

    above_pick = world_to_dh([box_w[0], box_w[1], LIFT_Z])
    pick = world_to_dh([box_w[0], box_w[1], z_box + TOOL_LENGTH])
    above_place = world_to_dh([PLACE_XY_W[0], PLACE_XY_W[1], LIFT_Z])
    place = world_to_dh([PLACE_XY_W[0], PLACE_XY_W[1], place_z])
    R_down_dh = R_WD.T @ R_DOWN

    # of the 4 equivalent grasp yaws, pick the one that needs the least wrist motion
    best = None
    for k in range(4):
        R = R_WD.T @ Rz(box_yaw + k * np.pi / 2) @ R_DOWN
        q, err, ok = ik_global(above_pick, R, seed, "full")
        if ok and sane(q):
            cost = abs(q[5]) + 0.3 * np.sum(np.abs(q - seed))
            if best is None or cost < best[0]:
                best = (cost, R, q)
    if best is None:
        if verbose:
            print("  [ABOVE_PICK] no IK solution (cube out of reach?)")
        return None
    _, R, q_above = best

    segs = [("ABOVE_PICK", [q_above], V_APPROACH)]

    down = cart_path(line(above_pick, pick), R, q_above, "full", "PICK down", verbose)
    if down is None:
        return None
    segs.append(("PICK", down, V_PICK))

    up = cart_path(line(pick, above_pick), R, down[-1], "full", "LIFT up", verbose)
    if up is None:
        return None
    segs.append(("LIFT", up, V_PICK))

    transfer = None
    for flip in (False, True):
        transfer = cart_path(arc(above_pick, above_place, above_place[2], flip=flip),
                             R_down_dh, up[-1], "axis", "ARC", verbose)
        if transfer is not None:
            break
    if transfer is None:
        return None
    segs.append(("TRANSFER", transfer, V_TRANSFER))

    lower = cart_path(line(above_place, place), R_down_dh, transfer[-1], "axis", "PLACE down", verbose)
    if lower is None:
        return None
    segs.append(("PLACE", lower, V_PICK))

    retreat = cart_path(line(place, above_place), R_down_dh, lower[-1], "axis", "RETREAT up", verbose)
    if retreat is None:
        return None
    segs.append(("RETREAT", retreat, V_RETREAT))
    return segs


def on_table(p, center, margin=0.0):
    return (abs(p[0] - center[0]) <= TABLE_SIZE[0] / 2 + margin and
            abs(p[1] - center[1]) <= TABLE_SIZE[1] / 2 + margin)


# ---------- vision ----------
def fit_homography(uv, xy):
    """Normalized DLT: pixels -> world XY, needs at least 4 points."""
    uv, xy = np.asarray(uv, float), np.asarray(xy, float)

    def norm_matrix(P):
        c = P.mean(0)
        s = np.sqrt(2) / np.mean(np.linalg.norm(P - c, axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])

    Tu, Tx = norm_matrix(uv), norm_matrix(xy)
    rows = []
    for (u, v), (x, y) in zip(uv, xy):
        pu = Tu @ [u, v, 1]
        px = Tx @ [x, y, 1]
        rows.append([pu[0], pu[1], 1, 0, 0, 0, -px[0] * pu[0], -px[0] * pu[1], -px[0]])
        rows.append([0, 0, 0, pu[0], pu[1], 1, -px[1] * pu[0], -px[1] * pu[1], -px[1]])

    _, _, Vt = np.linalg.svd(np.array(rows))
    H = np.linalg.inv(Tx) @ Vt[-1].reshape(3, 3) @ Tu
    return H / H[2, 2]


def apply_h(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def pose_from_detection(H, det, z):
    """Cube world pose: (x, y, z) and yaw from the first edge of the minAreaRect."""
    x, y = apply_h(H, det["cx"], det["cy"])
    p0 = apply_h(H, *det["box"][0])
    p1 = apply_h(H, *det["box"][1])
    return np.array([x, y, z]), float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


# ---------- Webots setup ----------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
DT = timestep / 1000.0

motors, sensors = [], []
for name in JOINT_NAMES:
    motors.append(robot.getDevice(name))
    s = robot.getDevice(name + "_sensor")
    s.enable(timestep)
    sensors.append(s)

gripper_motors = [robot.getDevice("ROBOTIQ 2F-85 Gripper::left finger joint"),
                  robot.getDevice("ROBOTIQ 2F-85 Gripper::right finger joint")]
gripper_sensors = [robot.getDevice("ROBOTIQ 2F-85 Gripper left finger joint sensor"),
                   robot.getDevice("ROBOTIQ 2F-85 Gripper right finger sensor")]
for s in gripper_sensors:
    s.enable(timestep)


def read_q():
    return np.array([s.getValue() for s in sensors])


def finger_pos():
    return float(np.mean([s.getValue() for s in gripper_sensors]))


def set_gripper(pos, speed=0.5):
    for m in gripper_motors:
        m.setVelocity(speed)
        m.setPosition(pos)


for m, a in zip(motors, HOME_ANGLES):
    m.setVelocity(1.0)
    m.setPosition(float(a))
set_gripper(GRIPPER_OPEN)
for _ in range(120):
    robot.step(timestep)

# robot calibration: DH frame <-> world frame
self_node = robot.getSelf()
base_pos = np.array(self_node.getPosition())
R_base = np.array(self_node.getOrientation()).reshape(3, 3)

tool_node = None
try:
    slot = self_node.getField("toolSlot")
    tool_node = slot.getMFNode(0) if slot is not None else None
except Exception:
    tool_node = None

print("Pick & place by camera: table A -> table B (UR5e + Robotiq 2F-85)")

yaw = 0.0
if tool_node is not None:
    _, pts, _ = forward_kinematics(read_q())
    p_local = R_base.T @ (np.array(tool_node.getPosition()) - base_pos)
    yaw_raw = np.arctan2(p_local[1], p_local[0]) - np.arctan2(pts[-1][1], pts[-1][0])
    yaw = round(yaw_raw / (np.pi / 2)) * (np.pi / 2)
    resid = np.linalg.norm(Rz(yaw) @ pts[-1] - p_local)
    print(f"Robot calibration: DH->world yaw = {np.degrees(yaw):.0f} deg, residual {resid * 1000:.1f} mm")
    if resid > 0.02:
        print("  WARNING: residual is large, check the toolSlot / base orientation")
else:
    print("WARNING: no node in toolSlot, using yaw = 0")
R_WD = R_base @ Rz(yaw)


def tool_world():
    if tool_node is not None:
        return np.array(tool_node.getPosition())
    return R_WD @ forward_kinematics(read_q())[1][-1] + base_pos


# ---------- scene ----------
root = robot.getRoot().getField("children")

try:
    props = root.getMFNode(0).getField("contactProperties")
    if props is not None:
        props.importMFNodeFromString(
            -1, 'ContactProperties { material1 "box_mat" material2 "default" coulombFriction [ 4 ] }')
except Exception as e:
    print("  (friction not changed:", e, ")")


def add_table(def_name, c, rgb):
    root.importMFNodeFromString(-1, f"""
    DEF {def_name} Solid {{
      translation {c[0]} {c[1]} {TABLE_H / 2}
      children [
        Shape {{
          geometry Box {{ size {TABLE_SIZE[0]} {TABLE_SIZE[1]} {TABLE_H} }}
          appearance PBRAppearance {{ baseColor {rgb} roughness 0.8 metalness 0 }}
        }}
      ]
      boundingObject Box {{ size {TABLE_SIZE[0]} {TABLE_SIZE[1]} {TABLE_H} }}
    }}
    """)


def make_pad():
    root.importMFNodeFromString(-1, f"""
    DEF place_pad Solid {{
      translation {PLACE_XY_W[0]} {PLACE_XY_W[1]} {TABLE_H + 0.001}
      children [
        Shape {{
          geometry Box {{ size 0.12 0.12 0.002 }}
          appearance PBRAppearance {{ baseColor 0.1 0.8 0.2 roughness 0.8 }}
        }}
      ]
    }}
    """)


def ensure_object(def_name, expected_xy, creator):
    node = robot.getFromDef(def_name)
    if node is None:
        creator()
        print(f"  ({def_name} was not in the world, created)")
        return
    p = node.getPosition()
    if abs(p[0] - expected_xy[0]) > 0.02 or abs(p[1] - expected_xy[1]) > 0.02:
        print(f"  WARNING: {def_name} is at ({p[0]:.2f}, {p[1]:.2f}), code expects "
              f"({expected_xy[0]:.2f}, {expected_xy[1]:.2f}). "
              f"Fix TABLE_A/TABLE_B/PLACE_XY_W or move the node.")


ensure_object("table_pick", TABLE_A, lambda: add_table("table_pick", TABLE_A, "0.55 0.45 0.35"))
ensure_object("table_place", TABLE_B, lambda: add_table("table_place", TABLE_B, "0.50 0.50 0.50"))
ensure_object("place_pad", PLACE_XY_W, make_pad)

box_node = robot.getFromDef("pick_box")  # an existing cube in the world is kept


def truth_pose():
    """Cube pose from the simulator. Used only for calibration and error checks."""
    pos = np.array(box_node.getPosition())
    Rm = np.array(box_node.getOrientation()).reshape(3, 3)
    return pos, float(np.arctan2(Rm[1, 0], Rm[0, 0]))


def spawn_cube(x=None, y=None, yaw_=None):
    """Put the cube on table A; random pose if no arguments are given."""
    global box_node
    if x is None:
        x, y = random.uniform(*SPAWN_X), random.uniform(*SPAWN_Y)
    if yaw_ is None:
        yaw_ = random.uniform(-0.7, 0.7) if SPAWN_RANDOM_YAW else 0.0
    z = TABLE_H + BOX_SIZE / 2 + 0.002

    if box_node is None:
        root.importMFNodeFromString(-1, f"""
        DEF pick_box Solid {{
          translation {x:.3f} {y:.3f} {z}
          rotation 0 0 1 {yaw_:.3f}
          children [
            Shape {{
              geometry Box {{ size {BOX_SIZE} {BOX_SIZE} {BOX_SIZE} }}
              appearance PBRAppearance {{ baseColor 0.1 0.4 1.0 roughness 0.5 }}
            }}
          ]
          contactMaterial "box_mat"
          boundingObject Box {{ size {BOX_SIZE} {BOX_SIZE} {BOX_SIZE} }}
          physics Physics {{ density -1 mass 0.2 }}
        }}
        """)
        box_node = robot.getFromDef("pick_box")
    else:
        box_node.getField("translation").setSFVec3f([x, y, z])
        box_node.getField("rotation").setSFRotation([0, 0, 1, yaw_])
        box_node.resetPhysics()

    for _ in range(40):
        robot.step(timestep)


# if the user already placed a cube on table A, remember it and restore it after calibration
user_cube = None
if box_node is not None:
    p0, y0 = truth_pose()
    if on_table(p0, TABLE_A) and abs(p0[2] - (TABLE_H + BOX_SIZE / 2)) < 0.03:
        user_cube = (p0.copy(), y0)
        print(f"Found user cube on table A: x={p0[0]:.3f} y={p0[1]:.3f}, will restore it after calibration")

# ---------- camera ----------
cam_node = robot.getFromDef("overhead_cam")
cam_created = cam_node is None
if cam_created:
    root.importMFNodeFromString(-1, f"""
    DEF overhead_cam Robot {{
      translation {CAM_POS[0]} {CAM_POS[1]} {CAM_POS[2]}
      rotation {CAM_CANDIDATES[0][1]}
      supervisor TRUE
      controller "camera_node"
      children [
        Camera {{ name "overhead_cam" width {CAM_RES} height {CAM_RES} fieldOfView {CAM_FOV} }}
      ]
    }}
    """)
    cam_node = robot.getFromDef("overhead_cam")
    cam_node.getField("translation").setSFVec3f(list(CAM_POS))
else:
    cp = cam_node.getPosition()
    print(f"Camera from world file: ({cp[0]:.2f}, {cp[1]:.2f}, {cp[2]:.2f})")


def read_det():
    try:
        s = cam_node.getField("customData").getSFString()
    except Exception:
        return None
    if not s:
        return None
    try:
        return json.loads(s)
    except ValueError:
        return None


def wait_fresh(t_min, max_steps=120):
    """Wait for a camera frame captured no earlier than t_min (simulation time)."""
    for _ in range(max_steps):
        d = read_det()
        if d is not None and d["t"] >= t_min:
            return d
        robot.step(timestep)
    return None


def orient_camera():
    """Pick the camera rotation that looks straight down and sees the cube."""
    rot_field = cam_node.getField("rotation")
    current = rot_field.getSFRotation()
    candidates = [("from world file", " ".join(f"{v:.5f}" for v in current))] + CAM_CANDIDATES

    best = None
    for k, (name, rot) in enumerate(candidates):
        rot_field.setSFRotation([float(v) for v in rot.split()])
        d = wait_fresh(robot.getTime() + 0.25, 500 if k == 0 else 120)
        area = d["det"]["area"] if (d and d["det"]) else 0.0
        print(f"  camera variant '{name}': blue blob {area:.0f} px^2")
        if best is None or area > best[0]:
            best = (area, rot, name)
        if area > 300:
            break

    if best[0] < 300:
        return False

    rot_field.setSFRotation([float(v) for v in best[1].split()])
    print(f"  camera looks down ({best[2]})")
    if best[2] != "from world file":
        print(f"  Tip: set rotation {best[1]} on overhead_cam in the world file to skip this search")
    return True


def calibrate_camera():
    """Fit a homography from 9 known cube positions, then verify on random poses."""
    uv, xy = [], []
    for x, y in CAL_POINTS:
        spawn_cube(x, y, 0.0)
        d = wait_fresh(robot.getTime() + 0.15, 80)
        if d is None or d["det"] is None:
            print(f"  WARNING: camera does not see the cube at ({x:.2f}, {y:.2f})")
            continue
        tp, _ = truth_pose()
        uv.append((d["det"]["cx"], d["det"]["cy"]))
        xy.append((tp[0], tp[1]))

    if len(uv) < 6:
        return None

    H = fit_homography(uv, xy)
    sq = [np.hypot(*(np.array(apply_h(H, *a)) - np.array(b))) ** 2 for a, b in zip(uv, xy)]
    print(f"  calibration on {len(uv)} points: RMS = {np.sqrt(np.mean(sq)) * 1000:.2f} mm")

    pos_errs, yaw_errs = [], []
    for _ in range(8):
        spawn_cube(random.uniform(*SPAWN_X), random.uniform(*SPAWN_Y), random.uniform(-0.8, 0.8))
        d = wait_fresh(robot.getTime() + 0.15, 80)
        if d is None or d["det"] is None:
            continue
        pos, ya = pose_from_detection(H, d["det"], TABLE_H + BOX_SIZE / 2)
        tp, ty = truth_pose()
        pos_errs.append(np.hypot(pos[0] - tp[0], pos[1] - tp[1]) * 1000)
        yaw_errs.append(abs(np.degrees(yaw_diff(ya, ty))))

    if pos_errs:
        print(f"  check on {len(pos_errs)} random poses: position mean {np.mean(pos_errs):.2f} mm "
              f"(max {np.max(pos_errs):.2f}), angle mean {np.mean(yaw_errs):.1f} deg "
              f"(max {np.max(yaw_errs):.1f})")
        if np.max(pos_errs) > 8:
            print("  WARNING: error > 8 mm, grasp may be inaccurate")
    return H


print("\nCamera setup")
spawn_cube(0.44, 0.0, 0.0)  # cube in the middle of table A to pick the camera orientation
if not orient_camera():
    raise RuntimeError(
        "Camera does not see the cube. Check: 1) controllers/camera_node/camera_node.py exists; "
        "2) the console shows '[camera_node] started'; 3) opencv-python is installed.")

H_cam = calibrate_camera()
if H_cam is None:
    raise RuntimeError("Camera calibration failed: cube was seen in fewer than 6 points.")

if user_cube is not None:
    spawn_cube(user_cube[0][0], user_cube[0][1], user_cube[1])
else:
    spawn_cube()
print("Camera ready\n")


# ---------- waypoint follower ----------
class Follower:
    def __init__(self):
        self.wps = []
        self.i = 0
        self.t = 0.0
        self.t_max = 0.0
        self.q_cmd = None
        self.speed = 0.5

    def start(self, wps, speed):
        self.wps = [np.array(w, float) for w in wps]
        self.speed, self.i, self.t = speed, 0, 0.0
        self.q_cmd = read_q()

        length, prev = 0.0, self.q_cmd
        for w in self.wps:
            length += np.max(np.abs(w - prev))
            prev = w
        self.t_max = length / speed * 1.5 + 6.0

        for m in motors:
            m.setVelocity(MOTOR_VEL)

    def step(self):
        """Returns True when the segment is finished (or timed out)."""
        self.t += DT
        if self.i < len(self.wps):
            ramp = min(1.0, 0.25 + self.t / 0.8)
            limit = self.speed * ramp * DT
            delta = self.wps[self.i] - self.q_cmd
            biggest = np.max(np.abs(delta))
            if biggest <= limit:
                self.q_cmd = self.wps[self.i].copy()
                self.i += 1
            else:
                self.q_cmd = self.q_cmd + delta * (limit / biggest)
            for m, a in zip(motors, self.q_cmd):
                m.setPosition(float(a))
            return False

        if np.max(np.abs(read_q() - self.wps[-1])) < 0.03:
            return True
        if self.t > self.t_max:
            print("  WARNING: timeout, arm did not reach the end of the segment")
            return True
        return False


follower = Follower()

# ---------- top-view debug window ----------
IMG, SPAN = 512, 0.8
vis_pos = None


def to_px(x, y):
    return int((x + SPAN) / (2 * SPAN) * IMG), int((SPAN - y) / (2 * SPAN) * IMG)


def show(label):
    if cv2 is None:
        return
    box_w, ee_w = np.array(box_node.getPosition()), tool_world()
    img = np.full((IMG, IMG, 3), 40, np.uint8)

    for c, col in ((TABLE_A, (90, 110, 140)), (TABLE_B, (140, 110, 90))):
        a = to_px(c[0] - TABLE_SIZE[0] / 2, c[1] + TABLE_SIZE[1] / 2)
        b = to_px(c[0] + TABLE_SIZE[0] / 2, c[1] - TABLE_SIZE[1] / 2)
        cv2.rectangle(img, a, b, col, -1)

    cv2.circle(img, to_px(0, 0), 12, (200, 200, 200), -1)

    px, py = to_px(*PLACE_XY_W)
    h = int(0.06 / (2 * SPAN) * IMG)
    cv2.rectangle(img, (px - h, py - h), (px + h, py + h), (0, 180, 0), -1)

    # blue square: cube from the simulator; white cross: cube position from the camera
    bx, by = to_px(box_w[0], box_w[1])
    hb = int(BOX_SIZE / 2 / (2 * SPAN) * IMG)
    cv2.rectangle(img, (bx - hb, by - hb), (bx + hb, by + hb), (255, 80, 20), -1)

    if vis_pos is not None:
        cv2.drawMarker(img, to_px(vis_pos[0], vis_pos[1]), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.circle(img, to_px(ee_w[0], ee_w[1]), 6, (0, 255, 255), -1)
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Top View", img)
    cv2.waitKey(1)


# ---------- state machine ----------
M_OPEN = "Opening gripper"
M_HOVER = "Moving above cube"
M_DESCEND = "Descending to cube"
M_GRASP = "GRASP"
M_LIFT = "Lifting cube"
M_TRANSFER = "Carrying to table B"
M_PLACE = "Lowering onto table B"
M_RELEASE = "RELEASE"
M_RETREAT = "Retreating up"


def make_cycle(segs):
    s = {n: (pts, v) for n, pts, v in segs}
    return [("open", None, M_OPEN),
            ("path", s["ABOVE_PICK"], M_HOVER),
            ("path", s["PICK"], M_DESCEND),
            ("close", None, M_GRASP),
            ("path", s["LIFT"], M_LIFT),
            ("path", s["TRANSFER"], M_TRANSFER),
            ("path", s["PLACE"], M_PLACE),
            ("open", None, M_RELEASE),
            ("path", s["RETREAT"], M_RETREAT)]


phase = "HOMING"
states, idx = [], 0
steps_in_state, grip_still, last_fp = 0, 0, 0.0
plan_pose, plan_area, cur_segs, done_count = None, 0.0, {}, 0
last_t, last_pos, last_yaw = -1.0, np.zeros(3), 0.0
still, tried, wait_msg, missing = 0, None, None, 0
last_guard_t, shift_cnt = -1.0, 0


def begin(i):
    global steps_in_state, grip_still, last_fp
    kind, data, msg = states[i]
    steps_in_state, grip_still, last_fp = 0, 0, finger_pos()
    print(f"[#{done_count + 1}] {msg}")
    if kind == "path":
        follower.start(*data)
    elif kind == "open":
        set_gripper(GRIPPER_OPEN, 0.8)
    elif kind == "close":
        set_gripper(GRIPPER_CLOSED, 0.4)


def go_wait():
    """Return to the waiting pose, then look for the cube with the camera."""
    global phase, tried, still, wait_msg, missing, last_t
    tried, still, wait_msg, missing, last_t = None, 0, None, 0, -1.0
    phase = "HOMING"
    follower.start([HOME_ANGLES], V_HOME)


def run_states(new_states, start=0):
    global states, idx, phase
    states, idx, phase = new_states, start, "RUN"
    begin(idx)


def say_once(key, text):
    global wait_msg
    if wait_msg != key:
        print(text)
    wait_msg = key


def handle_wait():
    """Look for the cube using the camera only. The cube can be moved at any time."""
    global last_t, last_pos, last_yaw, still, tried, missing, vis_pos
    global plan_pose, plan_area, cur_segs, shift_cnt, last_guard_t

    d = read_det()
    if d is None:
        say_once("nocam", "  WARNING: camera is silent (check controllers/camera_node/)")
        return
    if d["t"] == last_t:
        return
    last_t = d["t"]

    det = d["det"]
    if det is None:
        missing += 1
        still = 0
        vis_pos = None
        if missing >= 45:
            if AUTO_RESPAWN:
                print("\nCamera sees no cube on table A, spawning a new one")
                missing = 0
                spawn_cube()
            else:
                say_once("none", "Camera sees no cube on table A, put one there (translation of pick_box)")
        return
    missing = 0

    pos, cyaw = pose_from_detection(H_cam, det, TABLE_H + BOX_SIZE / 2)
    vis_pos = pos
    if not on_table(pos, TABLE_A, 0.03):
        say_once("off", "  Cube is outside table A, move it back")
        return

    moved = (np.linalg.norm(pos[:2] - last_pos[:2]) > 0.003 or
             abs(yaw_diff(cyaw, last_yaw)) > 0.03)
    last_pos, last_yaw = pos, cyaw
    still = 0 if moved else still + 1
    if still < 6:  # wait until the cube settles (~0.4 s)
        return

    if (tried is not None and np.linalg.norm(pos[:2] - tried[0][:2]) < 0.004
            and abs(yaw_diff(cyaw, tried[1])) < 0.02):
        return  # this pose was already tried
    tried = (pos.copy(), cyaw)

    tp, ty = truth_pose()
    print(f"\nCamera found cube: x={pos[0]:.3f} y={pos[1]:.3f}, "
          f"yaw {np.degrees(yaw_diff(cyaw, 0)):.0f} deg, area {det['area']:.0f} px^2")
    print(f"  (sim check: off by {np.hypot(pos[0] - tp[0], pos[1] - tp[1]) * 1000:.1f} mm, "
          f"{abs(np.degrees(yaw_diff(cyaw, ty))):.1f} deg)")

    print("Planning route")
    segs = plan_route(pos, cyaw, read_q())
    if segs is None:
        say_once(("unreach", round(pos[0], 2), round(pos[1], 2)),
                 "  Cube is out of reach. Move it closer to the middle of table A "
                 "(x 0.34..0.54, |y| <= 0.18), I will retry on my own.")
        return

    plan_pose, plan_area = (pos.copy(), cyaw), det["area"]
    cur_segs = {n: (p, v) for n, p, v in segs}
    shift_cnt, last_guard_t = 0, -1.0
    run_states(make_cycle(segs))


def guard_shift():
    """While the arm approaches, watch the cube: if it was moved, abort and replan."""
    global last_guard_t, shift_cnt
    d = read_det()
    if d is None or d["t"] == last_guard_t:
        return False
    last_guard_t = d["t"]

    det = d["det"]
    if det is None or det["area"] < 0.8 * plan_area:  # cube occluded by the arm, ignore
        return False

    pos, _ = pose_from_detection(H_cam, det, TABLE_H + BOX_SIZE / 2)
    if np.linalg.norm(pos[:2] - plan_pose[0][:2]) > 0.006:
        shift_cnt += 1
    else:
        shift_cnt = 0
    return shift_cnt >= 6


def handle_run():
    global idx, steps_in_state, grip_still, last_fp, done_count

    kind, data, msg = states[idx]
    steps_in_state += 1

    if kind == "path" and msg == M_HOVER and guard_shift():
        print("  Cube was moved, going back to the home pose and replanning")
        go_wait()
        return

    if kind == "path":
        done = follower.step()
    elif kind == "open":
        done = steps_in_state >= 35
    else:
        fp = finger_pos()
        grip_still = grip_still + 1 if abs(fp - last_fp) / DT < 0.03 else 0
        last_fp = fp
        done = (steps_in_state >= 15 and grip_still >= 10) or steps_in_state >= 160

    if not done:
        return
    for _ in range(15):  # short pause after each step
        robot.step(timestep)

    if msg == M_DESCEND:
        t, (tp, _) = tool_world(), truth_pose()
        print(f"  Sim check: XY miss {np.linalg.norm(t[:2] - tp[:2]) * 1000:.1f} mm, "
              f"flange above cube center {t[2] - tp[2]:.3f} m (set: {TOOL_LENGTH:.3f})")
        print("  Look at the fingers: if they are above the cube middle, decrease TOOL_LENGTH; "
              "if they touch the table, increase it.")

    if msg == M_GRASP:
        fp = finger_pos()
        held = fp < HELD_MAX
        print(f"  finger angle = {fp:.2f} rad -> " +
              ("cube is held" if held else "MISS: fingers closed on nothing"))
        if not held:
            run_states([("open", None, "Releasing (miss)"),
                        ("path", cur_segs["LIFT"], "Going up after miss")])
            return

    if msg in (M_LIFT, M_TRANSFER):
        fp = finger_pos()
        if fp >= HELD_MAX:
            print(f"  Cube dropped (finger angle {fp:.2f} rad)")
            run_states([("open", None, "Releasing (dropped)")])
            return
        print(f"  Cube in gripper (finger angle {fp:.2f} rad)")

    if msg == M_RETREAT:
        tp, _ = truth_pose()
        good = on_table(tp, TABLE_B) and abs(tp[2] - (TABLE_H + BOX_SIZE / 2)) < 0.02
        if good:
            done_count += 1
        print(f"  {'Placed on table B' if good else 'Cube is NOT on table B'} "
              f"(sim check): {np.round(tp, 3).tolist()}")

    idx += 1
    if idx >= len(states):
        go_wait()
    else:
        begin(idx)


print("Ready. You can move the cube at any time (translation of pick_box), "
      "the camera will pick it up and the route will be rebuilt.")
go_wait()

while robot.step(timestep) != -1:
    show(phase + (": " + states[idx][2] if phase == "RUN" else ""))
    if phase == "HOMING":
        if follower.step():
            phase = "WAIT"
    elif phase == "WAIT":
        handle_wait()
    else:
        handle_run()
