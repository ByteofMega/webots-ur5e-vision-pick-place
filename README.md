# Vision-Based Pick & Place with UR5e in Webots

A UR5e arm with a Robotiq 2F-85 gripper picks a blue cube from one table and places it on another.
The cube is located only by an overhead camera. Simulator ground truth is used for calibration and
for printing error checks, never for deciding where to grasp.

## Features

- HSV segmentation and `minAreaRect` pose estimation in a separate camera node
- Pixel-to-world homography fitted at startup from 9 known cube positions, then validated on random poses
- Custom forward kinematics and damped least-squares IK with multi-seed search and joint-limit checks
- Smooth Cartesian paths: vertical approach, arc around the base axis, vertical placement
- State machine with grasp verification via finger angle
- Automatic replanning if the cube is moved while the arm is approaching

## How it works

1. `camera_node` (a separate Webots Robot) segments the blue cube in HSV space, finds its contour and
   `minAreaRect`, and publishes `{t, w, h, det}` as JSON in its own `customData` field.
2. `fk_ik_controller` fits a pixel-to-world homography at startup by placing the cube at 9 known
   points, then validates it on 8 random poses and prints mean and max error in mm and degrees.
3. The arm plans a route with DH-based forward kinematics and damped least-squares IK:
   approach above the cube, vertical descent, lift, arc around the base axis, vertical placement, retreat.
4. A state machine runs the cycle, checks the grasp via finger angle, and replans if the cube is moved
   while the arm is approaching.

## Requirements

- Webots R2025 (with the UR5e and Robotiq 2F-85 PROTOs)
- Python 3.9+ and the packages from `requirements.txt`

```bash
pip install -r requirements.txt
```

Make sure `opencv-python` and `numpy` are installed for the same Python interpreter that Webots uses
(Webots > Preferences > Python command).

## Project layout

```
controllers/
  camera_node/
    camera_node.py          # overhead camera, cube detection
  fk_ik_controller/
    fk_ik_controller.py     # arm controller: calibration, IK, planning, state machine
worlds/
  pick_place.wbt            # world file
requirements.txt
README.md
```

Webots requires the controller folder name to match the script name.

## World setup

The world needs:

- a UR5e with a Robotiq 2F-85 in its `toolSlot`, with `supervisor TRUE` and
  `controller "fk_ik_controller"`;
- optionally, nodes with these `DEF` names: `table_pick`, `table_place`, `place_pad`, `pick_box`,
  `overhead_cam`. Any that are missing are created by the controller at startup.

If you define `overhead_cam` yourself, it must be a `Robot` with `supervisor TRUE`,
`controller "camera_node"` and a `Camera` device named `overhead_cam`, placed above table A.

## Configuration

The main constants are at the top of `fk_ik_controller.py`:

| Constant | Meaning |
|---|---|
| `TABLE_A`, `TABLE_B` | Pick and place table centers (world XY, m) |
| `TOOL_LENGTH` | Flange to fingertip center, m. Tune it if the fingers grasp too high or too low |
| `SPAWN_X`, `SPAWN_Y` | Area where new cubes are spawned |
| `AUTO_RESPAWN` | Spawn a new cube if the camera sees none for about 3 s |
| `V_*` | Joint-space speeds of each motion segment |

HSV thresholds and the minimum blob area are at the top of `camera_node.py`.

## Usage

1. Open the world in Webots and press Run.
2. Watch the console: camera orientation search, calibration RMS, then the pick and place cycle.
3. Move `pick_box` anywhere on table A during the run. The route is rebuilt automatically.

## Troubleshooting

- **Camera does not see the cube**: check that `controllers/camera_node/camera_node.py` exists, that
  `[camera_node] started` appears in the console, and that `opencv-python` is installed.
- **Fingers grasp too high or hit the table**: adjust `TOOL_LENGTH`.
- **Cube out of reach**: keep it around x 0.34 to 0.54 and |y| <= 0.18 on table A.
- **Calibration error above 8 mm**: check the camera height and field of view.
