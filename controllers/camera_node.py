# controllers/camera_node/camera_node.py
# Overhead camera: finds the blue cube and writes the result to this node's customData.
import json
import cv2
import numpy as np

HSV_LOW = (100, 150, 50)
HSV_HIGH = (130, 255, 255)
MIN_AREA = 150  # px^2, anything smaller is treated as noise
KERNEL = np.ones((3, 3), np.uint8)


def find_cube(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    biggest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(biggest))
    if area < MIN_AREA:
        return None, mask

    m = cv2.moments(biggest)
    corners = cv2.boxPoints(cv2.minAreaRect(biggest))
    result = {
        "cx": float(m["m10"] / m["m00"]),
        "cy": float(m["m01"] / m["m00"]),
        "area": area,
        "box": [[float(x), float(y)] for x, y in corners],
    }
    return result, mask


def draw_debug(frame, det, mask):
    img = frame.copy()
    if det:
        cv2.polylines(img, [np.array(det["box"], np.int32)], True, (0, 255, 0), 2)
        cv2.circle(img, (int(det["cx"]), int(det["cy"])), 4, (0, 0, 255), -1)
        text = f"({det['cx']:.0f}, {det['cy']:.0f}) S={det['area']:.0f}"
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(img, "no cube", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Overhead camera", cv2.resize(img, (480, 480)))
    cv2.imshow("Blue mask", cv2.resize(mask, (480, 480)))
    cv2.waitKey(1)


def main():
    from controller import Supervisor

    robot = Supervisor()
    dt = int(robot.getBasicTimeStep())

    cam = robot.getDevice("overhead_cam")
    cam.enable(2 * dt)
    w, h = cam.getWidth(), cam.getHeight()
    custom_data = robot.getSelf().getField("customData")
    print(f"[camera_node] started, {w}x{h}")

    tick = 0
    while robot.step(dt) != -1:
        tick += 1
        if tick % 2:  # process every second frame
            continue

        raw = cam.getImage()
        if raw is None:
            continue

        # BGRA -> BGR
        frame = np.frombuffer(raw, np.uint8).reshape((h, w, 4))[:, :, :3].copy()
        det, mask = find_cube(frame)

        custom_data.setSFString(json.dumps({"t": robot.getTime(), "w": w, "h": h, "det": det}))

        if tick % 6 == 0:
            try:
                draw_debug(frame, det, mask)
            except Exception:
                pass  # no display available, ignore


if __name__ == "__main__":
    main()
