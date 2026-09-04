from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import BoxVertices, find_box_vertices, homography_from_calibration, project_points, roll_center_is_safe
from .robot_plan import make_roll_placement_plan, plan_as_dicts


def synthetic_camera_frame() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((720, 1280, 3), (42, 47, 53), dtype=np.uint8)
    workcell = np.array([[140, 100], [1140, 100], [1140, 650], [140, 650]], dtype=np.float32)
    box = np.array([[440, 250], [830, 220], [875, 510], [405, 545]], dtype=np.int32)
    cv2.fillConvexPoly(image, box, (205, 188, 140))
    cv2.polylines(image, [box], True, (248, 248, 248), 6)
    cv2.putText(image, "synthetic overhead camera", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    return image, workcell


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision-to-RoboDK demo. Never connects to a physical robot.")
    parser.add_argument("--simulate", action="store_true", help="Build and run the Dobot Magician station in RoboDK simulation mode")
    parser.add_argument("--save-image", type=Path, default=Path("demo-output/e2e-vision.png"))
    args = parser.parse_args()
    image, workcell_px = synthetic_camera_frame()
    vertices_px = find_box_vertices(image)
    if vertices_px is None:
        raise SystemExit("Synthetic box was not detected")
    homography = homography_from_calibration(workcell_px, [[0, 0], [300, 0], [300, 200], [0, 200]])
    vertices_mm = BoxVertices(project_points(vertices_px.points, homography))
    target = vertices_mm.center
    safe = roll_center_is_safe(vertices_mm, roll_diameter_mm=36.0, clearance_mm=2.0)
    if not safe:
        raise SystemExit("Detected box cannot safely hold a 36 mm roll with the configured clearance")
    plan = make_roll_placement_plan((float(target[0]), float(target[1])))
    canvas = image.copy()
    cv2.polylines(canvas, [vertices_px.points.astype(int)], True, (0, 255, 0), 3)
    cv2.circle(canvas, tuple(vertices_px.center.astype(int)), 9, (0, 80, 255), -1)
    cv2.putText(canvas, f"36 mm target: {target[0]:.1f}, {target[1]:.1f} mm", (24, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    args.save_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.save_image), canvas)
    payload = {"safe_for_36mm_roll": safe, "vertices_px": vertices_px.as_dict(), "target_mm": [round(float(value), 2) for value in target], "plan": plan_as_dicts(plan)}
    if args.simulate:
        from .robodk_sim import build_and_run_station
        payload["robodk_station"] = str(build_and_run_station(plan, Path(".cache")))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
