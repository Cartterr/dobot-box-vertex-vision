from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .geometry import BoxVertices, find_box_vertices, homography_from_calibration, project_points, roll_center_is_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect box vertices for Dobot 36 mm roll placement. Vision only; no robot motion.")
    parser.add_argument("--source", required=True, help="Image/video path or webcam index such as 0")
    parser.add_argument("--weights", help="Custom YOLO26 box detector weights")
    parser.add_argument("--no-yolo", action="store_true", help="Use contour extraction on the whole image")
    parser.add_argument("--calibration", type=Path, help="YAML with image_points_px and table_points_mm")
    parser.add_argument("--roll-diameter-mm", type=float, default=36.0)
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", type=Path, help="Save last annotated frame")
    return parser.parse_args()


def load_homography(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        calibration = yaml.safe_load(handle)
    return homography_from_calibration(calibration["image_points_px"], calibration["table_points_mm"])


def box_roi_from_yolo(model, frame: np.ndarray) -> tuple[int, int, int, int] | None:
    result = model.predict(frame, conf=0.50, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    names = result.names
    detections = []
    for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        if str(names[int(class_id)]).lower() == "box":
            detections.append((float(confidence), xyxy))
    if not detections:
        return None
    _, xyxy = max(detections, key=lambda item: item[0])
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(int)
    padding = 12
    return max(0, x1 - padding), max(0, y1 - padding), min(w, x2 + padding), min(h, y2 + padding)


def frame_payload(vertices_px: BoxVertices, homography: np.ndarray | None, roll_diameter_mm: float, clearance_mm: float) -> dict:
    payload = {
        "vertices_px": vertices_px.as_dict(),
        "center_px": [round(float(value), 2) for value in vertices_px.center],
        "roll_diameter_mm": roll_diameter_mm,
    }
    if homography is not None:
        box_mm = BoxVertices(project_points(vertices_px.points, homography))
        payload["vertices_mm"] = box_mm.as_dict()
        payload["target_mm"] = [round(float(value), 2) for value in box_mm.center]
        payload["safe_for_roll"] = roll_center_is_safe(box_mm, roll_diameter_mm, clearance_mm)
    return payload


def annotate(frame: np.ndarray, vertices: BoxVertices, payload: dict) -> np.ndarray:
    canvas = frame.copy()
    points = vertices.points.astype(int)
    cv2.polylines(canvas, [points], True, (0, 255, 0), 2)
    for label, point in zip(("TL", "TR", "BR", "BL"), points):
        cv2.circle(canvas, tuple(point), 5, (0, 80, 255), -1)
        cv2.putText(canvas, label, tuple(point + [6, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
    if "target_mm" in payload:
        text = f"target mm: {payload['target_mm']} | safe: {payload['safe_for_roll']}"
        cv2.putText(canvas, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if payload["safe_for_roll"] else (0, 0, 255), 2)
    return canvas


def main() -> int:
    args = parse_args()
    if not args.no_yolo and not args.weights:
        raise SystemExit("Provide --weights for a custom box model, or use --no-yolo for contour-only mode.")
    model = None
    if args.weights:
        from ultralytics import YOLO
        model = YOLO(args.weights)
    homography = load_homography(args.calibration)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    last_canvas = None
    while True:
        success, frame = capture.read()
        if not success:
            break
        roi = box_roi_from_yolo(model, frame) if model is not None else None
        vertices = find_box_vertices(frame, roi)
        if vertices is not None:
            payload = frame_payload(vertices, homography, args.roll_diameter_mm, args.clearance_mm)
            print(json.dumps(payload), flush=True)
            last_canvas = annotate(frame, vertices, payload)
        else:
            last_canvas = frame
        if args.show:
            cv2.imshow("Dobot box vertex vision - q to quit", last_canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if isinstance(source, str):
            break
    capture.release()
    cv2.destroyAllWindows()
    if args.save and last_canvas is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), last_canvas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
