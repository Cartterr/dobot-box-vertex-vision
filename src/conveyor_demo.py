from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import BoxVertices, order_vertices


def candidate_vertices(frame: np.ndarray, roi: tuple[int, int, int, int], previous_center: np.ndarray | None) -> tuple[BoxVertices | None, float]:
    """Return the most plausible parcel quadrilateral in a configured conveyor ROI."""
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # Pale cardboard on wood has low-contrast rim edges; retain those instead
    # of using the stronger threshold intended for dark conveyor scenes.
    edges = cv2.morphologyEx(cv2.Canny(gray, 10, 40), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, BoxVertices]] = []
    minimum = crop.shape[0] * crop.shape[1] * 0.015
    maximum = crop.shape[0] * crop.shape[1] * 0.55
    for contour in contours:
        area = cv2.contourArea(contour)
        if not minimum <= area <= maximum:
            continue
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if min(width, height) < 35:
            continue
        points = cv2.boxPoints(rect) + np.array([x1, y1], dtype=np.float32)
        vertices = order_vertices(points)
        score = area
        if previous_center is not None:
            score /= 1.0 + float(np.linalg.norm(vertices.center - previous_center)) * 0.02
        candidates.append((score, vertices))
    if not candidates:
        return None, 0.0
    score, vertices = max(candidates, key=lambda item: item[0])
    confidence = min(0.99, score / (crop.shape[0] * crop.shape[1] * 0.12))
    return vertices, float(confidence)


def draw_overlay(frame: np.ndarray, vertices: BoxVertices | None, confidence: float, frame_number: int) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (width, 88), (18, 23, 31), -1)
    cv2.putText(canvas, "DOBOT VISION E2E  |  OPEN-TOP BOX DEMO", (28, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (68, 224, 255), 2)
    cv2.putText(canvas, "Vision -> rim vertices -> center candidate -> dry-run plan", (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (228, 234, 241), 1)
    cv2.rectangle(canvas, (width - 375, 105), (width - 24, 263), (18, 23, 31), -1)
    cv2.putText(canvas, "STATUS", (width - 350, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (68, 224, 255), 2)
    if vertices is None:
        cv2.putText(canvas, "Searching ROI...", (width - 350, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 190, 255), 2)
    else:
        labels = ("TL", "TR", "BR", "BL")
        polygon = vertices.points.astype(np.int32)
        cv2.polylines(canvas, [polygon], True, (65, 244, 140), 3)
        colors = ((255, 130, 54), (255, 224, 70), (90, 250, 130), (255, 82, 188))
        for label, point, color in zip(labels, polygon, colors):
            cv2.circle(canvas, tuple(point), 9, color, -1)
            cv2.putText(canvas, label, tuple(point + [12, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        center = vertices.center.astype(np.int32)
        cv2.drawMarker(canvas, tuple(center), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(canvas, f"VERTICES: LOCKED  {confidence:.0%}", (width - 350, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (65, 244, 140), 2)
        cv2.putText(canvas, f"CENTER PX: {center[0]}, {center[1]}", (width - 350, 204), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (228, 234, 241), 1)
        cv2.putText(canvas, "OPEN TOP: CANDIDATE", (width - 350, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 220, 140), 1)
    cv2.putText(canvas, f"frame {frame_number} | 36 mm roll target requires table-plane calibration before robot motion", (28, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 198, 92), 2)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a clearly labeled parcel-vertex conveyor demo.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("demo-output/conveyor-vertices-demo.mp4"))
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    input_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (1280, 720)
    # Tuned for an overhead open-box source: excludes peripheral props while
    # keeping the visible rim and its interior inside the search area.
    roi = (int(input_width * 0.22), int(input_height * 0.14), int(input_width * 0.75), int(input_height * 0.80))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    previous_center = None
    trace = []
    for index in range(int(args.seconds * fps)):
        ok, frame = capture.read()
        if not ok:
            break
        vertices, confidence = candidate_vertices(frame, roi, previous_center)
        if vertices is not None:
            previous_center = vertices.center
            trace.append({"frame": index, "confidence": round(confidence, 3), "vertices_px": vertices.as_dict()})
        writer.write(cv2.resize(draw_overlay(frame, vertices, confidence, index), output_size, interpolation=cv2.INTER_AREA))
    capture.release()
    writer.release()
    trace_path = args.output.with_suffix(".json")
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(args.output), "trace": str(trace_path), "tracked_frames": len(trace)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
