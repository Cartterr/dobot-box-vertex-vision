from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class BoxVertices:
    """Rectangle corners ordered TL, TR, BR, BL in image or table coordinates."""

    points: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0)

    def as_dict(self) -> dict[str, list[float]]:
        names = ("top_left", "top_right", "bottom_right", "bottom_left")
        return {name: [round(float(x), 2), round(float(y), 2)] for name, (x, y) in zip(names, self.points)}


def order_vertices(points: Iterable[Iterable[float]]) -> BoxVertices:
    """Return four points in a stable TL, TR, BR, BL order."""
    pts = np.asarray(list(points), dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected four 2D points, got shape {pts.shape}")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return BoxVertices(ordered)


def find_box_vertices(image: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> BoxVertices | None:
    """Find the strongest rectangular contour, optionally inside an x1,y1,x2,y2 ROI."""
    offset = np.array([0, 0], dtype=np.float32)
    view = image
    if roi is not None:
        x1, y1, x2, y2 = [int(v) for v in roi]
        if x2 <= x1 or y2 <= y1:
            return None
        view = image[y1:y2, x1:x2]
        offset = np.array([x1, y1], dtype=np.float32)
    if view.size == 0:
        return None
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(500.0, view.shape[0] * view.shape[1] * 0.02)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        approximation = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            candidates.append((area, approximation.reshape(4, 2).astype(np.float32)))
    if not candidates:
        return None
    _, corners = max(candidates, key=lambda candidate: candidate[0])
    return order_vertices(corners + offset)


def homography_from_calibration(image_points_px: Iterable[Iterable[float]], table_points_mm: Iterable[Iterable[float]]) -> np.ndarray:
    src = order_vertices(image_points_px).points
    dst = order_vertices(table_points_mm).points
    matrix = cv2.getPerspectiveTransform(src, dst)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Invalid calibration points; choose four distinct table-plane points")
    return matrix


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)


def roll_center_is_safe(box_mm: BoxVertices, roll_diameter_mm: float, clearance_mm: float) -> bool:
    """Check whether the box center is at least radius + clearance from every edge."""
    required = roll_diameter_mm / 2.0 + clearance_mm
    center = box_mm.center
    polygon = box_mm.points.astype(np.float32)
    return cv2.pointPolygonTest(polygon, (float(center[0]), float(center[1])), True) >= required
