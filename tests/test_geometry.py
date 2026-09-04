import numpy as np

from src.geometry import BoxVertices, homography_from_calibration, order_vertices, project_points, roll_center_is_safe


def test_order_vertices_is_stable_for_scrambled_rectangle():
    vertices = order_vertices([[100, 0], [0, 50], [100, 50], [0, 0]])
    assert vertices.points.tolist() == [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]


def test_homography_projects_a_rectangle():
    matrix = homography_from_calibration([[0, 0], [100, 0], [100, 100], [0, 100]], [[10, 20], [210, 20], [210, 120], [10, 120]])
    projected = project_points(np.array([[50, 50]], dtype=np.float32), matrix)
    assert np.allclose(projected, [[110, 70]])


def test_roll_center_requires_radius_and_clearance():
    large = BoxVertices(np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32))
    small = BoxVertices(np.array([[0, 0], [30, 0], [30, 30], [0, 30]], dtype=np.float32))
    assert roll_center_is_safe(large, roll_diameter_mm=36, clearance_mm=2)
    assert not roll_center_is_safe(small, roll_diameter_mm=36, clearance_mm=2)
