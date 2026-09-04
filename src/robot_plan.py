from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CartesianStep:
    name: str
    x_mm: float
    y_mm: float
    z_mm: float
    action: str = "move"


def make_roll_placement_plan(target_mm: tuple[float, float], pickup_mm: tuple[float, float] = (80.0, -120.0), pick_z_mm: float = 15.0, place_z_mm: float = 15.0, approach_z_mm: float = 80.0) -> list[CartesianStep]:
    """Create a simulation-only pick/place sequence in the calibrated table frame."""
    tx, ty = target_mm
    px, py = pickup_mm
    return [
        CartesianStep("home", 0.0, 0.0, approach_z_mm),
        CartesianStep("approach_pick", px, py, approach_z_mm),
        CartesianStep("pick", px, py, pick_z_mm),
        CartesianStep("close_gripper", px, py, pick_z_mm, "close_gripper"),
        CartesianStep("lift_from_pick", px, py, approach_z_mm),
        CartesianStep("approach_place", tx, ty, approach_z_mm),
        CartesianStep("place", tx, ty, place_z_mm),
        CartesianStep("open_gripper", tx, ty, place_z_mm, "open_gripper"),
        CartesianStep("retreat", tx, ty, approach_z_mm),
        CartesianStep("home", 0.0, 0.0, approach_z_mm),
    ]


def plan_as_dicts(plan: list[CartesianStep]) -> list[dict]:
    return [asdict(step) for step in plan]
