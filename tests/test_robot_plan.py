from src.robot_plan import make_roll_placement_plan


def test_roll_plan_has_safe_pick_and_place_sequence():
    plan = make_roll_placement_plan((210.0, 95.0))
    assert [step.name for step in plan] == ["home", "approach_pick", "pick", "close_gripper", "lift_from_pick", "approach_place", "place", "open_gripper", "retreat", "home"]
    assert plan[5].z_mm == 80.0
    assert (plan[6].x_mm, plan[6].y_mm) == (210.0, 95.0)
