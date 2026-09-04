from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from robodk.robolink import ITEM_TYPE_ROBOT, RUNMODE_SIMULATE, Robolink
from robodk.robomath import transl

from .robot_plan import CartesianStep

DOBOT_MAGICIAN_MODEL_URL = "https://cdn.robodk.com/downloads-library/library-robots/Dobot-Magician.robot"


def robot_model(cache_dir: Path) -> Path:
    """Fetch the public RoboDK Dobot Magician model once into ignored cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "Dobot-Magician.robot"
    if not path.exists():
        urlretrieve(DOBOT_MAGICIAN_MODEL_URL, path)
    return path


def build_and_run_station(plan: list[CartesianStep], cache_dir: Path) -> Path:
    """Build and execute a station in RUNMODE_SIMULATE only; never connects to hardware."""
    RDK = Robolink()
    RDK.setRunMode(RUNMODE_SIMULATE)
    RDK.AddStation("Dobot 36mm Roll Vision Demo")
    robot = RDK.AddFile(str(robot_model(cache_dir).resolve()))
    if not robot.Valid() or robot.Type() != ITEM_TYPE_ROBOT:
        raise RuntimeError("RoboDK could not load the Dobot Magician model")
    # The table makes the calibrated camera coordinate system visible in the
    # station. Motion targets below are absolute simulator poses to preserve
    # the Magician model's default TCP orientation.
    table = RDK.AddFrame("VisionTable")
    table.setPose(robot.Pose())
    robot.setSpeed(100)
    robot.setSpeedJoints(30)
    targets = []
    home_pose = robot.Pose()
    for index, step in enumerate(plan):
        # This demo model has a different reachable envelope than the measured
        # table. Compress its camera-frame footprint around the loaded home
        # pose for visualization only; production must use a measured station.
        simulated_x = (step.x_mm - 150.0) * 0.35
        simulated_y = (step.y_mm - 100.0) * 0.35
        simulated_z = (step.z_mm - 80.0) * 0.25
        target = RDK.AddTarget(f"{index + 1:02d}_{step.name}")
        target.setAsCartesianTarget()
        target.setPose(home_pose * transl(simulated_x, simulated_y, simulated_z))
        targets.append((step, target))
    for step, target in targets:
        if step.action == "move":
            RDK.ShowMessage(f"SIMULATION: {step.name}", False)
            (robot.MoveJ if step.name in {"home", "approach_pick", "approach_place"} else robot.MoveL)(target)
        else:
            RDK.ShowMessage(f"SIMULATION: {step.action}", False)
    station = cache_dir / "dobot_36mm_roll_vision_demo.rdk"
    RDK.Save(str(station.resolve()))
    return station
