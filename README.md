# Dobot Box Vertex Vision

YOLO26 + OpenCV pipeline for finding the four visible vertices of a placement box and calculating a safe center target for a **36 mm diameter roll**. It replaces ANPR/OCR with geometry that is useful for a Dobot pick-and-place workflow.

> Safety: this project only produces vision coordinates. It never moves a robot. Validate calibration, tool offsets, workspace limits, and clearance with the Dobot powered safely and in dry-run mode before sending any pose to a robot controller.

## How it works

1. A custom YOLO26 model detects the `box` and limits the image search area.
2. OpenCV extracts the largest rectangular contour inside that area.
3. Corners are returned in a stable order: `top_left`, `top_right`, `bottom_right`, `bottom_left`.
4. With a four-point calibration, the pixel vertices are projected into millimetres on the table plane.
5. The pipeline returns the box center and checks that a 36 mm roll center stays inside the box with a configurable clearance.

YOLO26 is used for localization because its official Python API supports custom detection models and live/image sources. The final sub-pixel-friendly vertices come from geometry, rather than forcing a detector to regress four separate points.

## Quick start (native Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m src.app --source 0 --weights models\box.pt --show
```

`models\box.pt` must be a YOLO26 model trained with one class named `box`. For a contour-only prototype without a trained model, pass `--no-yolo`.

```powershell
python -m src.app --source sample.jpg --no-yolo --save annotated.jpg
```

## Calibrating pixels to Dobot millimetres

Copy the example and replace the four pixel and physical table points using a fixed overhead camera and a flat calibration target:

```powershell
Copy-Item config\calibration.example.yaml config\calibration.yaml
python -m src.app --source 0 --weights models\box.pt --calibration config\calibration.yaml --show
```

The `table_points_mm` frame is your declared robot/table frame. Make its origin and axes match the frame used by your Dobot program, then apply your measured tool-center-point offset there. Do not infer an arm pose solely from camera pixels.

## Output contract

Each valid frame prints JSON like:

```json
{
  "vertices_px": {"top_left": [120.2, 80.1], "top_right": [520.3, 81.0], "bottom_right": [521.1, 380.5], "bottom_left": [119.4, 381.3]},
  "center_px": [320.3, 230.7],
  "roll_diameter_mm": 36.0,
  "target_mm": [210.4, 95.8],
  "safe_for_roll": true
}
```

`target_mm` appears only after calibration. `safe_for_roll` requires the center to have at least `18 mm + --clearance-mm` of room on every side of the detected quadrilateral.

## Training a box detector

Collect overhead images that represent real lighting, box materials, occlusions from the gripper and rolls, blur, and empty/failure states. Label each box with a single `box` bounding box, split train/validation images, then use Ultralytics' custom-detection flow:

```powershell
yolo detect train model=yolo26n.pt data=dataset.yaml epochs=100 imgsz=960
```

Use the resulting `runs\detect\train\weights\best.pt` as `--weights`. Keep a held-out set and measure vertex error in millimetres after calibration; detection mAP alone is not a robot-placement acceptance criterion.

## One-command E2E RoboDK demo

RoboDK is already installed on this PC. This local demo creates a synthetic overhead camera image, detects its box, projects it to a 300 × 200 mm table frame, validates a 36 mm roll margin, produces a pick/place plan, and animates it with RoboDK's Dobot Magician model.

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.demo --simulate
```

It saves the annotated vision frame to `demo-output/e2e-vision.png` and the station to `.cache/dobot_36mm_roll_vision_demo.rdk`. `--simulate` uses only RoboDK's `RUNMODE_SIMULATE`; it has no robot connection code. For a quick no-GUI validation, omit `--simulate`.

The plan prints the unmodified calibrated table coordinates. For visualization, the script scales that synthetic 300 × 200 mm table into the downloaded Magician model's reachable envelope around its home pose. It is intentionally not a real-cell transform. The example uses an 80 mm approach height, 15 mm pick/place height, and a fixed synthetic pickup point. Replace those only after measuring your fixture, TCP, rolls, camera calibration, robot reach, collision clearances, and emergency-stop process.

## Conveyor video overlay demo

Render a colorful, honest demonstration overlay on a conveyor clip:

```powershell
python -m src.conveyor_demo --source C:\path\to\open-top-box.mp4
```

The script marks the tracked quadrilateral's four ordered corners, confidence and pixel center, then saves MP4 + JSON trace to `demo-output`. The center is only a camera-space candidate until the source is calibrated to the real table plane; do not use it for robot motion before that step.

## Project layout

- `src/app.py` — camera/image CLI and JSON output
- `src/geometry.py` — vertex ordering, contour extraction, homography, and safety check
- `src/demo.py` — synthetic camera to safe 36 mm roll placement plan
- `src/robodk_sim.py` — simulation-only Dobot Magician station builder
- `config/calibration.example.yaml` — four-point table-plane calibration template
- `tests/test_geometry.py` — deterministic geometry tests

## License

MIT. Ultralytics software and model use are subject to their own licensing terms; review those terms before commercial deployment.
