# UrbanVision: Waste Detection Module (hand-off)

Detects waste on roads and around bins from a fixed camera (CCTV) view, and
tells you **when to raise an alert**. It is one Python class that takes a video
frame and returns JSON, so it can be plugged into a backend, a dashboard, or a
notification service.

## 1. What's in this folder

```
waste_handoff/
  waste_engine.py        <- THE MODULE. All detection + alert logic is here.
  demo_video.py          <- run it on a video / webcam and watch the result
  api.py                 <- optional web API (FastAPI) around the engine
  requirements.txt
  models/
    best_8.pt            <- road model   (copy it here, see section 3)
    best_pile_v2.pt      <- pile model   (copy it here)
  tests/
    test_engine_logic.py <- tests for the alert logic, need NO model files
  README.md
```

## 2. Quick start

Python **3.12** is recommended (Python 3.14 broke PyTorch on Windows). On
Windows also install the *Microsoft Visual C++ Redistributable (x64)*.

```
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install ultralytics opencv-python numpy

python tests\test_engine_logic.py                    # logic test, no models needed
python demo_video.py --source videos\clip1.mp4       # real test (needs the 2 model files)
```

## 3. Models

| File | Trained for | Classes (id: name) | How it is used |
|---|---|---|---|
| `models/best_8.pt` | Semarang CCTV waste data (YOLOv8s, 640 px, 7,627 images) | 0 Container, 1 waste_spilled, 2 waste_Overload | Container = normal bin (never an alert). waste_spilled / waste_Overload raise alerts. |
| `models/best_pile_v2.pt` | Waste piles on open roads | 0 *unknown*, 1 waste pile | **Only class id 1 is used**, as "waste_pile". Class 0 is ignored. |

Important notes:

* The engine uses the pile model's class **id** (`pile_class_id = 1`), not its
  name, so it works whether or not the file has the names `waste_pile_other` /
  `waste_pile`.
* The meaning of the pile model's class 1 was found **by testing on sample
  images**, not from the original `data.yaml`. Class 0 fired strongly on at
  least one test clip, and nobody has yet checked what it detects.
  **TODO:** find the training `data.yaml` for `best_pile_v2.pt` and fill in its
  dataset name, size and licence here.
* Road model data: *Semarang Sampah Dataset* ("CCTV Sampah", version 8),
  Roboflow Universe, licence **CC BY 4.0**. Credit it in any report or product.
* Two other models existed in the project and are **not used**: a material-type
  model (plastic / glass / metal / paper...), judged too unreliable on small CCTV
  crops, and an 8-class pile model (`best_pile_v5b.pt`) with unknown classes.

## 4. The logic, step by step

```
 video frame
     |
     +--> every Nth frame (frame_stride) run the models:
     |        |
     |        +--> ROAD model ----> Container ........ drawn only, never an alert
     |        |                     waste_spilled ..... kept if confidence >= 0.30
     |        |                     waste_Overload .... kept if confidence >= 0.35
     |        |
     |        +--> PILE model ----> class 1 (waste_pile) kept if confidence >= 0.35
     |
     +--> for each waste type (spilled / overflow / pile):
              present in this frame?  yes -> streak + 1
                                      no  -> streak = 0   (after miss_tolerance misses)
              streak >= alert_after?  yes -> ALERT ACTIVE
                                             "new_alerts" lists it on the first frame only
```

Why each step exists:

1. **Two models**: the road model knows bins and the waste around them
   (trained on CCTV scenes); the pile model recognises big heaps of waste on
   open roads, which the road model tends to miss.
2. **Container is never an alert**: a normal bin is not a problem. It is
   returned so a UI can draw it.
3. **Per-class confidence bars**: spilled waste gives weaker scores than bins,
   so each class has its own bar (`min_conf`, `pile_conf`).
4. **Persistence (`alert_after`)**: one noisy frame must not trigger an alarm.
   A waste type has to be seen in this many frames in a row.
5. **One alert per episode**: `new_alerts` is non-empty only on the frame an
   alert starts, so downstream systems get a single notification instead of one
   per frame. When the waste disappears, the alert ends and can fire again later.
6. **Per-camera state**: counters are kept per `camera_id`, so one detector
   object can serve many cameras.
7. **Frame stride**: models run on every 2nd frame by default for speed. The
   boxes from the last run are reused in between (`detections_fresh = false`).

## 5. Settings (`DEFAULT_CONFIG` in `waste_engine.py`)

| Setting | Default | Meaning |
|---|---|---|
| `pile_conf` | 0.35 | Confidence bar for the pile model |
| `min_conf.waste_spilled` | 0.30 | Bar for spilled waste (road model) |
| `min_conf.waste_Overload` | 0.35 | Bar for an overflowing bin |
| `road_floor_conf` | 0.25 | Lowest confidence the road model reports at all |
| `alert_after` | 8 | Frames in a row before an alert starts |
| `miss_tolerance` | 0 | Missed frames allowed before the count resets (raise to 2-3 if boxes flicker) |
| `frame_stride` | 2 | Run the models every Nth frame (higher = faster) |
| `imgsz` | 640 | Model input size |
| `pile_class_id` | 1 | Which class of the pile model counts as a pile |

Override like this: `WasteDetector({"alert_after": 15, "pile_conf": 0.45})`.

`alert_after` counts the frames the engine **receives**. A 30 fps video needs
about 8 frames (about 0.27 s). A camera that sends 1 picture per second should
use `alert_after = 3` (waste seen for 3 seconds).

## 6. Output format

`process_frame(frame, camera_id)` returns:

```json
{
  "camera_id": "cam-12",
  "frame_index": 42,
  "timestamp": "2026-10-19T06:30:12+00:00",
  "detections": [
    {"type": "waste_pile", "confidence": 0.69, "bbox": [120.0, 80.5, 610.2, 400.0],
     "source": "pile", "is_waste": true},
    {"type": "Container", "confidence": 0.72, "bbox": [300.1, 90.0, 380.4, 200.2],
     "source": "road", "is_waste": false}
  ],
  "detections_fresh": true,
  "waste_present": ["waste_pile"],
  "alerts_active": ["waste_pile"],
  "new_alerts": ["waste_pile"]
}
```

`bbox` is `[x1, y1, x2, y2]` in pixels of the frame that was passed in.
`type` is one of `Container`, `waste_spilled`, `waste_Overload`, `waste_pile`.
**Trigger notifications on `new_alerts`**, not on `alerts_active`.

## 7. Ways to integrate

**A. Import it (same Python process)**

```python
import cv2
from waste_engine import WasteDetector

detector = WasteDetector()
cap = cv2.VideoCapture("rtsp://camera-address/stream")   # or a file
while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = detector.process_frame(frame, camera_id="cam-12")
    if result["new_alerts"]:
        print("ALERT", result["new_alerts"])             # send it onward here
```

**B. Run it as a web service (any language can call it)**

```
pip install fastapi uvicorn python-multipart
uvicorn api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/detect -F "image=@frame.jpg" -F "camera_id=cam-12"
```

Endpoints: `GET /health`, `POST /detect` (image + camera_id, returns the JSON
above), `POST /reset` (clear one camera's counters). Send the frames of one
camera **in order**. The API runs the models on every frame and handles one
frame at a time. Environment variables `ALERT_AFTER`, `FRAME_STRIDE`,
`PILE_CONF`, `MISS_TOLERANCE` change the settings.

**C. Forward alerts to a municipality system**

```python
import requests
if result["new_alerts"]:
    requests.post(WEBHOOK_URL, timeout=5, json={
        "camera": result["camera_id"],
        "time": result["timestamp"],
        "types": result["new_alerts"],
        "detections": [d for d in result["detections"] if d["is_waste"]],
    })
```

Also save the frame (`cv2.imwrite`) when `new_alerts` is not empty, so a person
can see what triggered it.

## 8. Test results so far

Logic (no models): `python tests/test_engine_logic.py` runs 10 tests covering
alert start, one-alert-per-episode, thresholds, container, flicker tolerance,
frame stride, several cameras and JSON output. **All pass.**

Real videos (earlier settings: pile 0.45, spilled 0.40, overflow 0.35,
alert_after 8). Alert frames out of the most that were possible:

| Clip (stock footage, not CCTV) | Alert frames |
|---|---|
| Roadside waste heap | 410 / 443 |
| Large dump next to a road | 443 / 443 |
| Piled black bags | 246 / 246 |
| Overflowing street bin | 9 / 177 (mostly missed) |

On a laptop run, one clip showed no alert at first: the pile model's best
confidence was 0.69 but the clip flickered, and spilled waste peaked at 0.35,
below its bar. The current defaults (pile 0.35, spilled 0.30, alert_after 8) were
set after that and the alert then fired.

## 9. Known limitations and TODO before production

1. **No false-alarm test on clean roads yet.** The thresholds were lowered to
   catch more waste, which also raises false alarms. Run the demo on several
   clips with *no* waste (clean road, normal bin, people, vehicles, night) and
   raise the bars if it alerts.
2. **Tuned on a handful of clips.** There is no held-out evaluation of the
   combined system. The road model has its own validation / test scores from
   training (not included here: add `results.png`, `confusion_matrix.png` and
   the per-class mAP50 from the training run).
3. **Domain gap.** The road model learned Indonesian CCTV scenes (bins and
   collection points). Different bins, angles, lighting or camera heights
   (for example an overflowing street bin in a European setting) were mostly
   missed. For a new site, collect and label 200-300 frames from that camera
   and fine-tune.
4. **Static false boxes.** A confident wrong box on a fixed object (a tile, a
   board) will fire in every frame of that camera. Fix with extra "no waste"
   training frames, or an ignore-zone per camera.
5. **Pile model provenance** (section 3) and the meaning of its class 0 are
   unconfirmed.
6. **Speed.** On a CPU-only laptop expect a few frames per second. Use a larger
   `frame_stride`, a smaller `imgsz`, or a GPU for many cameras.
7. **The web API** was written and compile-checked but not run against real
   models yet. First thing to try on a new machine: `GET /health`, then one
   `POST /detect`.
8. **No alert de-duplication across cameras / places** and no storage or
   notification layer. That belongs to the integrating system.

## 10. Credit

Road model data: Semarang Sampah Dataset, "CCTV Sampah" v8, Roboflow Universe,
CC BY 4.0. Detector architecture: Ultralytics YOLOv8.
