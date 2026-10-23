"""
api.py - a small web API around the engine (for the integration team).

Start it:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000

Call it (one frame per request, in order, per camera):
    curl -X POST http://localhost:8000/detect -F "image=@frame.jpg" -F "camera_id=cam-12"

Settings can be changed with environment variables (all optional):
    ALERT_AFTER   frames in a row before an alert starts   (default 8)
    FRAME_STRIDE  run the models every Nth frame           (default 1 here, i.e. every frame)
    PILE_CONF     confidence bar for the pile model        (default 0.35)
    MISS_TOLERANCE missed frames allowed before reset      (default 0)

NOTE: "alert_after" counts the frames THIS API RECEIVES. If a camera sends
1 picture per second, ALERT_AFTER=3 means "waste seen for 3 seconds".
"""
import os
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from waste_engine import WasteDetector

app = FastAPI(title="UrbanVision waste detection")

_lock = threading.Lock()
_detector = None


def _config():
    cfg = {"frame_stride": 1}
    for env, key, cast in (("ALERT_AFTER", "alert_after", int),
                           ("FRAME_STRIDE", "frame_stride", int),
                           ("PILE_CONF", "pile_conf", float),
                           ("MISS_TOLERANCE", "miss_tolerance", int)):
        if os.environ.get(env):
            cfg[key] = cast(os.environ[env])
    return cfg


def get_detector():
    global _detector
    if _detector is None:
        _detector = WasteDetector(_config())
    return _detector


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
def detect(image: UploadFile = File(...), camera_id: str = Form("cam0")):
    data = image.file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read the image")
    with _lock:                       # one frame at a time (counters are per camera)
        return get_detector().process_frame(frame, camera_id=camera_id)


@app.post("/reset")
def reset(camera_id: str = Form("")):
    with _lock:
        get_detector().reset(camera_id or None)
    return {"status": "reset", "camera_id": camera_id or "all"}
