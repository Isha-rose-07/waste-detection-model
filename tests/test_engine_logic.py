"""
Tests for the alert logic. They use FAKE models, so no model files, GPU or
ultralytics are needed.

Run from the project folder:
    python tests/test_engine_logic.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from waste_engine import WasteDetector  # noqa: E402


class _Arr:
    def __init__(self, a):
        self.a = np.array(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, dets):
        n = len(dets)
        self.xyxy = _Arr([[10, 10, 50, 50]] * n if n else np.zeros((0, 4)))
        self.cls = _Arr([d[0] for d in dets])
        self.conf = _Arr([d[1] for d in dets])


class _Result:
    def __init__(self, dets):
        self.boxes = _Boxes(dets)


class FakeModel:
    """predict() returns the next scripted list of (class_id, confidence)
    and, like YOLO, drops anything below the `conf` argument."""

    def __init__(self, names, script):
        self.names = names
        self.script = script
        self.calls = 0

    def predict(self, frame, **kw):
        dets = self.script[self.calls % len(self.script)]
        self.calls += 1
        floor = kw.get("conf", 0.0)
        return [_Result([d for d in dets if d[1] >= floor])]


ROAD_NAMES = {0: "Container", 1: "waste_spilled", 2: "waste_Overload"}
PILE_NAMES = {0: "waste_pile_other", 1: "waste_pile"}
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def make(road_script, pile_script, **cfg):
    base = {"frame_stride": 1, "alert_after": 3}
    base.update(cfg)
    return WasteDetector(base, FakeModel(ROAD_NAMES, road_script), FakeModel(PILE_NAMES, pile_script))


def test_alert_starts_after_n_frames_and_fires_once():
    d = make([[]], [[(1, 0.8)]])
    r = [d.process_frame(FRAME) for _ in range(5)]
    assert [x["new_alerts"] for x in r] == [[], [], ["waste_pile"], [], []]
    assert [x["alerts_active"] for x in r] == [[], [], ["waste_pile"], ["waste_pile"], ["waste_pile"]]


def test_container_alone_never_alerts():
    d = make([[(0, 0.95)]], [[]])
    for _ in range(6):
        r = d.process_frame(FRAME)
    assert r["alerts_active"] == [] and r["waste_present"] == []
    assert r["detections"][0]["type"] == "Container" and r["detections"][0]["is_waste"] is False


def test_confidence_thresholds():
    # spilled needs 0.30 (default); 0.28 is ignored, 0.32 counts
    d = make([[(1, 0.28)]], [[]])
    assert d.process_frame(FRAME)["waste_present"] == []
    d = make([[(1, 0.32)]], [[]])
    assert d.process_frame(FRAME)["waste_present"] == ["waste_spilled"]
    # pile needs 0.35; the fake model drops 0.30 just like YOLO would
    d = make([[]], [[(1, 0.30)]])
    assert d.process_frame(FRAME)["waste_present"] == []
    d = make([[]], [[(1, 0.40)]])
    assert d.process_frame(FRAME)["waste_present"] == ["waste_pile"]


def test_pile_other_class_is_ignored():
    d = make([[]], [[(0, 0.9)]])
    assert d.process_frame(FRAME)["waste_present"] == []


def test_streak_resets_on_a_miss():
    # present, present, MISSING, present, present, present -> alert only at the end
    d = make([[]], [[(1, 0.8)], [(1, 0.8)], [], [(1, 0.8)], [(1, 0.8)], [(1, 0.8)]])
    r = [d.process_frame(FRAME) for _ in range(6)]
    assert [bool(x["new_alerts"]) for x in r] == [False, False, False, False, False, True]


def test_miss_tolerance_bridges_a_flicker():
    d = make([[]], [[(1, 0.8)], [(1, 0.8)], [], [(1, 0.8)]], miss_tolerance=1)
    r = [d.process_frame(FRAME) for _ in range(4)]
    assert r[3]["new_alerts"] == ["waste_pile"]


def test_alert_can_fire_again_in_a_new_episode():
    script = [[(1, 0.8)]] * 3 + [[]] + [[(1, 0.8)]] * 3
    d = make([[]], script)
    r = [d.process_frame(FRAME) for _ in range(7)]
    assert r[2]["new_alerts"] == ["waste_pile"]
    assert r[3]["alerts_active"] == []
    assert r[6]["new_alerts"] == ["waste_pile"]


def test_frame_stride_runs_models_every_nth_frame():
    d = make([[]], [[(1, 0.8)]], frame_stride=2, alert_after=4)
    r = [d.process_frame(FRAME) for _ in range(6)]
    assert d.pile.calls == 3
    assert [x["detections_fresh"] for x in r] == [True, False, True, False, True, False]
    assert r[3]["new_alerts"] == ["waste_pile"]   # 4 frames in a row


def test_cameras_are_independent():
    d = make([[]], [[(1, 0.8)]])
    for _ in range(3):
        a = d.process_frame(FRAME, "A")
    b = d.process_frame(FRAME, "B")
    assert a["alerts_active"] == ["waste_pile"] and b["alerts_active"] == []
    d.reset("A")
    assert d.process_frame(FRAME, "A")["frame_index"] == 0


def test_output_is_json_friendly():
    d = make([[(1, 0.9), (0, 0.8)]], [[(1, 0.7)]])
    out = json.loads(json.dumps(d.process_frame(FRAME, "cam-9")))
    for key in ("camera_id", "frame_index", "timestamp", "detections", "detections_fresh",
                "waste_present", "alerts_active", "new_alerts"):
        assert key in out
    assert set(out["detections"][0]) == {"type", "confidence", "bbox", "source", "is_waste"}


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\nAll {len(tests)} tests passed.")
