"""
demo_video.py - run the engine on a video file (or webcam) and watch it.

    python demo_video.py --source videos\\clip1.mp4
    python demo_video.py --source 0                 (webcam)
    python demo_video.py --source videos\\clip1.mp4 --out result.mp4 --no-show

Prints one JSON line every time a NEW alert starts. Press q to quit.
This file only DRAWS. All detection logic is in waste_engine.py.
"""
import argparse
import json

import cv2

from waste_engine import WasteDetector

COLORS = {                         # BGR
    "waste_pile": (0, 140, 255),   # orange
    "waste_spilled": (60, 60, 255),  # red
    "waste_Overload": (255, 0, 200),  # purple
    "Container": (255, 200, 0),    # light blue
}
NICE = {"waste_pile": "Waste pile", "waste_spilled": "Spilled waste",
        "waste_Overload": "Overflowing bin", "Container": "Bin"}


def draw(frame, result):
    for d in result["detections"]:
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = COLORS.get(d["type"], (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if d["is_waste"] else 2)
        text = f'{NICE.get(d["type"], d["type"])} {int(d["confidence"] * 100)}%'
        cv2.putText(frame, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if result["alerts_active"]:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
        label = "WASTE DETECTED: " + ", ".join(NICE[t] for t in result["alerts_active"])
        cv2.putText(frame, label, (15, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="video file, or 0 for the webcam")
    ap.add_argument("--camera-id", default="cam0")
    ap.add_argument("--out", default="", help="optional: save the annotated video here")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--width", type=int, default=960, help="shrink frames to this width")
    args = ap.parse_args()

    detector = WasteDetector()
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    scale = min(1.0, args.width / int(cap.get(3)))
    W, H = int(cap.get(3) * scale), int(cap.get(4) * scale)
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if args.out else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (W, H))

        result = detector.process_frame(frame, args.camera_id)
        if result["new_alerts"]:
            print(json.dumps({k: result[k] for k in
                              ("camera_id", "frame_index", "timestamp", "new_alerts", "detections")}))

        draw(frame, result)
        if out:
            out.write(frame)
        if not args.no_show:
            cv2.imshow("Waste detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
