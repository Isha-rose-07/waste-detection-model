import sys
import cv2
from ultralytics import YOLO

road = YOLO("models/best_8.pt")
pile = YOLO("models/best_pile_v2.pt")

cap = cv2.VideoCapture(sys.argv[1])
best = {}
count = {}
checked = 0
n = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if n % 5 == 0:
        checked += 1
        r = road.predict(frame, imgsz=640, conf=0.10, verbose=False)[0]
        for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = road.names[int(c)]
            best[name] = max(best.get(name, 0), s)
            count[name] = count.get(name, 0) + 1
        p = pile.predict(frame, imgsz=640, conf=0.10, verbose=False)[0]
        for c, s in zip(p.boxes.cls.tolist(), p.boxes.conf.tolist()):
            name = "pile:" + pile.names[int(c)]
            best[name] = max(best.get(name, 0), s)
            count[name] = count.get(name, 0) + 1
    n += 1

cap.release()
print("Frames checked:", checked)
print("\n".join(f"{k}: {count[k]} boxes, best confidence {best[k]:.2f}" for k in best) or "Nothing detected, even at the lowest setting.")