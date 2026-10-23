import argparse, base64, os, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

# ---------- settings ----------
ROAD_MODEL = "models/best_8.pt"
PILE_MODEL = "models/best_pile_v2.pt"

PILE_CLASS = 1
PILE_CONF = 0.35
MIN_CONF = {"waste_spilled": 0.30, "waste_Overload": 0.35}
ALERT_AFTER = 8        # frames in a row before the alert starts
POPUP_COOLDOWN = 30     # seconds before another popup can appear

NICE = {"waste_pile": "Waste pile", "waste_spilled": "Spilled waste",
        "waste_Overload": "Overflowing bin", "Container": "Bin"}
COLORS = {"waste_pile": (0, 140, 255),      # orange   (BGR colours)
          "waste_spilled": (60, 60, 255),   # red
          "waste_Overload": (255, 0, 200),  # purple
          "Container": (255, 200, 0)}       # light blue


# ---------- the popup window (runs as its own small program) ----------
def show_popup_window(kinds_text, image_path):
    import tkinter as tk
    kinds = [NICE.get(k, k) for k in kinds_text.split(",") if k]
    RED, YEL = "#d90429", "#ffbe0b"

    root = tk.Tk()
    root.overrideredirect(True)            # no title bar
    root.attributes("-topmost", True)      # always on top

    outer = tk.Frame(root, bg=RED, padx=8, pady=8)
    outer.pack()
    inner = tk.Frame(outer, bg="white")
    inner.pack()

    tk.Label(inner, text="\u26a0  WASTE DETECTED", bg=RED, fg="white",
             font=("Segoe UI", 24, "bold"), padx=30, pady=14).pack(fill="x")
    tk.Label(inner, text=" + ".join(kinds), bg="white", fg=RED,
             font=("Segoe UI", 20, "bold"), pady=10).pack()
    tk.Label(inner, text=datetime.now().strftime("%I:%M:%S %p    %d %b %Y"),
             bg="white", fg="#555555", font=("Segoe UI", 12)).pack()

    img = cv2.imread(image_path) if image_path and os.path.exists(image_path) else None
    if img is not None:
        h, w = img.shape[:2]
        img = cv2.resize(img, (420, int(h * 420 / w)))
        ok, buf = cv2.imencode(".png", img)
        photo = tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))
        pic = tk.Label(inner, image=photo, bg="white", bd=2, relief="solid")
        pic.image = photo
        pic.pack(pady=10, padx=20)

    tk.Label(inner, text="Please check the area and send a cleanup team.",
             bg="white", fg="#222222", font=("Segoe UI", 12)).pack()
    tk.Button(inner, text="OK, I WILL CHECK", command=root.destroy, bg=RED, fg="white",
              activebackground="#8d0116", activeforeground="white",
              font=("Segoe UI", 14, "bold"), relief="flat", padx=20, pady=8,
              cursor="hand2").pack(pady=14)

    # flashing red / yellow border
    state = {"on": True}
    def flash():
        outer.configure(bg=YEL if state["on"] else RED)
        state["on"] = not state["on"]
        root.after(350, flash)
    flash()

    # beeps
    def beeps():
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1400, 220)
                winsound.Beep(900, 220)
        except Exception:
            pass
    threading.Thread(target=beeps, daemon=True).start()

    # put it in the middle of the screen
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    x = (root.winfo_screenwidth() - w) // 2
    y = max(10, (root.winfo_screenheight() - h) // 3)
    root.geometry(f"+{x}+{y}")

    root.bind("<Return>", lambda e: root.destroy())
    root.bind("<Escape>", lambda e: root.destroy())
    root.after(20000, root.destroy)        # closes by itself after 20 seconds
    root.focus_force()
    root.mainloop()


def spawn_popup(kinds, image_path):
    cmd = [sys.executable, os.path.abspath(__file__),
           "--popup", ",".join(kinds), "--popup-image", image_path]
    return subprocess.Popen(cmd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


# ---------- detection ----------
def detect(frame, road, pile):
    dets, found = [], set()

    r = road.predict(frame, imgsz=640, conf=0.25, iou=0.4, agnostic_nms=True, verbose=False)[0]
    for box, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        name = road.names[int(c)]
        if name == "Container":
            dets.append((name, float(s), box))
            continue
        if s < MIN_CONF.get(name, 0.25):
            continue
        found.add(name)
        dets.append((name, float(s), box))

    p = pile.predict(frame, imgsz=640, conf=PILE_CONF, verbose=False)[0]
    for box, c, s in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.cls.cpu().numpy(), p.boxes.conf.cpu().numpy()):
        if int(c) != PILE_CLASS:
            continue
        found.add("waste_pile")
        dets.append(("waste_pile", float(s), box))

    return dets, found


# ---------- drawing ----------
def draw_box(frame, box, name, conf):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(name, (0, 255, 0))
    is_bin = name == "Container"

    if not is_bin:                                   # see-through colour fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_bin else 3)

    L = max(12, min(40, (x2 - x1) // 5, (y2 - y1) // 5))      # bright corner marks
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, 6)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, 6)

    text = f"{NICE.get(name, name)} {int(conf * 100)}%"        # name tag
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
    top = y1 - th - 12
    if top < 0:
        top = y1
    cv2.rectangle(frame, (x1, top), (x1 + tw + 14, top + th + 12), color, -1)
    cv2.putText(frame, text, (x1 + 7, top + th + 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def alert_overlay(frame, kinds, n):
    h, w = frame.shape[:2]
    col = (0, 0, 220) if (n // 8) % 2 == 0 else (0, 0, 120)     # flashing red

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 10)       # red border

    overlay = frame.copy()                                       # banner
    cv2.rectangle(overlay, (0, 0), (w, 74), col, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    tri = np.array([[56, 10], [86, 62], [26, 62]], np.int32)     # warning triangle
    cv2.fillPoly(frame, [tri], (0, 200, 255))
    cv2.putText(frame, "!", (50, 56), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, "WASTE DETECTED", (105, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "  +  ".join(NICE.get(k, k) for k in kinds), (107, 65),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="video file path, or 0 for the webcam")
    ap.add_argument("--every", type=int, default=2, help="run the models every N frames (higher = faster)")
    ap.add_argument("--width", type=int, default=960, help="shrink frames to this width")
    ap.add_argument("--popup", default="", help="(internal) show only the popup window")
    ap.add_argument("--popup-image", default="", help="(internal) picture for the popup")
    args = ap.parse_args()

    if args.popup:                                   # popup-only mode
        show_popup_window(args.popup, args.popup_image)
        return

    from ultralytics import YOLO
    road, pile = YOLO(ROAD_MODEL), YOLO(PILE_MODEL)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    scale = min(1.0, args.width / int(cap.get(3)))
    W, H = int(cap.get(3) * scale), int(cap.get(4) * scale)

    Path("alerts").mkdir(exist_ok=True)
    out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    streak = {"waste_spilled": 0, "waste_Overload": 0, "waste_pile": 0}
    dets, found, n = [], set(), 0
    last_popup, popup_proc = 0.0, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (W, H))
        if n % args.every == 0:
            dets, found = detect(frame, road, pile)

        for name, conf, box in dets:
            draw_box(frame, box, name, conf)

        for k in streak:
            streak[k] = streak[k] + 1 if k in found else 0
        active = [k for k, v in streak.items() if v >= ALERT_AFTER]

        if active:
            alert_overlay(frame, active, n)
            popup_busy = popup_proc is not None and popup_proc.poll() is None
            if time.time() - last_popup > POPUP_COOLDOWN and not popup_busy:
                last_popup = time.time()
                snap = os.path.abspath(f"alerts/alert_{int(last_popup)}.jpg")
                cv2.imwrite(snap, frame)
                popup_proc = spawn_popup(active, snap)

        out.write(frame)
        cv2.imshow("Waste detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        n += 1

    cap.release(); out.release(); cv2.destroyAllWindows()
    print("Done. Saved result.mp4 and alert snapshots in the alerts folder")


if __name__ == "__main__":
    main()