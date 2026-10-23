import subprocess, sys, shutil
from pathlib import Path

videos = sorted(Path("videos").glob("*.mp4"))
Path("results").mkdir(exist_ok=True)

for v in videos:
    print("Running on", v.name)
    subprocess.run([sys.executable, "waste_detector.py", "--source", str(v)])
    if Path("result.mp4").exists():
        shutil.move("result.mp4", f"results/{v.stem}_result.mp4")

print("All done. Look in the results folder.")