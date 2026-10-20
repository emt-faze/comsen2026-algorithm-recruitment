import os, time, json
import numpy as np
from ultralytics import YOLO

W = os.path.join(os.path.dirname(os.path.abspath(".")), "task4_yolo", "drone.pt")
W = "drone.pt"
model = YOLO(W)

# 用视频中间帧做阈值扫描
import cv2
cap = cv2.VideoCapture(os.path.join("example", "example.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
ok, frame = cap.read(); cap.release()
print("frame", frame.shape)

print(f"{'conf':>6} | {'总检出':>7} | {'Drone':>6} | {'Bird':>5}")
for conf in [0.10, 0.25, 0.40, 0.50, 0.60, 0.75]:
    r = model.predict(frame, conf=conf, verbose=False)[0]
    n = 0 if r.boxes is None else len(r.boxes)
    d = b = 0
    if r.boxes is not None:
        for bx in r.boxes:
            if r.names[int(bx.cls[0])] == "Drone": d += 1
            else: b += 1
    print(f"{conf:>6.2f} | {n:>7} | {d:>6} | {b:>5}")
