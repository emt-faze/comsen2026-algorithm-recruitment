import os, time, json
import numpy as np, cv2
from ultralytics import YOLO

model = YOLO("drone.pt")

# 采样视频中的 60 帧做阈值扫描（每 6 帧取一帧）
cap = cv2.VideoCapture(os.path.join("example", "example.mp4"))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % 6 == 0: frames.append(f)
    i += 1
cap.release()
print("采样帧数", len(frames))

rows = []
for conf in [0.10, 0.25, 0.40, 0.50]:
    tot = drone = bird = 0
    for f in frames:
        r = model.predict(f, conf=conf, verbose=False)[0]
        if r.boxes is None: continue
        for bx in r.boxes:
            tot += 1
            if r.names[int(bx.cls[0])] == "Drone": drone += 1
            else: bird += 1
    n = len(frames)
    row = {"conf": conf, "total": tot, "drone": drone, "bird": bird,
           "per_frame": round(tot/n, 2), "drone_per_frame": round(drone/n, 2)}
    rows.append(row)
    print(f"conf={conf:.2f}  总{tot:4d} (每帧{tot/n:.2f})  Drone {drone:3d} ({drone/n:.2f})  Bird {bird:3d} ({bird/n:.2f})")

json.dump(rows, open(os.path.join("outputs","yolo_threshold_sweep.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved")
