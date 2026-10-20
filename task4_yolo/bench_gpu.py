import os, time, json
import numpy as np, cv2
from ultralytics import YOLO

BEST = os.path.join("outputs","drone_finetune","weights","best.pt")
print("权重:", BEST, os.path.getsize(BEST)/1e6, "MB")

def bench(model_path, tag, frames):
    m = YOLO(model_path)
    for f in frames[:5]: m.predict(f, conf=0.25, verbose=False)   # 预热
    ts, dets = [], []
    for f in frames:
        t0=time.perf_counter()
        r=m.predict(f, conf=0.25, verbose=False)[0]
        ts.append((time.perf_counter()-t0)*1000)
        dets.append(0 if r.boxes is None else len(r.boxes))
    a=np.array(ts)
    return {"tag":tag,"size_mb":round(os.path.getsize(model_path)/1e6,2),
            "mean_ms":round(float(a.mean()),2),"p50":round(float(np.percentile(a,50)),2),
            "p95":round(float(np.percentile(a,95)),2),"fps":round(float(1000/a.mean()),2),
            "dets":round(float(np.mean(dets)),2)}

cap = cv2.VideoCapture(os.path.join("example","example.mp4"))
frames=[]; i=0
while True:
    ok,f=cap.read()
    if not ok: break
    if i%8==0: frames.append(f)
    i+=1
cap.release()
print("基准帧数", len(frames))

res=[]
res.append(bench(os.path.join("drone_finetune","weights","best.pt") if False else BEST, "微调后 best.pt (GPU)", frames))
res.append(bench("drone.pt", "原始 drone.pt (GPU)", frames))
for r in res:
    print(f"{r['tag']:<26} {r['size_mb']:>5.2f} MB  {r['mean_ms']:>6.2f} ms  {r['fps']:>7.2f} FPS  每帧 {r['dets']}")
json.dump(res, open(os.path.join("outputs","yolo_gpu_bench.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
