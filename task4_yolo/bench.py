import os, time, json
import numpy as np, cv2

# 取 40 帧做统一基准
cap = cv2.VideoCapture(os.path.join("example","example.mp4"))
frames=[]
i=0
while True:
    ok,f = cap.read()
    if not ok: break
    if i % 8 == 0: frames.append(f)
    i+=1
cap.release()
print("基准帧数", len(frames))

def bench(model_path, tag):
    from ultralytics import YOLO
    m = YOLO(model_path)
    # 预热
    for f in frames[:3]:
        m.predict(f, conf=0.25, verbose=False)
    ts=[]
    det=[]
    for f in frames:
        t0=time.perf_counter()
        r=m.predict(f, conf=0.25, verbose=False)[0]
        ts.append((time.perf_counter()-t0)*1000)
        det.append(0 if r.boxes is None else len(r.boxes))
    a=np.array(ts)
    return {"tag":tag,"size_mb":round(os.path.getsize(model_path)/1e6,2),
            "mean_ms":round(float(a.mean()),2),"p50_ms":round(float(np.percentile(a,50)),2),
            "fps":round(float(1000/a.mean()),2),"mean_dets":round(float(np.mean(det)),2)}

res=[]
res.append(bench("drone.pt","PyTorch (.pt)"))
res.append(bench("drone.onnx","ONNX Runtime"))
for r in res:
    print(f"{r['tag']:<16} {r['size_mb']:>6.2f} MB  {r['mean_ms']:>7.2f} ms  {r['fps']:>7.2f} FPS  每帧目标 {r['mean_dets']}")
json.dump(res, open(os.path.join("outputs","yolo_backend_bench.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
