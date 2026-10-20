import os, time, json
import numpy as np, cv2
from ultralytics import YOLO

W = os.path.join("outputs","drone_finetune","weights")
cap = cv2.VideoCapture(os.path.join("example","example.mp4"))
frames=[]; i=0
while True:
    ok,f=cap.read()
    if not ok: break
    if i%8==0: frames.append(f)
    i+=1
cap.release()

def bench(path, tag, device="0"):
    try:
        m = YOLO(path)
        for f in frames[:5]: m.predict(f, conf=0.25, verbose=False, device=device)
        ts=[]; dets=[]
        for f in frames:
            t0=time.perf_counter()
            r=m.predict(f, conf=0.25, verbose=False, device=device)[0]
            ts.append((time.perf_counter()-t0)*1000)
            dets.append(0 if r.boxes is None else len(r.boxes))
        a=np.array(ts)
        return {"tag":tag,"size_mb":round(os.path.getsize(path)/1e6,2),
                "mean_ms":round(float(a.mean()),2),"p50":round(float(np.percentile(a,50)),2),
                "fps":round(float(1000/a.mean()),2),"dets":round(float(np.mean(dets)),2),"ok":True}
    except Exception as e:
        return {"tag":tag,"size_mb":round(os.path.getsize(path)/1e6,2),"ok":False,"err":f"{type(e).__name__}: {e}"[:160]}

res=[]
res.append(bench(os.path.join(W,"best.pt"), "PyTorch FP32 (.pt) GPU"))
res.append(bench(os.path.join(W,"drone_fp32.onnx"), "ONNX FP32 (GPU)"))
res.append(bench(os.path.join(W,"drone_fp16.onnx"), "ONNX FP16 (GPU)"))
for r in res:
    if r["ok"]:
        print(f"{r['tag']:<26} {r['size_mb']:>6.2f} MB  {r['mean_ms']:>6.2f} ms  {r['fps']:>7.2f} FPS  每帧 {r['dets']}")
    else:
        print(f"{r['tag']:<26} 失败: {r['err']}")
json.dump(res, open(os.path.join("outputs","yolo_quant_bench.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
