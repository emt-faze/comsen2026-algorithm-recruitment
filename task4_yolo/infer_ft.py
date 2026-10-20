import os, time, json
import numpy as np, cv2
from ultralytics import YOLO

BEST = os.path.join("outputs","drone_finetune","weights","best.pt")
m = YOLO(BEST)

# 图片推理
rows=[]
for name in ["coco-bear.jpg","coco-bus.jpg","coco-person.jpg"]:
    src=os.path.join("example",name)
    r = m.predict(src, conf=0.25, verbose=False)[0]
    dst=os.path.join("outputs", f"yolo_{os.path.splitext(name)[0]}_finetuned.jpg")
    cv2.imwrite(dst, r.plot())
    n=0 if r.boxes is None else len(r.boxes)
    dets=[]
    if r.boxes is not None:
        for b in r.boxes:
            dets.append({"cls": r.names[int(b.cls[0])], "conf": round(float(b.conf[0]),4)})
    rows.append({"image":name,"objects":n,"detections":dets})
    print(f"[图片] {name}: {n} 个目标 {dets}")

# 视频推理
src=os.path.join("example","example.mp4")
cap=cv2.VideoCapture(src)
fps=cap.get(cv2.CAP_PROP_FPS) or 24
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dst=os.path.join("outputs","yolo_example_finetuned.mp4")
vw=cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
ts=[]; cc={}; cnt=[]; fid=0
while True:
    ok,f=cap.read()
    if not ok: break
    fid+=1
    t0=time.perf_counter(); r=m.predict(f, conf=0.25, verbose=False)[0]; t1=time.perf_counter()
    if fid>5:
        ts.append((t1-t0)*1000)
        cnt.append(0 if r.boxes is None else len(r.boxes))
    if r.boxes is not None:
        for b in r.boxes:
            nm=r.names[int(b.cls[0])]; cc[nm]=cc.get(nm,0)+1
    vw.write(r.plot())
cap.release(); vw.release()
a=np.array(ts)
stats={"frames":fid,"mean_ms":round(float(a.mean()),2),"p50":round(float(np.percentile(a,50)),2),
       "p95":round(float(np.percentile(a,95)),2),"fps":round(float(1000/a.mean()),2),
       "avg_objects":round(float(np.mean(cnt)),2),"class_counts":cc,"output":os.path.basename(dst)}
print(f"[视频] {fid} 帧, {stats['mean_ms']} ms/帧, {stats['fps']} FPS, 每帧 {stats['avg_objects']} 目标, {cc}")
json.dump({"images":rows,"video":stats}, open(os.path.join("outputs","yolo_finetuned_result.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
