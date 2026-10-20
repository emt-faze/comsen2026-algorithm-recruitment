import os, shutil
from ultralytics import YOLO
BEST = os.path.join("outputs","drone_finetune","weights","best.pt")
W = os.path.join("outputs","drone_finetune","weights")

# FP32
m = YOLO(BEST); p = m.export(format="onnx", imgsz=640, half=False, opset=12)
shutil.move(p, os.path.join(W, "drone_fp32.onnx"))
print("FP32:", round(os.path.getsize(os.path.join(W,"drone_fp32.onnx"))/1e6,2), "MB")

# FP16
m2 = YOLO(BEST); p2 = m2.export(format="onnx", imgsz=640, half=True, opset=12)
shutil.move(p2, os.path.join(W, "drone_fp16.onnx"))
print("FP16:", round(os.path.getsize(os.path.join(W,"drone_fp16.onnx"))/1e6,2), "MB")

# 清理默认名
if os.path.exists(os.path.join(W,"best.onnx")): os.remove(os.path.join(W,"best.onnx"))
