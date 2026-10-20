import os
from ultralytics import YOLO
BEST = os.path.join("outputs","drone_finetune","weights","best.pt")
m = YOLO(BEST)

# 1) FP16 ONNX
try:
    p = m.export(format="onnx", imgsz=640, half=True, opset=12)
    print("FP16 ONNX:", p, round(os.path.getsize(p)/1e6,2), "MB")
except Exception as e:
    print("FP16 ONNX 失败:", type(e).__name__, e)

# 2) FP32 ONNX（对比基线）
try:
    m2 = YOLO(BEST)
    p2 = m2.export(format="onnx", imgsz=640, half=False, opset=12)
    print("FP32 ONNX:", p2, round(os.path.getsize(p2)/1e6,2), "MB")
except Exception as e:
    print("FP32 ONNX 失败:", type(e).__name__, e)
