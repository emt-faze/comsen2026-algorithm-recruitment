import os, time, json
import numpy as np, cv2
from ultralytics import YOLO

model = YOLO("drone.pt")

# 导出 ONNX（动态 batch，固定 640）
out = model.export(format="onnx", imgsz=640, opset=12, simplify=False)
print("exported:", out, os.path.getsize(out)/1e6, "MB")
