# -*- coding: utf-8 -*-
"""
task4 方向一（第 3 步）：基于无人机数据集与 drone.pt 训练 YOLO 模型

数据集：drone and bird v2（Roboflow），2 类 bird / drone
    train 3893 / valid 649 / test 196

训练策略：
    以题目提供的 drone.pt 为初始权重做微调（fine-tune），
    因为该权重已在同类数据上收敛，微调比从 COCO 预训练权重从头训练更快收敛。

用法：
    python q2_yolo_train.py                # 默认 30 epoch
    python q2_yolo_train.py --epochs 50
    python q2_yolo_train.py --device cpu   # 无 GPU 时
"""
import argparse
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
WEIGHTS = os.path.join(BASE_DIR, "drone.pt")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0 = 第一块 GPU，cpu = 纯 CPU")
    ap.add_argument("--model", default=WEIGHTS, help="初始权重")
    ap.add_argument("--name", default="drone_finetune")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    print(f"torch {torch.__version__} | CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"找不到数据集配置: {DATASET_YAML}")

    model = YOLO(args.model)

    results = model.train(
        data=DATASET_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=OUT_DIR,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        # 数据增强
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
        # 优化
        optimizer="auto",
        patience=15,
        workers=4,
        verbose=True,
    )

    # 把关键绘图复制到 outputs 根目录，方便报告引用
    run_dir = os.path.join(OUT_DIR, args.name)
    for f in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
              "PR_curve.png", "F1_curve.png", "labels.jpg", "train_batch0.jpg"]:
        src = os.path.join(run_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT_DIR, f"train_{f}"))
            print(f"复制 {f}")

    print("\n训练完成，结果目录:", run_dir)
    print("最佳权重:", os.path.join(run_dir, "weights", "best.pt"))


if __name__ == "__main__":
    main()
