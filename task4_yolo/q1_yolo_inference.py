# -*- coding: utf-8 -*-
"""
task4 方向1（目标追踪与识别）：基于 YOLO 的目标识别与模型量化

本题分四步，本脚本完成第 2、3、4 步的推理与指标分析部分：
    1. 用题目提供的无人机预训练权重 drone.pt 对 example 下三张图片推理并保存
    2. 对 example.mp4 无人机飞行视频逐帧推理，保存标注视频
    3. 统计模型大小(MB) 与推理速度(FPS)，并与官方 COCO 预训练权重对比
    4. （提高）导出 ONNX 并对比推理速度（见 export_quantize.py）

用法：
    python q1_yolo_inference.py            # 图片 + 视频 + 指标
    python q1_yolo_inference.py --images   # 只做图片
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(BASE_DIR, "example")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 题目提供的无人机预训练权重（放在题目根目录上一层）
DRONE_WEIGHTS = os.path.join(os.path.dirname(BASE_DIR), "drone.pt")
if not os.path.exists(DRONE_WEIGHTS):
    DRONE_WEIGHTS = os.path.join(BASE_DIR, "drone.pt")

IMAGES = ["coco-bear.jpg", "coco-bus.jpg", "coco-person.jpg"]
VIDEO = "example.mp4"


def model_size_mb(path):
    """权重文件大小（MB，十进制）。"""
    return os.path.getsize(path) / 1e6


def count_params(model):
    """可学习参数量。"""
    return sum(p.numel() for p in model.parameters())


def run_images(model, conf=0.25):
    """对三张示例图片推理并保存标注结果。"""
    rows = []
    for name in IMAGES:
        src = os.path.join(EXAMPLE_DIR, name)
        if not os.path.exists(src):
            print(f"[跳过] 找不到 {src}")
            continue

        t0 = time.perf_counter()
        results = model.predict(src, conf=conf, verbose=False)
        t1 = time.perf_counter()

        r = results[0]
        dst = os.path.join(OUT_DIR, f"yolo_{os.path.splitext(name)[0]}_drone.jpg")
        cv2.imwrite(dst, r.plot())          # plot() 返回 BGR 标注图

        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                dets.append({
                    "cls": r.names[int(b.cls[0])],
                    "conf": round(float(b.conf[0]), 4),
                    "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                })

        rows.append({
            "image": name,
            "objects": len(dets),
            "infer_ms": round((t1 - t0) * 1000, 2),
            "detections": dets,
        })
        print(f"[图片] {name}: {len(dets)} 个目标, {(t1 - t0) * 1000:.1f} ms -> {os.path.basename(dst)}")
    return rows


def run_video(model, conf=0.25, warmup=5):
    """对视频逐帧推理，保存标注视频并统计 FPS。"""
    src = os.path.join(EXAMPLE_DIR, VIDEO)
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    cap = cv2.VideoCapture(src)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dst = os.path.join(OUT_DIR, "yolo_example_drone.mp4")
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    infer_times = []      # 纯推理耗时（毫秒）
    det_counts = []
    class_counts = {}
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        t0 = time.perf_counter()
        results = model.predict(frame, conf=conf, verbose=False)
        t1 = time.perf_counter()

        r = results[0]
        n = 0 if r.boxes is None else len(r.boxes)
        if frame_id > warmup:                    # 跳过预热帧，统计更稳定
            infer_times.append((t1 - t0) * 1000)
            det_counts.append(n)
        if r.boxes is not None:
            for b in r.boxes:
                nm = r.names[int(b.cls[0])]
                class_counts[nm] = class_counts.get(nm, 0) + 1

        writer.write(r.plot())
        if frame_id % 50 == 0:
            print(f"  已处理 {frame_id}/{n_total} 帧", flush=True)

    cap.release()
    writer.release()

    arr = np.array(infer_times) if infer_times else np.array([0.0])
    stats = {
        "video": VIDEO,
        "frames": frame_id,
        "resolution": f"{w}x{h}",
        "source_fps": round(float(fps_src), 2),
        "infer_ms_mean": round(float(arr.mean()), 2),
        "infer_ms_p50": round(float(np.percentile(arr, 50)), 2),
        "infer_ms_p95": round(float(np.percentile(arr, 95)), 2),
        "fps_model": round(float(1000.0 / arr.mean()), 2),
        "avg_objects_per_frame": round(float(np.mean(det_counts)) if det_counts else 0.0, 2),
        "class_counts": class_counts,
        "output": os.path.basename(dst),
    }
    print(f"[视频] {frame_id} 帧, 纯推理 {stats['infer_ms_mean']} ms/帧, "
          f"{stats['fps_model']} FPS -> {os.path.basename(dst)}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="只做图片推理")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO

    print(f"权重: {DRONE_WEIGHTS}")
    if not os.path.exists(DRONE_WEIGHTS):
        raise FileNotFoundError(f"找不到 drone.pt: {DRONE_WEIGHTS}")

    model = YOLO(DRONE_WEIGHTS)
    print(f"模型: {os.path.basename(model.ckpt_path)} | 类别: {model.names}")

    info = {
        "weights": os.path.basename(DRONE_WEIGHTS),
        "model_size_mb": round(model_size_mb(DRONE_WEIGHTS), 2),
        "params": count_params(model.model),
        "classes": {int(k): v for k, v in model.names.items()},
        "num_classes": len(model.names),
    }
    print(f"模型大小 {info['model_size_mb']} MB, 参数量 {info['params']:,}")

    images = run_images(model, conf=args.conf)
    video = None
    if not args.images:
        video = run_video(model, conf=args.conf)

    result = {"model_info": info, "images": images, "video": video}

    # 保存 JSON（供报告引用）
    with open(os.path.join(OUT_DIR, "yolo_inference_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存人类可读摘要
    lines = [
        "YOLO 无人机识别推理结果",
        "=" * 40,
        f"权重: {info['weights']}",
        f"模型大小: {info['model_size_mb']} MB",
        f"参数量: {info['params']:,}",
        f"类别: {info['classes']}",
        "",
        "图片推理:",
    ]
    for row in images:
        lines.append(f"  {row['image']}: {row['objects']} 个目标, {row['infer_ms']} ms")
        for d in row["detections"]:
            lines.append(f"      - {d['cls']} conf={d['conf']}")
    if video:
        lines += [
            "",
            "视频推理:",
            f"  视频: {video['video']} ({video['resolution']}, {video['frames']} 帧)",
            f"  纯推理耗时: 平均 {video['infer_ms_mean']} ms/帧 "
            f"(P50 {video['infer_ms_p50']}, P95 {video['infer_ms_p95']})",
            f"  推理速度: {video['fps_model']} FPS",
            f"  平均每帧目标数: {video['avg_objects_per_frame']}",
            f"  各类别累计检出: {video['class_counts']}",
            f"  输出视频: {video['output']}",
        ]
    with open(os.path.join(OUT_DIR, "yolo_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
