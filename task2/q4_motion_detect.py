# -*- coding: utf-8 -*-
"""
task2 题4：对 example2.mp4 做运动目标检测，对比两种方法。

方法A 帧差法（未使用背景减除）：
    只比较相邻两帧的差别。背景里静止的物体不会出现，
    但任何画面整体变化（光照、镜头抖动）都会被当成“运动”。

方法B MOG2 背景减除：
    先学习一段背景模型，再把每一帧和背景模型比较。
    模型会持续更新，抗光照变化和镜头抖动的能力更强。
"""
import os

import cv2

from cv_utils import VideoWriterCN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "example2.mp4")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 形态学核：把检测区域连成块
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def find_boxes(mask, min_area=500):
    """在二值前景图里找出运动目标的包围框。"""
    # 开运算去噪点，膨胀让断裂的区域连起来
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    clean = cv2.dilate(clean, KERNEL, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def draw_boxes(frame, boxes, color, label):
    """在画面上标出运动目标。"""
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"objects: {len(boxes)}", (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def process(method):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 两遍共用同一套输出配置
    if method == "frame_diff":
        out_path = os.path.join(OUT_DIR, "example2_frame_diff.mp4")
        color, label = (0, 255, 255), "Frame Difference"
    else:
        out_path = os.path.join(OUT_DIR, "example2_mog2.mp4")
        color, label = (0, 255, 0), "MOG2 Background Subtraction"
    writer = VideoWriterCN(out_path, fps, (width, height))

    # MOG2 参数：
    #   history  参与建模的帧数（越长背景越“稳定”）
    #   varThreshold 像素和背景的差异阈值（越大越不容易误报）
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=40, detectShadows=False)

    prev_gray = None
    frame_id = 0
    total_objects = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if method == "frame_diff":
            if prev_gray is None:
                prev_gray = gray
                continue
            diff = cv2.absdiff(prev_gray, gray)
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            prev_gray = gray
        else:
            mask = bg_subtractor.apply(gray)

        boxes = find_boxes(mask)
        total_objects += len(boxes)
        draw_boxes(frame, boxes, color, label)
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    avg = total_objects / max(frame_id, 1)
    print(f"[{method}] 处理 {frame_id} 帧, 平均每帧检出 {avg:.1f} 个目标 -> {out_path}")
    return {"frames": frame_id, "total_objects": total_objects, "avg_objects": avg}


if __name__ == "__main__":
    stats_a = process("frame_diff")
    stats_b = process("mog2")

    report = (
        "example2.mp4 运动目标检测对比\n"
        "============================\n"
        f"帧差法   : 处理 {stats_a['frames']} 帧, "
        f"平均每帧检出 {stats_a['avg_objects']:.2f} 个目标\n"
        f"MOG2法   : 处理 {stats_b['frames']} 帧, "
        f"平均每帧检出 {stats_b['avg_objects']:.2f} 个目标\n"
    )
    print(report)
    with open(os.path.join(OUT_DIR, "q4_result.txt"), "w", encoding="utf-8") as f:
        f.write(report)
