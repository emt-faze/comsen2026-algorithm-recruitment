# -*- coding: utf-8 -*-
"""
task2 题5：用 OpenCV 追踪器追踪 example3.mp4 / example4.mp4 里的小球。

思路：
    1. 在第一帧里自动找球（白色、接近圆形的小目标）。
    2. 用 OpenCV 内置的 MIL 追踪器持续追踪。
    3. 追丢时（update 返回失败）回到“颜色 + 圆度”重新找球，
       再重新初始化追踪器，相当于给追踪器加一层兜底。
"""
import os

import cv2
import numpy as np

from cv_utils import VideoWriterCN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 连续多少帧判定为“跟丢”
LOST_LIMIT = 15


def detect_ball(frame):
    """在整帧里找最像“白色小球”的目标，返回 (x, y, r) 或 None。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 白色特征：饱和度低、亮度高
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 4 or area > 3000:  # 太小是噪声，太大可能是白衣服/白线
            continue
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius <= 0:
            continue
        # 圆度：越接近 1 说明越圆（长条形的白线圆度很低）
        circularity = area / (np.pi * radius * radius)
        if circularity < 0.45:
            continue
        # 综合评分：优先选圆、且面积别太小的目标
        score = circularity * np.sqrt(area)
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(radius))

    return best


def track_video(video_name, out_name):
    video_path = os.path.join(BASE_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = VideoWriterCN(os.path.join(OUT_DIR, out_name), fps, (width, height))

    ok, first = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("视频读取失败")

    ball = detect_ball(first)
    if ball is None:
        cap.release()
        writer.release()
        raise RuntimeError("第一帧里没找到小球，请调整 detect_ball 的阈值")

    def make_bbox(x, y, r):
        pad = max(6.0, r * 0.8)  # 给追踪器留一点上下文
        # OpenCV 5 的追踪器要求包围框是整数元组 (x, y, w, h)
        return (int(x - pad), int(y - pad), int(2 * pad), int(2 * pad))

    bbox = make_bbox(*ball)
    tracker = cv2.TrackerMIL_create()
    tracker.init(first, bbox)

    lost_count = 0
    tracked_frames = 1
    reinit_count = 0
    frame_id = 1

    annotated = first.copy()
    x, y, r = ball
    cv2.circle(annotated, (int(x), int(y)), int(r) + 6, (0, 255, 0), 2)
    writer.write(annotated)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        success, bbox = tracker.update(frame)
        if success:
            lost_count = 0
            tracked_frames += 1
        else:
            lost_count += 1
            if lost_count >= LOST_LIMIT:
                # 追丢了：重新用颜色找球并重置追踪器
                ball = detect_ball(frame)
                if ball is not None:
                    bbox = make_bbox(*ball)
                    tracker = cv2.TrackerMIL_create()
                    tracker.init(frame, bbox)
                    reinit_count += 1
                    lost_count = 0

        # 标注：成功时画绿框，丢失期间画红框提示
        if success or lost_count == 0:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "tracking", (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"lost {lost_count} frames", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    ratio = tracked_frames / frame_id
    stats = {
        "video": video_name,
        "frames": frame_id,
        "tracked": tracked_frames,
        "ratio": ratio,
        "reinit": reinit_count,
    }
    print(f"[{video_name}] 共 {frame_id} 帧, 追踪覆盖率 {ratio:.1%}, "
          f"重定位 {reinit_count} 次 -> {out_name}")
    return stats


if __name__ == "__main__":
    stats = []
    stats.append(track_video("example3.mp4", "example3_tracked.mp4"))
    stats.append(track_video("example4.mp4", "example4_tracked.mp4"))

    lines = [f"{s['video']}: {s['frames']} 帧, 追踪覆盖率 {s['ratio']:.1%}, "
             f"重定位 {s['reinit']} 次" for s in stats]
    report = "小球追踪统计\n" + "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, "q5_result.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
