# -*- coding: utf-8 -*-
"""
task2 题1、题2：OpenCV + Pillow 打开图片、通道分离、几何变换。

用法：
    python q1_q2_image_ops.py          # 只保存结果到 outputs/
    python q1_q2_image_ops.py --show   # 保存的同时弹出窗口显示
"""
import os

import cv2
import numpy as np
from PIL import Image

from cv_utils import imread_cn, imwrite_cn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "example1.jpg")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def q1_open_and_split_channels():
    """题1：两种库打开同一张图，并分离三个颜色通道。"""
    # OpenCV 读图：BGR 顺序
    img_bgr = imread_cn(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"读取失败: {IMG_PATH}")

    # Pillow 读图：RGB 顺序
    img_rgb = Image.open(IMG_PATH).convert("RGB")

    b, g, r = cv2.split(img_bgr)  # OpenCV 分离 BGR
    r_pil, g_pil, b_pil = img_rgb.split()  # Pillow 分离 RGB

    # 保存时把单通道转回三通道，方便直接查看颜色
    imwrite_cn(os.path.join(OUT_DIR, "q1_opencv_R.jpg"), cv2.cvtColor(r, cv2.COLOR_GRAY2BGR))
    imwrite_cn(os.path.join(OUT_DIR, "q1_opencv_G.jpg"), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    imwrite_cn(os.path.join(OUT_DIR, "q1_opencv_B.jpg"), cv2.cvtColor(b, cv2.COLOR_GRAY2BGR))
    r_pil.convert("RGB").save(os.path.join(OUT_DIR, "q1_pillow_R.jpg"))
    g_pil.convert("RGB").save(os.path.join(OUT_DIR, "q1_pillow_G.jpg"))
    b_pil.convert("RGB").save(os.path.join(OUT_DIR, "q1_pillow_B.jpg"))

    print(f"[题1] 原图尺寸: {img_bgr.shape[1]} x {img_bgr.shape[0]}")
    print("[题1] 通道图片已保存: q1_opencv_RGB.jpg / q1_pillow_RGB.jpg")
    return img_bgr


def q2_geometric_transforms(img):
    """题2：缩放、旋转、平移、翻转。"""
    h, w = img.shape[:2]

    # 1. 缩放：缩小到 50%
    scaled = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    # 2. 旋转：绕图片中心逆时针旋转 45 度，缩放因子 1.0
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))

    # 3. 平移：向右 100 像素、向下 50 像素
    matrix = np.float32([[1, 0, 100], [0, 1, 50]])
    translated = cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))

    # 4. 翻转：1 = 水平镜像，0 = 垂直镜像
    flipped_h = cv2.flip(img, 1)
    flipped_v = cv2.flip(img, 0)

    imwrite_cn(os.path.join(OUT_DIR, "q2_scaled.jpg"), scaled)
    imwrite_cn(os.path.join(OUT_DIR, "q2_rotated.jpg"), rotated)
    imwrite_cn(os.path.join(OUT_DIR, "q2_translated.jpg"), translated)
    imwrite_cn(os.path.join(OUT_DIR, "q2_flipped_h.jpg"), flipped_h)
    imwrite_cn(os.path.join(OUT_DIR, "q2_flipped_v.jpg"), flipped_v)

    print(f"[题2] 缩放后尺寸: {scaled.shape[1]} x {scaled.shape[0]}")
    print("[题2] 变换结果已保存: q2_scaled / rotated / translated / flipped_h / flipped_v")


def show_results(img, wait_ms=0):
    """可选：弹出窗口查看结果（按任意键切换下一张）。"""
    windows = [
        ("original", img),
        ("scaled", cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))),
        ("rotated 45", cv2.warpAffine(
            img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 45, 1.0),
            (img.shape[1], img.shape[0]))),
        ("flipped", cv2.flip(img, 1)),
    ]
    for name, frame in windows:
        cv2.imshow(name, frame)
        cv2.waitKey(wait_ms)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show = "--show" in os.sys.argv
    image = q1_open_and_split_channels()
    q2_geometric_transforms(image)
    if show:
        show_results(image)
    print("题1、题2 完成。")
