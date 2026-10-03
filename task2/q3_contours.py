# -*- coding: utf-8 -*-
"""
task2 题3：检测 example1.jpg 的轮廓，保存结果并计算总面积、总周长。

思路：
    原图是白底深色图案。先转灰度图，再做反二值化
    （图案变成白色、背景变成黑色），最后 findContours 提取轮廓。
"""
import os

import cv2

from cv_utils import imread_cn, imwrite_cn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "example1.jpg")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    img = imread_cn(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"读取失败: {IMG_PATH}")

    # 1. 转灰度图并轻度去噪
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. OTSU 自动阈值 + 反二值化：图案区域变白
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 提取最外层轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 计算总面积与总周长
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)

    # 5. 把轮廓画回原图（绿色描边）并保存
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 200, 0), 2)
    imwrite_cn(os.path.join(OUT_DIR, "q3_contours.jpg"), result)

    # 6. 二值化中间结果也保存一份，方便写报告时对照
    imwrite_cn(os.path.join(OUT_DIR, "q3_binary.jpg"), binary)

    lines = [
        f"轮廓数量: {len(contours)}",
        f"总面积: {total_area:.2f} 像素",
        f"总周长: {total_perimeter:.2f} 像素",
    ]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, "q3_result.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
