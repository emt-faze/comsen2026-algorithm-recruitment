# -*- coding: utf-8 -*-
"""
OpenCV 中文路径工具函数。

OpenCV 在 Windows 上遇到含中文的路径时，imread / imwrite 会静默失败，
所以这里统一用 imdecode / imencode + numpy 文件读写来绕开这个问题。
"""
import os

import cv2
import numpy as np


def imread_cn(path, flags=cv2.IMREAD_COLOR):
    """读取图片（支持中文路径）。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_cn(path, img):
    """保存图片（支持中文路径）。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


class VideoWriterCN:
    """保存视频（支持中文路径）。

    先写入临时文件（纯英文路径），关闭时再移动到目标路径，
    避免部分 OpenCV 版本写不进中文路径的问题。
    """

    def __init__(self, path, fps, size):
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.tmp = os.path.join(os.environ.get("TEMP", "."), "cv_tmp_out.mp4")
        self.writer = cv2.VideoWriter(self.tmp, fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器: {self.tmp}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        if os.path.exists(self.tmp):
            os.replace(self.tmp, self.path)
