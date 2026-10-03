# -*- coding: utf-8 -*-
"""
task1：环境配置验证。

运行方式（在项目根目录）：
    conda activate comsen
    python task1/hello.py

如果能打印出 Hello Python 和各库版本号，说明环境配置成功。
"""
import sys


def main():
    print("Hello Python")
    print(f"Python 解释器: {sys.executable}")
    print(f"Python 版本: {sys.version.split()[0]}")

    # 常用库版本检查
    libs = {}
    for name in ["numpy", "pandas", "matplotlib", "cv2", "PIL", "torch"]:
        try:
            module = __import__(name)
            libs[name] = getattr(module, "__version__", "未知")
        except ImportError:
            libs[name] = "未安装"

    for name, version in libs.items():
        print(f"{name:12s} {version}")


if __name__ == "__main__":
    main()
