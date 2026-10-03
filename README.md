# COMSEN 2026 算法组招新题目解答

大一新生四道题的完整实现：环境配置、OpenCV 图像处理、
PyTorch MLP 识别 MNIST、LSTM 时序预测（task4 选择时序模型方向）。

## 环境配置（task1）

本机已安装 Miniconda，并创建了 `comsen` 环境（Python 3.11）：

```powershell
conda activate comsen
python task1/hello.py
```

`hello.py` 会打印 `Hello Python` 并检查各依赖库版本，
输出正常即说明环境配置成功。

## 项目结构

```text
task1/                  环境验证脚本
task2/                  OpenCV + Pillow 题目
  q1_q2_image_ops.py    打开显示、通道分离、缩放/旋转/平移/翻转
  q3_contours.py        轮廓检测、总面积与总周长
  q4_motion_detect.py   帧差法 vs MOG2 运动目标检测
  q5_ball_track.py      MIL 追踪器 + 颜色重定位的小球追踪
  cv_utils.py           OpenCV 中文路径读写工具
  报告.md               结果与分析文档
task3/                  PyTorch 题目
  q1_concepts.md        深度学习概念理解
  q2_least_squares.py   自动求导 + 梯度下降最小二乘拟合
  q3_mlp_mnist.py       MLP 识别 MNIST
  报告.md               结果与分析文档
task4_timeseries/       时序模型方向（LSTM）
  lstm_air_passengers.py LSTM 预测航空旅客数据
  报告.md               结果与分析文档
```

每个 task 的生成结果（图片、视频、指标文件）都在各自的
`outputs/` 文件夹里，视频和报告按提交要求打包发送。

## 运行方式

```powershell
conda activate comsen

# task2（在 task2 目录下运行）
python q1_q2_image_ops.py     # 加 --show 可弹出窗口实时查看
python q3_contours.py
python q4_motion_detect.py
python q5_ball_track.py

# task3（在 task3 目录下运行）
python q2_least_squares.py
python q3_mlp_mnist.py

# task4（在 task4_timeseries 目录下运行）
python lstm_air_passengers.py  # 首次运行会自动下载数据集
```

所有脚本固定了随机种子（seed=42），重复运行结果一致。

## 依赖安装

```powershell
pip install -r requirements.txt
```
