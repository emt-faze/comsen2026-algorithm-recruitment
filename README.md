# COMSEN 2026 算法组招新题目解答

四道题的完整实现：环境配置、OpenCV 图像处理、PyTorch MLP 识别 MNIST，
以及 task4 自主选题的两个方向（时序模型 LSTM + YOLO 无人机目标识别与量化）。

## 环境配置（task1）

本机已安装 Miniconda，并创建了 `comsen` 环境（Python 3.11）：

```powershell
conda activate comsen
python task1/hello.py
```

`hello.py` 会打印 `Hello Python` 并检查各依赖库版本，
输出正常即说明环境配置成功。

**GPU 说明**：`torch` 与 `torchvision` 必须同为 CUDA 版本，否则 YOLO 训练会在
AMP 检查阶段报 `Could not run 'torchvision::nms' with arguments from the 'CUDA' backend`。

```powershell
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
```

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
  q2_least_squares_2d.py  二元非线性函数拟合（题目要求，7 参数）
  q2_least_squares.py   一元线性拟合练习版（非本题答案，保留作对照）
  q3_mlp_mnist.py       MLP 识别 MNIST
  报告.md               结果与分析文档
task4_timeseries/       方向三：时序模型（LSTM）
  lstm_air_passengers.py LSTM 预测航空旅客数据
  报告.md               结果与分析文档
task4_yolo/             方向一：YOLO 无人机目标识别与量化
  q1_yolo_inference.py  预训练权重推理（图片 + 视频 + FPS 统计）
  q2_yolo_train.py      基于无人机数据集微调训练
  infer_ft.py           微调后权重推理
  exp3.py               导出 ONNX FP32 / FP16
  bench_q.py            三种后端速度对比
  thresh_sweep.py       置信度阈值扫描实验
  报告.md               训练曲线、混淆矩阵与量化分析
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
python q2_least_squares_2d.py   # 二元非线性拟合（本题正式答案）
python q3_mlp_mnist.py

# task4 方向三（在 task4_timeseries 目录下运行）
python lstm_air_passengers.py   # 首次运行会自动下载数据集

# task4 方向一（在 task4_yolo 目录下运行）
python q1_yolo_inference.py     # 用 drone.pt 推理
python q2_yolo_train.py --epochs 30 --batch 16 --device 0   # GPU 训练
python infer_ft.py              # 微调后推理
python exp3.py                  # 导出 ONNX FP32/FP16
python bench_q.py               # 后端速度对比
```

> `task4_yolo/dataset/`（Roboflow 无人机数据集）与 `drone.pt` 权重
> 因体积原因未纳入仓库，需从题目提供的链接自行下载。

所有脚本固定了随机种子（seed=42），重复运行结果一致。

## 主要结果

| 任务 | 关键指标 |
|---|---|
| task2 轮廓检测 | 17 个轮廓，总面积 41622.50 px²，总周长 3446.95 px |
| task2 运动检测 | 帧差法 6.37 目标/帧，MOG2 6.85 目标/帧 |
| task2 小球追踪 | example3 229 帧、example4 581 帧，覆盖率 100% |
| task3 二元拟合 | R² = 0.948643，7 个参数误差均 < 0.07 |
| task3 MLP | 训练 99.37%，测试 98.05%（235,146 参数） |
| task4 LSTM | MAE 22.42 人，RMSE 28.51 人，MAE/均值 5.32% |
| task4 YOLO | drone 类 mAP50 = 0.931，ONNX FP16 达 132.65 FPS |

## 依赖安装

```powershell
pip install -r requirements.txt
```

GPU 训练与 ONNX 导出另需：

```powershell
pip install ultralytics onnx onnxruntime-gpu onnxslim
```
