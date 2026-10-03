# -*- coding: utf-8 -*-
"""
task3 题二：用 PyTorch 自动求导 + 梯度下降完成最小二乘拟合。

目标函数：y = 3.5x + 4.2 + noise，noise 是 [-0.5, 0.5] 的随机数。
要求：学习率 0.01，迭代 1000 次，初始权重 0、初始偏置 0，
     绘制原函数与拟合函数图像，输出拟合参数和 R²。
"""
import os

import matplotlib
import torch

# 无显示环境也能保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 让 matplotlib 正常显示中文
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)  # 固定随机种子，结果可复现

# ---------- 1. 生成带噪声的数据 ----------
x = torch.linspace(-10, 10, 100)
noise = torch.rand(100) - 0.5          # 均匀分布 [-0.5, 0.5]
y_true = 3.5 * x + 4.2                 # 不含噪声的原函数
y_noisy = y_true + noise               # 实际观测数据

# ---------- 2. 初始化待拟合参数 ----------
# requires_grad=True 让 PyTorch 记录它们参与的运算，以便自动求导
w = torch.zeros(1, requires_grad=True)  # 初始权重 0
b = torch.zeros(1, requires_grad=True)  # 初始偏置 0

LR = 0.01      # 学习率
EPOCHS = 1000  # 迭代次数

# ---------- 3. 梯度下降训练 ----------
for epoch in range(EPOCHS):
    y_pred = w * x + b                     # 前向传播
    loss = torch.mean((y_pred - y_noisy) ** 2)  # MSE 损失

    loss.backward()                        # 反向传播：自动计算 dL/dw, dL/db

    # 手动更新参数（相当于最朴素的 SGD）
    with torch.no_grad():
        w -= LR * w.grad
        b -= LR * b.grad
        w.grad.zero_()                     # 梯度清零，避免累加
        b.grad.zero_()

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch+1:4d}  loss={loss.item():.6f}  "
              f"w={w.item():.4f}  b={b.item():.4f}")

# ---------- 4. 计算 R² ----------
with torch.no_grad():
    y_fit = w * x + b
    ss_res = torch.sum((y_noisy - y_fit) ** 2)       # 残差平方和
    ss_tot = torch.sum((y_noisy - y_noisy.mean()) ** 2)  # 总变差
    r2 = 1 - ss_res / ss_tot

    w_val, b_val = w.item(), b.item()
    print(f"\n拟合结果: w = {w_val:.4f}, b = {b_val:.4f}")
    print(f"真实参数: w = 3.5000, b = 4.2000")
    print(f"决定系数 R² = {r2.item():.6f}")

# ---------- 5. 绘图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x.numpy(), y_true.numpy(), s=18, c="#1f77b4", label="样本点")
axes[0].plot(x.numpy(), y_true.numpy(), c="#d62728", lw=2, label="y = 3.5x + 4.2")
axes[0].set_title("原函数（不含噪声）")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(x.numpy(), y_noisy.numpy(), s=18, c="#1f77b4", label="带噪声样本")
axes[1].plot(x.numpy(), y_fit.numpy(), c="#2ca02c", lw=2,
             label=f"拟合: y = {w_val:.2f}x + {b_val:.2f}")
axes[1].set_title("PyTorch 拟合函数")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "q2_fit.png")
plt.savefig(out_png, dpi=150)
plt.close()

# ---------- 6. 保存文字结果 ----------
result = (f"拟合参数: w = {w_val:.4f}, b = {b_val:.4f}\n"
          f"真实参数: w = 3.5000, b = 4.2000\n"
          f"决定系数 R² = {r2.item():.6f}\n"
          f"学习率 = {LR}, 迭代次数 = {EPOCHS}\n")
with open(os.path.join(OUT_DIR, "q2_result.txt"), "w", encoding="utf-8") as f:
    f.write(result)

print(f"图片已保存: {out_png}")
