# -*- coding: utf-8 -*-
"""
task3 题二：用 PyTorch 自动求导 + 梯度下降，拟合二元非线性函数。

目标函数（题目给定，x1、x2 取 [-3, 3]，噪声为 [-3, 3] 的均匀随机数）：
    y = 2*x1^2 + 1.5*x1*x2 + 3*sin(x1) + 0.5*cos(2*x2)
        + exp(-0.5*x1) + 0.8*x2 + 2026 + noise

待拟合的参数形式（7 个未知量）：
    y = a1*x1^2 + a2*x1*x2 + a3*sin(x1) + a4*cos(2*x2)
        + exp(a5*x1) + a6*x2 + b

实现要点：
    1. 用 torch 张量表示 a1..a6 与 b，requires_grad=True 交给自动求导
    2. 手写梯度下降更新，不调用 optimizer（题目要求「自动求导 + 梯度下降」）
    3. b 的量级（约 2026）远大于其它项，若从 0 初始化会因梯度爆炸发散成 NaN，
       因此 b 用样本均值初始化 —— 这一点在报告里有专门说明
    4. 输出参数值、R²，并绘制真实曲面 / 拟合曲面 3D 对比图
"""
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- 配置 ----------
N_SAMPLES = 2000     # 采样点数
RANGE_LO, RANGE_HI = -3.0, 3.0   # x1、x2 的取值范围
NOISE_LO, NOISE_HI = -3.0, 3.0   # 噪声区间
LR = 0.01            # 学习率
EPOCHS = 3000        # 迭代次数

# 真实参数（题目给定）
TRUE_A = torch.tensor([2.0, 1.5, 3.0, 0.5, -0.5, 0.8])
TRUE_B = 2026.0


# ---------- 1. 生成带噪声的数据 ----------
def target(x1, x2, a1, a2, a3, a4, a5, a6, b):
    """按题目的函数形式计算 y（不含噪声）。"""
    return (a1 * x1 ** 2 + a2 * x1 * x2 + a3 * torch.sin(x1)
            + a4 * torch.cos(2 * x2) + torch.exp(a5 * x1) + a6 * x2 + b)


gen = torch.Generator().manual_seed(SEED)
x1 = torch.rand(N_SAMPLES, generator=gen) * (RANGE_HI - RANGE_LO) + RANGE_LO
x2 = torch.rand(N_SAMPLES, generator=gen) * (RANGE_HI - RANGE_LO) + RANGE_LO
noise = torch.rand(N_SAMPLES, generator=gen) * (NOISE_HI - NOISE_LO) + NOISE_LO

y_true = target(x1, x2, *TRUE_A, TRUE_B)
y_noisy = y_true + noise

print(f"样本数: {N_SAMPLES}, x1/x2 范围 [{RANGE_LO}, {RANGE_HI}], "
      f"噪声 [{NOISE_LO}, {NOISE_HI}]")
print(f"y 均值 {y_noisy.mean():.3f}, 标准差 {y_noisy.std():.3f}")

# ---------- 2. 初始化待拟合参数 ----------
# a1..a6 从 0 开始；b 用样本均值初始化，避免 2026 这个大量级导致梯度爆炸
a = torch.zeros(6, requires_grad=True)
b = torch.tensor([y_noisy.mean().item()], requires_grad=True)

# ---------- 3. 梯度下降训练 ----------
loss_history = []
for epoch in range(EPOCHS):
    y_pred = target(x1, x2, a[0], a[1], a[2], a[3], a[4], a[5], b)
    loss = torch.mean((y_pred - y_noisy) ** 2)   # MSE
    loss.backward()                              # 自动求导

    with torch.no_grad():
        a -= LR * a.grad                         # 手写梯度下降
        b -= LR * b.grad
        a.grad.zero_()                           # 梯度累加，必须清零
        b.grad.zero_()

    loss_history.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"epoch {epoch + 1:5d}/{EPOCHS}  loss={loss.item():.6f}")

# ---------- 4. 计算 R² ----------
with torch.no_grad():
    y_fit = target(x1, x2, a[0], a[1], a[2], a[3], a[4], a[5], b)
    ss_res = torch.sum((y_noisy - y_fit) ** 2)            # 残差平方和
    ss_tot = torch.sum((y_noisy - y_noisy.mean()) ** 2)   # 总变差
    r2 = float(1 - ss_res / ss_tot)

a_val = [float(v) for v in a.detach()]
b_val = float(b.detach())

names = ["a1", "a2", "a3", "a4", "a5", "a6"]
print("\n拟合结果 vs 真实参数：")
for n, fv, tv in zip(names, a_val, TRUE_A.tolist()):
    print(f"  {n}: 拟合 {fv:9.4f}   真实 {tv:7.4f}   误差 {abs(fv - tv):.4f}")
print(f"  b : 拟合 {b_val:9.4f}   真实 {TRUE_B:7.4f}   误差 {abs(b_val - TRUE_B):.4f}")
print(f"\n决定系数 R² = {r2:.6f}")

# ---------- 5. 绘图：真实曲面 vs 拟合曲面 ----------
grid = torch.linspace(RANGE_LO, RANGE_HI, 60)
G1, G2 = torch.meshgrid(grid, grid, indexing="ij")

with torch.no_grad():
    Z_true = target(G1, G2, *TRUE_A, TRUE_B)
    Z_fit = target(G1, G2, a[0], a[1], a[2], a[3], a[4], a[5], b)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(G1.numpy(), G2.numpy(), Z_true.numpy(),
                 cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
ax1.scatter(x1.numpy(), x2.numpy(), y_noisy.numpy(),
            c="red", s=4, alpha=0.35, label="带噪声样本")
ax1.set_title(f"原函数（不含噪声）\n真实参数")
ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_zlabel("y")
ax1.legend(loc="upper left", fontsize=8)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(G1.numpy(), G2.numpy(), Z_fit.numpy(),
                 cmap="magma", alpha=0.85, linewidth=0, antialiased=True)
ax2.scatter(x1.numpy(), x2.numpy(), y_noisy.numpy(),
            c="blue", s=4, alpha=0.35, label="带噪声样本")
ax2.set_title(f"PyTorch 拟合函数\nR² = {r2:.4f}")
ax2.set_xlabel("x1"); ax2.set_ylabel("x2"); ax2.set_zlabel("y")
ax2.legend(loc="upper left", fontsize=8)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "q2_fit_2d.png")
plt.savefig(out_png, dpi=150)
plt.close()

# ---------- 6. 绘图：损失曲线 ----------
plt.figure(figsize=(8, 4.5))
plt.plot(loss_history, c="#d62728", lw=1.5)
plt.title("梯度下降训练损失 (MSE)")
plt.xlabel("Iteration"); plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
loss_png = os.path.join(OUT_DIR, "q2_loss_2d.png")
plt.savefig(loss_png, dpi=150)
plt.close()

# ---------- 7. 保存文字结果 ----------
lines = [
    "二元非线性函数拟合（自动求导 + 梯度下降）",
    f"样本数 {N_SAMPLES}, x1/x2 ∈ [{RANGE_LO}, {RANGE_HI}], 噪声 ∈ [{NOISE_LO}, {NOISE_HI}]",
    f"学习率 {LR}, 迭代 {EPOCHS} 次",
    "",
    "参数对比:",
]
for n, fv, tv in zip(names, a_val, TRUE_A.tolist()):
    lines.append(f"  {n}: 拟合 {fv:9.4f}  真实 {tv:7.4f}  误差 {abs(fv - tv):.4f}")
lines.append(f"  b : 拟合 {b_val:9.4f}  真实 {TRUE_B:7.4f}  误差 {abs(b_val - TRUE_B):.4f}")
lines.append("")
lines.append(f"决定系数 R² = {r2:.6f}")
lines.append(f"最终 MSE = {loss_history[-1]:.6f}")

result = "\n".join(lines) + "\n"
with open(os.path.join(OUT_DIR, "q2_result_2d.txt"), "w", encoding="utf-8") as f:
    f.write(result)

print("\n" + result)
print(f"曲面对比图: {out_png}")
print(f"损失曲线: {loss_png}")
