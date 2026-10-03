# -*- coding: utf-8 -*-
"""
task4（时序模型方向）：用 LSTM 预测 Air Passengers 航空旅客数量。

数据集：1949-1960 年每月国际航班旅客数，共 144 条月度数据。
任务：用过去 12 个月的旅客数预测下一个月（滑动窗口预测）。

流程：
    1. 读取 CSV，按时间顺序排列
    2. 只用训练集统计做 min-max 归一化，避免数据泄露
    3. 构造滑动窗口样本：前 12 个月 -> 第 13 个月
    4. 训练 LSTM，用 MSE 损失 + Adam 优化器
    5. 在测试集上预测，计算 MAE / RMSE
    6. 额外预测未来 12 个月，观察模型是否学到趋势
"""
import os
import urllib.request

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "airline-passengers.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/airline-passengers.csv")

WINDOW = 12      # 用过去 12 个月预测下一个月
EPOCHS = 600
LR = 1e-3

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 1. 准备数据 ----------
if not os.path.exists(CSV_PATH):
    print("本地没有数据集，开始下载...")
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)

df = pd.read_csv(CSV_PATH)
series = df["Passengers"].to_numpy(dtype=np.float32)
months = df["Month"].tolist()
print(f"数据量: {len(series)} 个月 ({months[0]} ~ {months[-1]})")

# 按时间划分：前 72% 训练，后 28% 测试
split = int(len(series) * 0.72)
train_series = series[:split]
test_series = series[split - WINDOW:]  # 保留窗口历史，测试集可完整预测

# 归一化只用训练集统计量，防止测试信息泄露
data_min, data_max = train_series.min(), train_series.max()
def normalize(arr):
    return (arr - data_min) / (data_max - data_min)
def denormalize(arr):
    return arr * (data_max - data_min) + data_min

def make_windows(arr):
    """把一维序列切成 (前12个月, 下一个月) 样本。"""
    x, y = [], []
    for i in range(len(arr) - WINDOW):
        x.append(arr[i:i + WINDOW])
        y.append(arr[i + WINDOW])
    return (torch.tensor(np.array(x)).float().unsqueeze(-1),
            torch.tensor(np.array(y)).float())

train_x, train_y = make_windows(normalize(train_series))
test_x, test_y = make_windows(normalize(test_series))
print(f"训练样本: {len(train_x)}, 测试样本: {len(test_x)}")

# ---------- 2. 定义 LSTM 模型 ----------
class AirLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)     # out: (batch, seq, hidden)
        return self.fc(out[:, -1, :]).squeeze(-1)  # 取最后一步

model = AirLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------- 3. 训练 ----------
losses = []
for epoch in range(EPOCHS):
    model.train()
    pred = model(train_x.to(device))
    loss = criterion(pred, train_y.to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1:4d}/{EPOCHS}  loss={loss.item():.6f}")

# ---------- 4. 测试集评估 ----------
model.eval()
with torch.no_grad():
    pred_norm = model(test_x.to(device)).cpu().numpy()

pred_real = denormalize(pred_norm)          # 反归一化回真实人数
true_real = denormalize(test_y.numpy())

mae = float(np.mean(np.abs(pred_real - true_real)))
rmse = float(np.sqrt(np.mean((pred_real - true_real) ** 2)))
mean_value = float(np.mean(true_real))
print(f"\n测试集 MAE  = {mae:.2f} 人")
print(f"测试集 RMSE = {rmse:.2f} 人")
print(f"测试集均值  = {mean_value:.2f} 人")
print(f"MAE 占均值比例 = {mae / mean_value:.2%}")

# ---------- 5. 预测未来 12 个月 ----------
with torch.no_grad():
    last_window = normalize(series[-WINDOW:])
    future = []
    for _ in range(12):
        x = torch.tensor(last_window[-WINDOW:]).float().reshape(1, WINDOW, 1)
        nxt = model(x.to(device)).cpu().numpy()[0]
        future.append(nxt)
        last_window = np.append(last_window, nxt)
future_real = denormalize(np.array(future))

# ---------- 6. 绘图 ----------
# 图1：训练损失
plt.figure(figsize=(8, 4.5))
plt.plot(losses, c="#d62728", lw=1.5)
plt.title("LSTM 训练损失 (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, "lstm_loss.png")
plt.savefig(loss_path, dpi=150)
plt.close()

# 图2：真实值 vs 预测值（测试区间）
test_months = months[split:]
plt.figure(figsize=(12, 5))
plt.plot(months[:split], series[:split], c="#7f7f7f", label="训练区间真实值")
plt.plot(test_months, series[split:], c="#1f77b4", lw=2, label="测试区间真实值")
plt.plot(test_months, pred_real, c="#d62728", ls="--", lw=2, label="测试区间预测值")
step = 12
plt.xticks(range(0, len(months), step), [months[i] for i in range(0, len(months), step)],
           rotation=45, fontsize=8)
plt.title("Air Passengers：LSTM 预测结果")
plt.ylabel("旅客人数")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
pred_path = os.path.join(OUT_DIR, "lstm_prediction.png")
plt.savefig(pred_path, dpi=150)
plt.close()

# 图3：未来 12 个月预测
future_months = [f"1961-{m+1:02d}" for m in range(12)]
plt.figure(figsize=(12, 5))
plt.plot(months[-36:], series[-36:], c="#1f77b4", lw=2, label="历史真实值")
plt.plot(future_months, future_real, c="#2ca02c", ls="--", marker="o",
         ms=4, lw=2, label="未来 12 个月预测")
plt.xticks(rotation=45, fontsize=8)
plt.title("LSTM 对 1961 年旅客人数的预测")
plt.ylabel("旅客人数")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
future_path = os.path.join(OUT_DIR, "lstm_future.png")
plt.savefig(future_path, dpi=150)
plt.close()

# ---------- 7. 保存结果 ----------
result = (
    f"模型: LSTM (hidden=64, layers=2), 窗口={WINDOW} 个月\n"
    f"训练: Adam lr={LR}, epochs={EPOCHS}, MSE 损失\n"
    f"数据: 1949-1960 共 {len(series)} 个月, 训练 {split} / 测试 {len(series)-split}\n"
    f"测试集 MAE  = {mae:.2f} 人\n"
    f"测试集 RMSE = {rmse:.2f} 人\n"
    f"测试集均值  = {mean_value:.2f} 人\n"
    f"MAE 占均值比例 = {mae / mean_value:.2%}\n"
    f"未来12个月预测: {np.round(future_real).astype(int).tolist()}\n"
)
with open(os.path.join(OUT_DIR, "lstm_result.txt"), "w", encoding="utf-8") as f:
    f.write(result)

print(f"损失曲线: {loss_path}")
print(f"预测对比: {pred_path}")
print(f"未来预测: {future_path}")
