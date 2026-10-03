# -*- coding: utf-8 -*-
"""
task3 题三：用 PyTorch 搭建 MLP 识别 MNIST 手写数字。

数据：题目提供的 task3/train-00000-of-00001.parquet（60000 张 28x28 手写数字）。
模型：784 -> 256 -> 128 -> 10 的全连接网络（MLP）。
"""
import io
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

# 无显示环境也能保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train-00000-of-00001.parquet")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 1. 读取并解码数据 ----------
print("读取 parquet 数据...")
df = pd.read_parquet(DATA_PATH)
print(f"总样本数: {len(df)}, 类别数: {df['label'].nunique()}")

def decode_image(row):
    """把 parquet 里的 PNG 字节解码成 28x28 的 numpy 数组。"""
    img = Image.open(io.BytesIO(row["bytes"])).convert("L")
    return np.array(img, dtype=np.float32) / 255.0

# images: (60000, 28, 28)，取值 [0,1]
images = np.stack([decode_image(r) for r in df["image"]])
labels = df["label"].to_numpy(dtype=np.int64)

# 前 50000 张训练，后 10000 张测试
train_x = torch.tensor(images[:50000]).reshape(-1, 28 * 28)
train_y = torch.tensor(labels[:50000])
test_x = torch.tensor(images[50000:]).reshape(-1, 28 * 28)
test_y = torch.tensor(labels[50000:])
print(f"训练集: {train_x.shape}, 测试集: {test_x.shape}")

# ---------- 2. 定义 MLP 模型 ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),   # 输入层 -> 隐藏层1
            nn.ReLU(),
            nn.Dropout(0.2),       # 随机丢弃 20% 神经元，防止过拟合
            nn.Linear(256, 128),   # 隐藏层1 -> 隐藏层2
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),    # 隐藏层2 -> 输出层（10 个数字类别）
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)

# ---------- 3. 训练配置 ----------
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 20
BATCH_SIZE = 128

history = {"loss": [], "train_acc": [], "test_acc": []}

# ---------- 4. 训练循环 ----------
for epoch in range(EPOCHS):
    model.train()
    # 每个打乱后的 mini-batch 更新一次参数
    perm = torch.randperm(len(train_x))
    total_loss, correct = 0.0, 0

    for start in range(0, len(train_x), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        bx = train_x[idx].to(device)
        by = train_y[idx].to(device)

        optimizer.zero_grad()
        out = model(bx)             # 前向传播
        loss = criterion(out, by)   # 计算损失
        loss.backward()             # 反向传播
        optimizer.step()            # 更新参数

        total_loss += loss.item() * len(idx)
        correct += (out.argmax(1) == by).sum().item()

    train_loss = total_loss / len(train_x)
    train_acc = correct / len(train_x)

    # 每轮结束评估测试集
    model.eval()
    with torch.no_grad():
        pred = model(test_x.to(device)).argmax(1).cpu()
        test_acc = (pred == test_y).float().mean().item()

    history["loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)
    print(f"epoch {epoch+1:2d}/{EPOCHS}  loss={train_loss:.4f}  "
          f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

# ---------- 5. 保存模型 ----------
torch.save(model.state_dict(), os.path.join(OUT_DIR, "mlp_mnist.pth"))

# ---------- 6. 绘制训练曲线 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
epochs = range(1, EPOCHS + 1)

axes[0].plot(epochs, history["loss"], marker="o", ms=3, c="#d62728")
axes[0].set_title("训练损失 Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross Entropy Loss")
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, history["train_acc"], marker="o", ms=3,
             label="Train Acc", c="#1f77b4")
axes[1].plot(epochs, history["test_acc"], marker="s", ms=3,
             label="Test Acc", c="#2ca02c")
axes[1].set_title("训练 / 测试准确度")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(OUT_DIR, "q3_curves.png")
plt.savefig(curves_path, dpi=150)
plt.close()

# ---------- 7. 可视化预测结果（8 张测试图） ----------
model.eval()
with torch.no_grad():
    logits = model(test_x.to(device))
    preds = logits.argmax(1).cpu().numpy()

# 从测试集选 8 张（前 4 张预测正确的 + 后 4 张随机的，保证多样）
sample_idx = list(range(4)) + list(np.random.choice(len(test_x), 4, replace=False))
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, idx in zip(axes.flat, sample_idx):
    ax.imshow(test_x[idx].reshape(28, 28), cmap="gray")
    true_label = test_y[idx].item()
    pred_label = preds[idx]
    color = "green" if pred_label == true_label else "red"
    ax.set_title(f"真实: {true_label}  预测: {pred_label}", color=color)
    ax.axis("off")

plt.suptitle("MLP 手写数字识别结果（绿色=正确，红色=错误）", fontsize=13)
plt.tight_layout()
samples_path = os.path.join(OUT_DIR, "q3_samples.png")
plt.savefig(samples_path, dpi=150)
plt.close()

# ---------- 8. 保存结果 ----------
final_test_acc = history["test_acc"][-1]
result = (f"模型: 784-256-128-10 MLP (ReLU + Dropout 0.2)\n"
          f"优化器: Adam (lr=1e-3), batch={BATCH_SIZE}, epochs={EPOCHS}\n"
          f"最终训练准确度: {history['train_acc'][-1]:.4f}\n"
          f"最终测试准确度: {final_test_acc:.4f}\n")
with open(os.path.join(OUT_DIR, "q3_result.txt"), "w", encoding="utf-8") as f:
    f.write(result)

print(result)
print(f"曲线图: {curves_path}")
print(f"预测可视化: {samples_path}")
