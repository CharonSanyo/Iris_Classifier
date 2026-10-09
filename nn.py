from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_loader import iris_dataloader


# 定义神经网络
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))

        # CrossEntropyLoss 接收原始分数，不提前做 softmax
        x = self.layer3(x)
        return x


# 验证或测试：返回平均损失和准确率
def evaluate(model, loader, loss_f, device):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)

            outputs = model(data)
            loss = loss_f(outputs, label)

            predictions = outputs.argmax(dim=1)
            batch_size = label.size(0)

            loss_sum += loss.item() * batch_size
            correct += (predictions == label).sum().item()
            total += batch_size

    return loss_sum / total, correct / total


# 绘制并保存训练、验证损失曲线
def plot_loss(train_losses, val_losses, save_path):
    epoch_numbers = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epoch_numbers,
        train_losses,
        marker="o",
        markersize=4,
        color="royalblue",
        label="Training loss"
    )

    ax.plot(
        epoch_numbers,
        val_losses,
        marker="s",
        markersize=4,
        color="darkorange",
        label="Validation loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average cross-entropy loss")
    ax.set_title("Iris Classification - Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)

    print("Loss 曲线保存位置：", save_path)

    plt.show()
    plt.close(fig)


def main(lr=0.005, epochs=200):
    # 固定随机种子，便于比较实验
    torch.manual_seed(42)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print("运行设备：", device)

    # 所有文件路径以 nn.py 所在文件夹为起点
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "Iris_data.txt"

    custom_dataset = iris_dataloader(data_path)

    # 划分训练集、验证集和测试集
    train_size = int(len(custom_dataset) * 0.7)
    val_size = int(len(custom_dataset) * 0.2)
    test_size = len(custom_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 只使用训练集计算标准化参数
    train_features = custom_dataset.data[train_dataset.indices]
    mean = train_features.mean(dim=0)
    std = train_features.std(
        dim=0, unbiased=False
    ).clamp_min(1e-8)

    # 三个子集共享底层数据，统一使用训练集的均值和标准差
    custom_dataset.data = (custom_dataset.data - mean) / std

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    print(
        f"训练集：{len(train_dataset)},"
        f"验证集：{len(val_dataset)},"
        f"测试集：{len(test_dataset)}"
    )

    model = NN(4, 12, 6, 3).to(device)
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 创建输出文件夹
    results_dir = project_dir / "results"
    weights_dir = results_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = weights_dir / "nn.pth"
    figure_path = results_dir / "loss_curve.png"

    # 保存每一轮的损失
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()

        loss_sum = 0.0
        correct = 0
        total = 0

        # 每轮结束清除进度条，只保留统计信息
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=False,
            ascii=True
        )

        for data, label in train_bar:
            data = data.to(device)
            label = label.to(device)

            # 清空上一批次的梯度
            optimizer.zero_grad()

            # 前向计算
            outputs = model(data)
            loss = loss_f(outputs, label)

            # 反向传播并更新参数
            loss.backward()
            optimizer.step()

            # 累计这一轮的统计数据
            predictions = outputs.argmax(dim=1)
            batch_size = label.size(0)

            correct += (predictions == label).sum().item()
            total += batch_size
            loss_sum += loss.item() * batch_size

            train_bar.set_postfix(
                batch_loss=f"{loss.item():.3f}"
            )

        train_loss = loss_sum / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model, val_loader, loss_f, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"train_acc: {train_acc:.2%} | "
            f"val_acc: {val_acc:.2%}"
        )

        # 按验证集损失选择最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "epoch": best_epoch
                },
                checkpoint_path
            )

    print("\n训练完成!")
    print(f"验证集损失最低的轮次：第 {best_epoch} 轮")

    # 只在训练结束后，用最佳模型评估测试集
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model, test_loader, loss_f, device
    )

    print(f"测试集损失：{test_loss:.4f}")
    print(f"测试集准确率：{test_acc:.2%}")
    print("模型保存位置：", checkpoint_path)

    # 绘制所有轮次的损失曲线
    plot_loss(train_losses, val_losses, figure_path)


if __name__ == "__main__":
    main(lr=0.005, epochs=20)