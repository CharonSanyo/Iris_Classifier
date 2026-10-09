from pathlib import Path
from torch.utils.data import DataLoader
from data_loader import iris_dataloader

data_path = Path(__file__).resolve().parent / "Iris_data.txt"
dataset = iris_dataloader(data_path)

print("样本总数：", len(dataset))
print("所有特征的形状：", dataset.data.shape)
print("所有标签的形状：", dataset.label.shape)

features, label = dataset[0]
print("第一条特征：", features)
print("第一条标签：", label)

loader = DataLoader(dataset, batch_size=16, shuffle=False)
batch_data, batch_label = next(iter(loader))

print("一批特征的形状：", batch_data.shape)
print("一批标签的形状：", batch_label.shape)    