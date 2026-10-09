from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class iris_dataloader(Dataset):
    def __init__(self, data_path):
        data_path = Path(data_path)

        if not data_path.is_file():
            raise FileNotFoundError(f"找不到数据文件：{data_path}")

        # 数据文件没有表头，每行有四个特征和一个类别
        df = pd.read_csv(
            data_path,
            header=None,
            names=[0, 1, 2, 3, 4]
        )

        class_to_id = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2
        }

        labels = df[4].astype(str).str.strip().map(class_to_id)

        if labels.isna().any():
            raise ValueError(
                "存在无法识别的类别，请检查数据文件最后一列。"
            )

        # 特征：[样本数, 4]，浮点数
        self.data = torch.from_numpy(
            df.iloc[:, :4].to_numpy(dtype="float32")
        )

        # 标签：[样本数]，整数
        self.label = torch.from_numpy(
            labels.to_numpy(dtype="int64")
        )

        if not torch.isfinite(self.data).all():
            raise ValueError("特征中存在缺失值或无穷值。")

        print("当前数据集的大小：", len(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]