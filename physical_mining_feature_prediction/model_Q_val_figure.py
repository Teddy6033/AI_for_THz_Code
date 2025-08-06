import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader

import auxiliary_tools.parseUnit as parseUtils
from feature_prediction_network import ForwardNetMlp as forward_design_net
from auxiliary_tools.dataSet_forward_FQIW import DataSet

args = parseUtils.MyParse(debug=False).args

# 加载数据集
dataset_tar = torch.load(r"../data_space/datasets_tar/unloaded_dataset_for_science.pth")
dataset_train = dataset_tar['dataset_train']
dataset_val = dataset_tar['dataset_val']
train_loader = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=dataset_val, batch_size=64, shuffle=True)

# 加载模型
model = forward_design_net().to(args.device)
forward_tar = r"..\data_space/checkpoints/feature-Q-prediction-network-checkpoint.pth.tar"
if os.path.exists(forward_tar):
    print("load forward checkpoint...")
    checkpoint = torch.load(forward_tar)
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'])
else:
    print("no found")


# 评估模型函数
def evaluate_model(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for structure_nor, F, Q, I, W in loader:
            index = Q
            structure = structure_nor.to(device)
            index = index.to(device)

            # 正向传播
            predict_index = model(structure)

            # 保存预测值和标签
            all_preds.append(predict_index.cpu().numpy())
            all_labels.append(index.cpu().numpy())

    # 合并所有预测和标签
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return preds, labels


# 从训练集和验证集中获取预测值和真实值
train_preds, train_labels = evaluate_model(train_loader, model, args.device)
val_preds, val_labels = evaluate_model(val_loader, model, args.device)

# 设置随机种子并抽取1000个样本
np.random.seed(190730)
train_indices = np.random.choice(len(train_preds), 1000, replace=False)
val_indices = np.random.choice(len(val_preds), 1000, replace=False)

# 获取抽样后的数据
train_preds_sample = train_preds[train_indices]
train_labels_sample = train_labels[train_indices]
val_preds_sample = val_preds[val_indices]
val_labels_sample = val_labels[val_indices]

# 绘制散点图
plt.figure(figsize=(10, 6))

plt.xlim(0, 1)
plt.ylim(0, 1)

# 验证集散点图 - 实心蓝色正方形
plt.scatter(val_preds_sample, val_labels_sample, color='blue', marker='s', label='Validation Set', alpha=0.4, s=50)
# 训练集散点图 - 红色圆点
plt.scatter(train_preds_sample, train_labels_sample, color='red', marker='o', label='Training Set', alpha=0.4, s=50)

# 图例和标签
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.legend()
plt.title('Predicted vs True Values for Training and Validation Sets')
plt.show()

# 计算训练集和验证集的均方误差 (MSE)
train_mse = (train_preds - train_labels) ** 2
val_mse = (val_preds - val_labels) ** 2

# 设置MSE的区间范围
mse_min = min(train_mse.min(), val_mse.min())
mse_max = max(train_mse.max(), val_mse.max())
# bins_range = np.linspace(0, 0.01, 6)  # 31个区间
max_mse = max(train_mse.max(), val_mse.max())
bins_range = [0, 0.002, 0.004, 0.006, max_mse + 1e-6]
bin_labels = ['[0, 0.002)', '[0.002, 0.004)', '[0.004, 0.006)', '[0.006, max]']
# 绘制均方误差直方图
# 获取频数
train_counts, _ = np.histogram(train_mse, bins=bins_range)
val_counts, _ = np.histogram(val_mse, bins=bins_range)

# 设置等宽柱位置
x = np.arange(len(bin_labels))  # x = [0, 1, 2, 3]
width = 0.35  # 每组柱子宽度

# 绘图
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, train_counts, width=width, color='red', label='Training', edgecolor='black')
plt.bar(x + width/2, val_counts, width=width, color='blue', label='Validation', edgecolor='black')

# 添加标签
for i in range(len(x)):
    plt.text(x[i] - width/2, train_counts[i] + 1, str(train_counts[i]), ha='center', color='black')
    plt.text(x[i] + width/2, val_counts[i] + 1, str(val_counts[i]), ha='center', color='black')

# 坐标轴设置
plt.xticks(x, bin_labels)
plt.xlabel('MSE Range')
plt.ylabel('Frequency')
plt.title('MSE Distribution for Q(Equal Width Bars)')
plt.legend()
plt.tight_layout()
plt.show()
