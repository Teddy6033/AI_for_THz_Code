import os
import sys

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

import auxiliary_tools.parseUnit as parseUtils
from spectrum_prediction_network import ForwardNetMlp as forward_design_net
from auxiliary_tools.dataSet_forward import DataSet
from spectrum_prediction_network import ForwardNetMlp as F_Net

args = parseUtils.MyParse(debug=False).args

dataset_tar = torch.load(r"../data_space/datasets_tar/unloaded_dataset_forward_for_technology.pth")
dataset_train = dataset_tar['dataset_train']
dataset_val = dataset_tar['dataset_val']
train_loader = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=dataset_val, batch_size=64, shuffle=True)

model = forward_design_net().to(args.device)
forward_tar = r"../data_space/checkpoints/spectrum-unloaded-prediction-network-checkpoint.pth.tar"
if os.path.exists(forward_tar):
    print("load forward checkpoint...")
    checkpoint = torch.load(forward_tar)
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'])
else:
    print("no found")


def data_trans_inverse(data):
    data = data.view(1, -1)
    scale = args.structure_range
    for i in range(0, data.shape[1]):
        data[0][i] = data[0][i] * (scale[i][1] - scale[i][0]) + scale[i][0]
    return data


def data_trans(data):
    scale = args.structure_range
    for i in range(0, data.__len__()):
        data[i] = (data[i] - scale[i][0]) / (scale[i][1] - scale[i][0])
    return data

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def evaluate_model(loader, model, device):
    """计算模型在给定数据加载器上的MSE、R²、MAE和RMSE"""
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0  # 用于累加损失

    with torch.no_grad():
        for index, (structure_nor, valley_pos, spectrum_total, structure_raw) in enumerate(loader):
            structure = structure_nor.to(device)
            spectrum_total = spectrum_total.to(device)

            # 正向传播
            predict_spectrum = model(structure)

            # 保存预测值和标签
            all_preds.append(predict_spectrum.cpu().numpy())
            all_labels.append(spectrum_total.cpu().numpy())

            # 计算损失并累加
            loss = nn.MSELoss()(predict_spectrum, spectrum_total)  # 直接计算MSE
            running_loss += loss.item() * structure.size(0)  # 累加总损失

            # 更新进度
            progress = (index + 1) / len(loader) * 100
            sys.stdout.write(f'\rCurrent Progress: {progress:.2f}%')
            sys.stdout.flush()

    # 合并所有预测和标签
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 计算平均MSE
    mse = running_loss / len(loader.dataset)  # 计算平均MSE
    # 计算RMSE
    rmse = np.sqrt(mse)  # 计算RMSE
    # 计算R²和MAE
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)

    return mse, rmse, r2, mae

# 训练集评估
train_loss, train_rmse, train_r2, train_mae = evaluate_model(train_loader, model, args.device)
print(f"\nTrain MSE: {train_loss:.7f}")  # MSE
print(f"Train RMSE: {train_rmse:.7f}")  # RMSE
print(f"Train R²: {train_r2:.7f}")      # R²
print(f"Train MAE: {train_mae:.7f}")    # MAE

# 验证集评估
val_loss, val_rmse, val_r2, val_mae = evaluate_model(val_loader, model, args.device)
print(f"\nValidation MSE: {val_loss:.7f}")  # MSE
print(f"Validation RMSE: {val_rmse:.7f}")  # RMSE
print(f"Validation R²: {val_r2:.7f}")      # R²
print(f"Validation MAE: {val_mae:.7f}")    # MAE
