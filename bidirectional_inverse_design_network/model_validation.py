import os
import sys

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

import auxiliary_tools.parseUnit as parseUtils
from bidirectional_forward_design_network.forward_design_network import ForwardNetMlp as F_Net
from auxiliary_tools.dataSet_inverse import DataSet
from inverse_design_network import InverseNet as I_Net

args = parseUtils.MyParse(debug=False).args
dataset = torch.load(r"..\data_space\datasets_tar\unloaded_dataset_inverse_for_technology.pth")
dataset_train = dataset['dataset_train']
dataset_val = dataset['dataset_val']
# 创建数据加载器
train_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=dataset_val, batch_size=64, shuffle=True)

f_model = F_Net().to(args.device)
forward_tar = r"../data_space/checkpoints/bidirectional-forward-design-network-checkpoint.pth.tar"
if os.path.exists(forward_tar):
    print("load forward checkpoint...")
    checkpoint = torch.load(forward_tar)
    f_model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'])
else:
    print("no found")

i_model = I_Net().to(args.device)
inverse_tar = r"../data_space/checkpoints/bidirectional-inverse-design-network-checkpoint.pth.tar"
if os.path.exists(inverse_tar):
    print("load inverse checkpoint...")
    checkpoint = torch.load(inverse_tar)
    i_model.load_state_dict(checkpoint['state_dict'])
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

def evaluate_model(loader, f_model, i_model, device):
    """计算模型在给定数据加载器上的MSE、R²、MAE和RMSE"""
    f_model.eval()
    i_model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0  # 用于累加损失

    with torch.no_grad():
        for index, (input_vs, valley_pos, real_structure, mask,
                    reverse_mask, structure_raw, spectrum_total) in enumerate(loader):
            input_vs = input_vs.to(device)
            valley_pos = valley_pos.to(device)
            real_structure = real_structure.to(device)
            mask = mask.to(device)
            reverse_mask = reverse_mask.to(device)

            # 逆向传播
            structure_out = i_model(input_vs)
            # 中途处理
            structure_1 = structure_out.masked_fill(mask == 0, 0)
            structure_2 = real_structure.masked_fill(mask == 0, 0)
            # loss_structure = nn.MSELoss()(structure_1, structure_2)
            structure_pre = structure_out.masked_fill(reverse_mask == 0, 0)
            structure_in = structure_2 + structure_pre
            # 正向传播
            valley_out = f_model(structure_in)
            loss_valley = nn.MSELoss()(valley_pos, valley_out)
            # 计算损失
            loss = loss_valley

            # 保存预测值和标签
            all_preds.append(valley_out.cpu().numpy())
            all_labels.append(valley_pos.cpu().numpy())


            running_loss += loss.item() * real_structure.size(0) # 累加总损失

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
train_loss, train_rmse, train_r2, train_mae = evaluate_model(train_loader, f_model, i_model, args.device)
print(f"\nTrain MSE: {train_loss:.7f}")  # MSE
print(f"Train RMSE: {train_rmse:.7f}")  # RMSE
print(f"Train R²: {train_r2:.7f}")      # R²
print(f"Train MAE: {train_mae:.7f}")    # MAE

# 验证集评估
val_loss, val_rmse, val_r2, val_mae = evaluate_model(val_loader, f_model, i_model, args.device)
print(f"\nValidation MSE: {val_loss:.7f}")  # MSE
print(f"Validation RMSE: {val_rmse:.7f}")  # RMSE
print(f"Validation R²: {val_r2:.7f}")      # R²
print(f"Validation MAE: {val_mae:.7f}")    # MAE
