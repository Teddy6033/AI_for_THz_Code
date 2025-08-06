import bz2
import pickle

import os
import numpy as np
import shap
import torch
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Arial'
import auxiliary_tools.parseUnit as parseUtils
from physical_mining_feature_prediction.feature_prediction_network import ForwardNetMlp as forward_design_net

args = parseUtils.MyParse(debug=False).args
args.device = "cpu"
model = forward_design_net().to(args.device)

features = ["d2", "d1", "gx", "gy"]
# 创建带下标和斜体的标签
features = [f"${f[0].lower()}_{f[1]}$" for f in features]

forward_tar = r"../data_space/checkpoints/feature-I-prediction-network-checkpoint.pth.tar"
if os.path.exists(forward_tar):
    print("load forward checkpoint...")
    f_checkpoint = torch.load(forward_tar)
    model.load_state_dict(f_checkpoint['state_dict'])
    print(f_checkpoint['epoch'])
else:
    print("no found")

# 从压缩文件中读取五个列表
with bz2.BZ2File(r'..\data_space\dataset_pbz2\unloaded_dataset_val_for_science.pbz2', 'rb') as f:
    X1, X2, X3, X4, F, Q, I, W = pickle.load(f)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    y_test = np.array(F)
    X_test = np.column_stack((X1, X2, X3, X4))

# 从压缩文件中读取五个列表
with bz2.BZ2File(r'..\data_space\dataset_pbz2\unloaded_dataset_train_for_science.pbz2', 'rb') as f:
    X1, X2, X3, X4, F, Q, I, W = pickle.load(f)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    y_train = np.array(F)
    X_train = np.column_stack((X1, X2, X3, X4))

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

features = ["d2", "d1", "gx", "gy"]
# 创建带下标和斜体的标签
features = [f"${f[0].lower()}_{f[1]}$" for f in features]
explainer = shap.DeepExplainer(model, X_train_tensor)
# explainer = shap.KernelExplainer(model, X_train)

pbz2_path = r'..\data_space\dataset_pbz2\shap_values_I.pbz2'
if os.path.exists(pbz2_path):
    # 从压缩文件中读取
    with bz2.BZ2File(pbz2_path, 'rb') as f:
        shap_values = pickle.load(f)
else:
    shap_values = explainer.shap_values(X_test_tensor).squeeze()
    # 打包保存到压缩文件中
    with bz2.BZ2File(pbz2_path, 'wb') as f:
        pickle.dump(shap_values, f)

# 自定义 RGB 值的渐变色
# colors = [(0x0A/255, 0x6F/255, 0xB4/255),(0xEB/255, 0x30/255, 0x32/255)]
colors = [(44/255, 88/255, 156/255),(247/255, 228/255, 228/255),(177/255, 36/255, 36/255)]
# 创建线性渐变 colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

shap_values = shap_values[:, [2, 3, 1, 0]]
features = ["gx", "gy", "d1", "d2"]
# 创建带下标和斜体的标签
features = [f"${f[0].lower()}_{f[1]}$" for f in features]
X_test = X_test[:, [2, 3, 1, 0]]

shap.summary_plot(shap_values, X_test, feature_names=features, sort=False,cmap=custom_cmap, show=False)
# # 将 ndarray 转换为 DataFrame
# df = pd.DataFrame(shap_values, columns=['d1', 'd2', 'gx', 'gy'])
# # 导出为 Excel 文件
# df.to_excel('F_shap.xlsx', index=True)
# # 将 ndarray 转换为 DataFrame
# df = pd.DataFrame(X_test, columns=['d1', 'd2', 'gx', 'gy'])
# # 导出为 Excel 文件
# df.to_excel('F_real.xlsx', index=True)
# 获取当前 Axes 对象
ax = plt.gca()
ax.set_xlim(-0.5,0.5)
# 获取当前图形
fig = plt.gcf()
# 调整图形的宽高比（例如，设置宽度为12，高度为6）
fig.set_size_inches(4, 3)
plt.title("I",fontsize=15)
# 保存为高分辨率图像
# plt.savefig(r'..\data_space\shap_figures\shap dot on I.png', dpi=1200)  # 300 DPI for high resolution
plt.show()
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=features)
# shap.summary_plot(shap_values, X_test, plot_type="violin", feature_names=features)


# # 绘制 SHAP Dependence Plot
# shap.dependence_plot(0, shap_values, X_test, feature_names=features, interaction_index=None)
# shap.dependence_plot(1, shap_values, X_test, feature_names=features, interaction_index=None)
# shap.dependence_plot(2, shap_values, X_test, feature_names=features, interaction_index=None)
# shap.dependence_plot(3, shap_values, X_test, feature_names=features, interaction_index=None)

# # 绘制 d1 SHAP Dependence Plot
# shap.dependence_plot(0, shap_values, X_test, interaction_index=1, feature_names=features)
# shap.dependence_plot(0, shap_values, X_test, interaction_index=2, feature_names=features)
# shap.dependence_plot(0, shap_values, X_test, interaction_index=3, feature_names=features)
# # 绘制 d2 SHAP Dependence Plot
# shap.dependence_plot(1, shap_values, X_test, interaction_index=0, feature_names=features)
# shap.dependence_plot(1, shap_values, X_test, interaction_index=2, feature_names=features)
# shap.dependence_plot(1, shap_values, X_test, interaction_index=3, feature_names=features)
# # 绘制 gx SHAP Dependence Plot
# shap.dependence_plot(2, shap_values, X_test, interaction_index=0, feature_names=features)
# shap.dependence_plot(2, shap_values, X_test, interaction_index=1, feature_names=features)
# shap.dependence_plot(2, shap_values, X_test, interaction_index=3, feature_names=features)
# # 绘制 gy SHAP Dependence Plot
# shap.dependence_plot(3, shap_values, X_test, interaction_index=0, feature_names=features)
# shap.dependence_plot(3, shap_values, X_test, interaction_index=1, feature_names=features)
# shap.dependence_plot(3, shap_values, X_test, interaction_index=2, feature_names=features)


def normalize_list(lst):
    total = sum(lst)
    if total == 0:
        return [0] * len(lst)  # 避免除以0的情况
    return [x / total for x in lst]

# 计算每个特征的平均绝对 SHAP 值
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
importances = normalize_list(mean_abs_shap_values)
print(importances)
plt.bar(["gx","gy","d1","d2"], importances)
plt.xlabel('Features')
plt.ylabel('mean(|SHAP value|)')
plt.title('SHAP on I')
plt.show()