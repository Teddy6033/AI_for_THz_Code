import os
import random
from shutil import copyfile


def split_data(source_dir, train_dir, val_dir, split_ratio=0.8, seed=666):
    # 创建训练集和验证集文件夹
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 获取文件列表
    file_list = os.listdir(source_dir)
    # 过滤掉非文件的项
    file_list = [file for file in file_list if os.path.isfile(os.path.join(source_dir, file))]

    # 设置随机种子以确保划分结果可重复
    random.seed(seed)
    random.shuffle(file_list)

    # 计算划分点
    split_point = int(len(file_list) * split_ratio)

    # 将文件复制到训练集和验证集文件夹
    for i, file in enumerate(file_list):
        if i < split_point:
            copyfile(os.path.join(source_dir, file), os.path.join(train_dir, file))
        else:
            copyfile(os.path.join(source_dir, file), os.path.join(val_dir, file))


# 使用示例
source_folder = r"..\data_space\datasets_txt\dataset_test"
train_folder = source_folder + r"\train"
val_folder = source_folder + r"\val"
split_data(source_folder, train_folder, val_folder, split_ratio=0.9)
