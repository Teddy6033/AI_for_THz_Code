import pickle
import bz2
import os

from auxiliary_tools import parseUnit
from auxiliary_tools.dataUnit_read_plus import DataUnit

def is_float(value):
    return isinstance(value, float)

data_dir = r"../data_space/datasets_txt/unloaded_dataset_for_science/train"
file_list = os.listdir(data_dir)  # 获取数据文件列表
args = parseUnit.MyParse().args
D1 = []
D2 = []
Gx = []
Gy = []
F = []
Q = []
I = []
W = []
for file_id, file_name in enumerate(file_list):
    file_path = os.path.join(data_dir, file_name)
    d = DataUnit(args, file_path)
    if is_float(d.structure_nor[0]) and is_float(d.structure_nor[1]) and is_float(d.structure_nor[2]) and is_float(
            d.structure_nor[3]) and is_float(d.F):
        if 0 < d.structure_nor[0] < 1 and 0 < d.structure_nor[1] < 1 and 0 < d.structure_nor[2] < 1 and 0 < d.structure_nor[3] < 1 and 0.8 < d.F < 1.3:
            D1.append(d.structure_nor[0])
            D2.append(d.structure_nor[1])
            Gx.append(d.structure_nor[2])
            Gy.append(d.structure_nor[3])
            F.append(d.F)
            Q.append(d.Q/200)
            I.append(d.I)
            W.append(d.W * 2)
        else:
            print("有叛徒")
    else:
        print("有叛徒")

# 将五个列表打包保存到压缩文件中
with bz2.BZ2File(r'../data_space/dataset_pbz2/unloaded_dataset_train_for_science_test.pbz2', 'wb') as f:
    pickle.dump((D1, D2, Gx, Gy, F, Q, I, W), f)

data_dir = r"../data_space/datasets_txt/unloaded_dataset_for_science/val"
file_list = os.listdir(data_dir)  # 获取数据文件列表
args = parseUnit.MyParse().args
D1 = []
D2 = []
Gx = []
Gy = []
F = []
Q = []
I = []
W = []
for file_id, file_name in enumerate(file_list):
    file_path = os.path.join(data_dir, file_name)
    d = DataUnit(args, file_path)
    if is_float(d.structure_nor[0]) and is_float(d.structure_nor[1]) and is_float(d.structure_nor[2]) and is_float(
            d.structure_nor[3]) and is_float(d.F):
        if 0 < d.structure_nor[0] < 1 and 0 < d.structure_nor[1] < 1 and 0 < d.structure_nor[2] < 1 and 0 < d.structure_nor[3] < 1 and 0.8 < d.F < 1.3:
            D1.append(d.structure_nor[0])
            D2.append(d.structure_nor[1])
            Gx.append(d.structure_nor[2])
            Gy.append(d.structure_nor[3])
            F.append(d.F)
            Q.append(d.Q/200)
            I.append(d.I)
            W.append(d.W * 2)
        else:
            print("有叛徒")
    else:
        print("有叛徒")

# 将五个列表打包保存到压缩文件中
with bz2.BZ2File(r'../data_space/dataset_pbz2/unloaded_dataset_val_for_science_test.pbz2', 'wb') as f:
    pickle.dump((D1, D2, Gx, Gy, F, Q, I, W), f)


