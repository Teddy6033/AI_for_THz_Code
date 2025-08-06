import os
import shutil

from auxiliary_tools import parseUnit
from auxiliary_tools.dataUnit_automark import DataUnit

input_dir = r"../data_space/datasets_txt/dataset_test"
output_folder = r"../data_space/datasets_txt/dataset_test2"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_list = os.listdir(input_dir)
args = parseUnit.MyParse().args

for i, file_name in enumerate(file_list):
    file_path_input = os.path.join(input_dir, file_name)
    d = DataUnit(args, file_path_input)
    if d.abandon_flag is None:
        file_name_output = f"d1={d.structure_raw[0]};d2={d.structure_raw[1]};gx={d.structure_raw[2]};gy={d.structure_raw[3]};F={d.F:.4f};I={d.I:.4f};Q={d.Q:.4f};RH={d.RH:.4f};W={d.W:.4f}.txt;"
        file_path_output = os.path.join(output_folder, file_name_output)
        shutil.copy(file_path_input, file_path_output)
    else:
        print("本文件未被复制:" + f"{file_name}")

    print("已完成：" + str(i + 1) + "/" + str(file_list.__len__()))