import os
import shutil

from auxiliary_tools import parseUnit
from auxiliary_tools.dataUnit import DataUnit

data_dir = r"../data_space/datasets_txt/dataset_test"
output_folder = r"../data_space/datasets_txt/dataset_test2"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_list = os.listdir(data_dir)
args = parseUnit.MyParse().args

def save_output_params(output_folder, file_name, spectrum_total):

    file_path = os.path.join(output_folder, file_name)
    with open(file_path, 'w') as file:
        for i, param in enumerate(spectrum_total):
            c = f"{(i/1000*0.5+0.8):.6f} {param:.6f}\n"
            file.write(c)

for i, file_name in enumerate(file_list):

    file_path = os.path.join(data_dir, file_name)
    d = DataUnit(args, file_path)
    spectrum_total = [x ** 2 for x in d.spectrum_total]
    save_output_params(output_folder, file_name, spectrum_total)
    print("已完成：" + str(i + 1) + "/" + str(file_list.__len__()))
