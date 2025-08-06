import os
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader
from auxiliary_tools import parseUnit
import auxiliary_tools.dataUnit_read as dataUnit
from auxiliary_tools.dataUnit_automark import DataUnit as DataUnit_2


class DataSet(Dataset):
    def __init__(self, args, set_type, dataset):
        super(DataSet, self).__init__()

        self.args = args
        self.set_type = set_type
        self.data_dir = os.path.join(self.args.root_dir, "data_space/datasets_txt", dataset, set_type)
        self.file_list = os.listdir(self.data_dir)  # 获取数据文件列表

        self.devices = []
        self.devices_path = []
        self.devices_out = []
        self.devices_out_path = []

        self.scan_folders()

    def __len__(self):
        return len(self.file_list)

    class deviceClass:
        pass

    def scan_folders(self):

        nums = len(self.file_list)

        for file_name in self.file_list:
            if file_name.endswith(".txt"):
                match = re.search(r'RH=(.*?)\;', file_name)
                if match:
                    rh = float(match.group(1))
                else:
                    print("未找到匹配项")
                file_path = os.path.join(self.data_dir, file_name)
                device_raw = DataUnit_2(args, file_path, RH=rh)
                if (device_raw.structure_nor.__len__() + device_raw.spectrum_total.__len__() +
                        device_raw.wavelength_total.__len__() == 2006):
                    device = self.deviceClass()
                    device.structure_nor = torch.Tensor(device_raw.structure_nor)
                    device.structure_raw = torch.Tensor(device_raw.structure_raw)
                    device.spectrum_total = torch.Tensor(device_raw.spectrum_total)
                    device.F_nor = torch.Tensor([(device_raw.F-0.8)*2])
                    # device.Q = torch.Tensor([device_raw.Q])
                    device.Q_nor = torch.Tensor([device_raw.Q/200])
                    device.I = torch.Tensor([device_raw.I])
                    device.W = torch.Tensor([device_raw.W*20])

                    self.devices.append(device)
                else:
                    self.devices_out.append(device_raw)
                    self.devices_out_path.append(file_name)

        print("find {} sample: {}".format(self.set_type, self.devices.__len__()))

    def __getitem__(self, index):
        device = self.devices[index]
        return device.structure_nor, device.F_nor, device.Q_nor, device.I, device.W


if __name__ == "__main__":
    args = parseUnit.MyParse().args
    # 创建自定义数据集
    dataset_train = DataSet(args, "train", "unloaded_dataset_for_science")
    print(dataset_train.devices.__len__())
    # print(dataset_train.devices_out.__len__())
    # print(dataset_train.devices_out_path)
    dataset_val = DataSet(args, "val", "unloaded_dataset_for_science")
    print(dataset_val.devices.__len__())
    # print(dataset_val.devices_out.__len__())
    # print(dataset_val.devices_out_path)

    state_info = {
        "dataset_train": dataset_train,
        "dataset_val": dataset_val,
    }
    torch.save(state_info, r"..\data_space\datasets_tar\unloaded_dataset_for_science_test.pth")
