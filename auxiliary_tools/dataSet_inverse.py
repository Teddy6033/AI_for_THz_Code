import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from auxiliary_tools import parseUnit
import auxiliary_tools.dataUnit as dataUnit
import itertools

class DataSet(Dataset):
    def __init__(self, args, set_type, dataset):
        super(DataSet, self).__init__()

        self.args = args
        self.set_type = set_type
        self.data_dir = os.path.join(self.args.root_dir, "data_space\datasets_txt", dataset, set_type)
        self.file_list = os.listdir(self.data_dir)

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

        for file_name in self.file_list:  # random.sample(files, nums):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.data_dir, file_name)
                device_raw = dataUnit.DataUnit(self.args, file_path)
                if (device_raw.structure_nor.__len__() + device_raw.spectrum_total.__len__() +
                        device_raw.wavelength_total.__len__() == 2006):
                    # 生成所有可能的四维向量
                    vectors = list(itertools.product([0, 1], repeat=4))
                    for vector in vectors:
                        device = self.deviceClass()

                        device.real_structure = device_raw.structure_nor
                        device.structure_raw = device_raw.structure_raw
                        device.spectrum_total = device_raw.spectrum_total
                        device.valley_pos = [device_raw.valley_index * 0.001]
                        device.mask = vector
                        device.reverse_mask = [1 if bit == 0 else 0 for bit in vector]
                        device.input_structure = [x * y for x, y in zip(device.real_structure, device.mask)]
                        device.input_structure = [x if x != 0 else -1 for x in device.input_structure]
                        device.input = device.valley_pos + device.input_structure

                        device.real_structure = torch.Tensor(device.real_structure)
                        device.structure_raw = torch.Tensor(device.structure_raw)
                        device.spectrum_total = torch.Tensor(device.spectrum_total)
                        device.valley_pos = torch.Tensor(device.valley_pos)
                        device.mask = torch.Tensor(device.mask)
                        device.reverse_mask = torch.Tensor(device.reverse_mask)
                        device.input_structure = torch.Tensor(device.input_structure)
                        device.input = torch.Tensor(device.input)

                        self.devices.append(device)
                else:
                    self.devices_out.append(device_raw)
                    self.devices_out_path.append(file_name)

        print("find {} sample: {}".format(self.set_type, self.devices.__len__()))

    def __getitem__(self, index):
        device = self.devices[index]
        return (device.input, device.valley_pos, device.real_structure, device.mask,
                device.reverse_mask, device.structure_raw, device.spectrum_total)


if __name__ == "__main__":
    args = parseUnit.MyParse().args

    dataset_train = DataSet(args, "train", "unloaded_dataset_for_technology")
    print(dataset_train.devices.__len__())
    print(dataset_train.devices_out.__len__())
    print(dataset_train.devices_out_path)
    dataset_val = DataSet(args, "val", "unloaded_dataset_for_technology")
    print(dataset_val.devices.__len__())
    print(dataset_val.devices_out.__len__())
    print(dataset_val.devices_out_path)

    state_info = {
        "dataset_train": dataset_train,
        "dataset_val": dataset_val,
    }
    torch.save(state_info, r"..\data_space\datasets_tar\unloaded_dataset_inverse_for_technology_test.pth")