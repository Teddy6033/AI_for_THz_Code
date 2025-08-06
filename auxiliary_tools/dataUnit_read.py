import os

import numpy as np
import torch

from auxiliary_tools import parseUnit


class DataUnit(object):

    def __init__(self, args, file_path):
        self.wavelength_nor = []
        self.valley_value = None
        self.valley_index = None
        self.file_path = file_path
        self.args = args

        self.structure_nor = []
        self.structure_raw = []

        self.wavelength_total = []
        self.spectrum_total = []

        self.get_structure()
        self.get_spectrum()

    def is_dight(self, s):
        # 尝试将字符串转换为浮点数
        try:
            float_value = float(s)
        except ValueError:
            return False
        return True

    def get_structure(self):
        file_name, file_suffix = os.path.splitext(self.file_path)
        file_name_split = file_name.replace("=", " ").replace(";", " ").split()
        file_name_split = file_name_split[:16]
        structure_raw = []
        structure_nor = []
        dight_split = []
        for v in file_name_split:
            if self.is_dight(v):
                dight_split.append(float(v))
        structure_raw = dight_split[:4]
        for index, value in enumerate(structure_raw):
            value = round(((value - self.args.structure_range[index][0]) /
                           (self.args.structure_range[index][1] - self.args.structure_range[index][0])), 6)
            structure_nor.append(value)
        self.structure_raw = structure_raw
        self.structure_nor = structure_nor
        self.F = dight_split[4]
        self.I = dight_split[5]
        self.Q = dight_split[6]
        self.RH = dight_split[7]

    def get_spectrum(self):
        with open(self.file_path) as f:
            line = f.readline()
            while line:
                if line.split().__len__() == 2:
                    wavelength, spectrum = line.split()
                    self.wavelength_total.append(wavelength)
                    self.spectrum_total.append(spectrum)
                line = f.readline()

        self.wavelength_nor = [float(item) for item in self.wavelength_total]
        # self.wavelength_total = [round((wavelength - 0.8) / 0.5, 6) for wavelength in self.wavelength_total]
        self.wavelength_total = np.linspace(0, 1, 1001)
        self.spectrum_total = [float(item) for item in self.spectrum_total]

    def find_local_maximum(self, data, h=0.0005):
        n = len(data)
        if n < 3:
            return []  # 数据点不足，无法找到局部极小值点

        local_maximum = []

        # 计算一阶导数
        first_derivatives = []
        for i in range(1, n - 1):
            first_derivative = (data[i + 1] - data[i - 1]) / (2 * h)
            first_derivatives.append(first_derivative)

        # 寻找局部极小值点
        for i in range(1, n - 2):
            if first_derivatives[i - 1] >= 0 and first_derivatives[i] <= 0:
                local_maximum.append(i)

        return local_maximum

    def find_local_minima(self, data, h=0.0005):
        n = len(data)
        if n < 3:
            return []  # 数据点不足，无法找到局部极小值点

        local_minima = []

        # 计算一阶导数
        first_derivatives = []
        for i in range(1, n - 1):
            first_derivative = (data[i + 1] - data[i - 1]) / (2 * h)
            first_derivatives.append(first_derivative)

        # 寻找局部极小值点
        for i in range(1, n - 2):
            if first_derivatives[i - 1] <= 0 and first_derivatives[i] >= 0:
                local_minima.append(i)

        return local_minima

    def find_second_derivative_max_at_minima(self, data, minima_indices, h=0.0005):
        if not minima_indices:
            return None

        max_second_derivative_value = float('-inf')
        max_second_derivative_index = None

        # 计算每个局部极小值点的二阶导数
        for idx in minima_indices:
            if idx > 0 and idx < len(data) - 1:
                second_derivative = (data[idx + 1] - 2 * data[idx] + data[idx - 1]) / (h ** 2)
                if second_derivative > max_second_derivative_value:
                    max_second_derivative_value = second_derivative
                    max_second_derivative_index = idx

        return (max_second_derivative_index, data[max_second_derivative_index], max_second_derivative_value)

    def get_valley(self):
        # self.valley_index = np.argmin(self.spectrum_total)
        # 找出所有局部极小值点
        minima_indices = self.find_local_minima(self.spectrum_total)
        # 在局部极小值点中找到二阶导数最大的点
        (self.valley_index, self.valley_value, self.max_second_derivative_value) = self.find_second_derivative_max_at_minima(
            self.spectrum_total,minima_indices)

if __name__ == "__main__":
    args = parseUnit.MyParse().args

