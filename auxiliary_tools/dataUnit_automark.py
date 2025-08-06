import os
import numpy as np
from scipy import interpolate, optimize

from auxiliary_tools import parseUnit


class DataUnit(object):

    def __init__(self, args, file_path, RH=None):
        self.I = None
        self.Q = None
        self.RH = RH
        self.W = None
        self.F = None
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
        self.get_valley()
        self.get_F_I_Q_RH()

        self.is_abandon()

    def is_abandon(self):

        self.abandon_flag = None
        if self.F is None or self.I is None or self.Q is None or self.RH is None:
            self.abandon_flag = True
        if self.max_second_derivative_value < 200:
            self.abandon_flag = True
        # if self.F < 0.87 or self.F > 1.23:
        #     self.abandon_flag = True
        if self.structure_raw.__len__() != 4:
            self.abandon_flag = True
        elif self.spectrum_total.__len__() != 1001:
            self.abandon_flag = True

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
        file_name_split = file_name_split[:8]
        structure_raw = []
        structure_nor = []
        for v in file_name_split:
            if self.is_dight(v):
                structure_raw.append(float(v))
        for index, value in enumerate(structure_raw):
            value = round(((value - self.args.structure_range[index][0]) /
                           (self.args.structure_range[index][1] - self.args.structure_range[index][0])), 6)
            structure_nor.append(value)
        self.structure_raw = structure_raw
        self.structure_nor = structure_nor

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

    def get_valley(self):

        def find_local_minima(data, h=0.0005):
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

        def find_second_derivative_max_at_minima(data, minima_indices, h=0.0005):
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

        # 找出所有局部极小值点
        minima_indices = find_local_minima(self.spectrum_total)
        # 在局部极小值点中找到二阶导数最大的点
        (self.valley_index, self.valley_value, self.max_second_derivative_value) = find_second_derivative_max_at_minima(
            self.spectrum_total, minima_indices)

    def get_F_I_Q_RH(self):

        self.F = self.valley_index / 1000 * 0.5 + 0.8

        def find_local_maximum(data, h=0.0005):
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

        maximum_indices = find_local_maximum(self.spectrum_total)
        maximum_pos = [i / 1000 * 0.5 + 0.8 for i in maximum_indices]
        maximum_pos.append(0.8)
        maximum_pos.append(1.3)

        def find_one_closest_values(lst, target):
            # 计算每个元素与指定值的差值，并生成 (差值, 元素) 的元组列表
            diffs = [(abs(x - target), x) for x in lst]
            # 根据差值进行排序
            diffs.sort()
            # 选择差值最小的两个元素
            closest_values = diffs[0][1]
            return closest_values

        # 找到与指定值最接近的两个值
        maximum_pos = find_one_closest_values(maximum_pos, self.F)
        if self.RH is None:
            self.RH = self.spectrum_total[int((maximum_pos - 0.8) * 2000)]
            RH_index = self.spectrum_total.index(self.RH)
            self.RH_pos = RH_index / 1000 * 0.5 + 0.8
        else:
            self.RH_pos = None

        hh = (self.RH + self.valley_value) / 2

        x = [(i / 1000) * 0.5 + 0.8 for i in range(1001)]
        y = self.spectrum_total
        # 创建三次样条插值函数
        f = interpolate.interp1d(x, y, kind="cubic")
        # 特定的y值
        y_target = hh

        # 定义新的函数 g(x) = f(x) - y_target
        def g(x):
            return f(x) - y_target

        # 找到 g(x) = 0 的根
        x_found = []
        # 检查插值范围内的每个区间
        for i in range(len(x) - 1):
            x0, x1 = x[i], x[i + 1]
            if g(x0) * g(x1) < 0:  # 检查是否跨越y_target
                root = optimize.root_scalar(g, bracket=[x0, x1]).root
                x_found.append(root)
        # print(f"y = {y_target} 时对应的 x 值有：{x_found}")

        if x_found.__len__() > 2:
            def find_two_closest_values(lst, target):
                # 计算每个元素与指定值的差值，并生成 (差值, 元素) 的元组列表
                diffs = [(abs(x - target), x) for x in lst]
                # 根据差值进行排序
                diffs.sort()
                # 选择差值最小的两个元素
                closest_values = [diffs[0][1], diffs[1][1]]
                return closest_values

            # 找到与指定值最接近的两个值
            x_found = find_two_closest_values(x_found, self.F)

        if x_found.__len__() == 2:
            f1 = min(x_found)
            f2 = max(x_found)
            fwhm = f2 - f1
            self.W = fwhm
            self.Q = self.F / fwhm
            self.I = self.RH - self.valley_value


if __name__ == "__main__":

    data_dir = r"../data_space/datasets_txt/dataset_test"
    file_list = os.listdir(data_dir)  # 获取数据文件列表
    args = parseUnit.MyParse().args
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        d = DataUnit(args, file_path)
        print(d.valley_index)
        print(d.valley_value)
