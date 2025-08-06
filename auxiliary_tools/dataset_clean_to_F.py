from matplotlib import pyplot as plt
import numpy as np
import torch
import auxiliary_tools.parseUnit as parseUtils
import math
import os
from scipy import interpolate

from auxiliary_tools import parseUnit
from auxiliary_tools.dataUnit import DataUnit

data_dir = r"../data_space/datasets_txt/dataset_test"
file_list = os.listdir(data_dir)
args = parseUnit.MyParse().args

def curve_show():

    global button_press_flag, nearly_point_index, update_nearlyPoint_flag, delete_flag
    button_press_flag = 0
    nearly_point_index = 0
    update_nearlyPoint_flag = 0
    delete_flag = None
    global true_spectrum, predict_spectrum, predict_structure
    true_spectrum = []
    predict_spectrum = []
    predict_structure = []

    def key_press(event):

        global delete_flag
        if event.key == 'p':  # 定义按键为 't'
            print("按下了 'p' 键")
            # 下一个文件展示
            delete_flag = None
        elif event.key == 'd':  # 定义按键为 't'
            print("按下了 'd' 键")
            # 删除当前文件，展示下一个文件
            delete_flag = True


    def data_trans_inverse(data):
        data = data.view(1, -1)
        scale = args.structure_range
        for i in range(0, data.shape[1]):
            data[0][i] = data[0][i] * (scale[i][1] - scale[i][0]) + scale[i][0]
        return data

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

    def scan_folder():

        global delete_flag
        nums = file_list.__len__()
        for index, file_name in enumerate(file_list):
            fig = plt.figure()
            # fig.canvas.mpl_connect('button_press_event', button_press)
            # fig.canvas.mpl_connect('button_release_event', button_release)
            # fig.canvas.mpl_connect('motion_notify_event', motion)
            fig.canvas.mpl_connect('key_press_event', key_press)

            file_path = os.path.join(data_dir, file_name)
            d = DataUnit(args, file_path)
            structure_raw = d.structure_raw
            wavelength_nor = d.wavelength_nor
            spectrum_total = d.spectrum_total
            plt.xlim(0.8, 1.3)
            plt.ylim(0, 1)
            plt.title("d1:" + str(round(structure_raw[0], 0)) + "d2:" + str(round(structure_raw[1], 0))
                      + "gx:" + str(round(structure_raw[2], 0)) + "gy:" + str(round(structure_raw[3], 0)),
                      fontsize=14, fontstyle='italic')
            plt.ylabel('Reflectance', {'family': 'Times New Roman', 'weight': 'normal', 'size': 20})
            plt.xlabel('Wavelength(nm)', {'family': 'Times New Roman', 'weight': 'normal', 'size': 20})
            plt.plot(wavelength_nor, spectrum_total, label="spectram", linewidth=3, )

            # 找出所有局部极小值点
            minima_indices = find_local_minima(d.spectrum_total)
            print("所有局部极小值点的索引:", minima_indices)
            for i in minima_indices:
                plt.axvline(x=i / 1000 * 0.5 + 0.8, color='b', linestyle='--')

            # 找出所有局部极小值点
            maximum_indices = find_local_maximum(d.spectrum_total)
            print("所有局部极小值点的索引:", maximum_indices)
            for i in maximum_indices:
                plt.axvline(x=i / 1000 * 0.5 + 0.8, color='r', linestyle='--')

            # 在局部极小值点中找到二阶导数最大的点
            (d.valley_index, _, max_second_derivative_value) = find_second_derivative_max_at_minima(d.spectrum_total, minima_indices)
            plt.scatter(d.valley_index/1000*0.5+0.8, d.spectrum_total[d.valley_index], s=200, c="purple")
            print("最大二阶导大小:", max_second_derivative_value)

            plt.legend()
            plt.show(block=True)
            # plt.pause(1)
            plt.close()

            if delete_flag:
                os.remove(file_path)
                print("删除文件" + file_name)
            else:
                print("保留文件")
            delete_flag = None

            print(f"已阅 {index+1} / {nums}")

    scan_folder()

if __name__ == '__main__':
    curve_show()
