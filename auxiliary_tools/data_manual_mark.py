import shutil

from matplotlib import pyplot as plt
import numpy as np
import torch
import auxiliary_tools.parseUnit as parseUtils
import math
import os
from scipy import interpolate

from auxiliary_tools import parseUnit
from auxiliary_tools.dataUnit_read import DataUnit
from auxiliary_tools.dataUnit_automark import DataUnit as DataUnit_2

global file_id, d, file_list
data_dir = r"../data_space/datasets_txt/dataset_test"
file_list = os.listdir(data_dir)

def execute():
    global point, file_id, d
    plt.ion()
    fig, ax = plt.subplots()
    point = [1.3, 0]
    file_id = 1 - 1
    file_name = file_list[file_id]
    file_path = os.path.join(data_dir, file_name)
    print(f"正在处理第{file_id + 1}个文件/ {file_list.__len__()}")
    args = parseUnit.MyParse().args
    d = DataUnit(args, file_path)
    d.refresh_flag = None

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

        global delete_flag,d
        if event.key == 'p':  # 定义按键为 't'
            print("按下了 'p' 键")
            # 下一个文件展示
            get_next_data()
        elif event.key == 'o':  # 定义按键为 't'
            print("按下了 'o' 键")
            # 上一个文件展示
            get_previous_data()
        elif event.key == 'd':  # 定义按键为 't'
            print("按下了 'd' 键")
            # 删除当前文件，展示下一个文件
            os.remove(d.file_path)
            print("删除文件" + d.file_path)
            get_next_data()
        elif event.key == 'r':  # 定义按键为 't'
            print("按下了 'r' 键")
            d.refresh_flag = True

        propose_file()


    def button_press(event):
        """
        鼠标点击事件
        :param event:
        :return:
        """
        if event.button == 1:
            global button_press_flag, update_nearlyPoint_flag
            button_press_flag = 1
            update_nearlyPoint_flag = 1
            point_move_show(event=event)
        elif event.button == 2:
            write_txt(true_spectrum, predict_spectrum, predict_structure)

    def button_release(event):
        """
        鼠标释放事件
        :param event:
        :return:
        """
        global button_press_flag
        button_press_flag = 0

    def motion(event):
        """
        鼠标移动事件
        :param event:
        :return:
        """
        if button_press_flag == 1:
            point_move_show(event=event)

    def point_move_show(**kw):
        """
        刷新图表上的点和曲线
        :param kw:
        :return:
        """
        global point

        if "event" in kw:
            update_point(kw["event"])
            d.refresh_flag = True
        propose_file()

    def update_point(event):
        """
        更新控制点
        :param event:
        :return:
        """
        global point, update_nearlyPoint_flag
        mouse_x = (event.xdata - 0.8) / 0.5
        mouse_y = event.ydata
        point_x = (point[0] - 0.8) / 0.5
        point_y = point[1]
        if event.xdata and event.ydata and 0 <= mouse_x <= 1 and 0 <= mouse_y <= 1:
            distance = math.sqrt(math.pow(mouse_x - point_x, 2) + math.pow(mouse_y - point_y, 2))
            if distance < 1:
                point = [event.xdata, event.ydata]

    def get_previous_data():
        global file_id, d, file_list, pointpoint
        point = [1.3, 0]
        file_id -= 1
        print(f"正在处理第{file_id + 1}个文件/ {file_list.__len__()}")
        file_list = os.listdir(data_dir)
        file_name = file_list[file_id]
        file_path = os.path.join(data_dir, file_name)
        d = DataUnit(args, file_path)
        d.refresh_flag = None
        d.file_path = file_path

    def get_next_data():
        global file_id, d, file_list, point
        point = [1.3, 0]
        file_id += 1
        print(f"正在处理第{file_id + 1}个文件/ {file_list.__len__()}")
        file_list = os.listdir(data_dir)
        file_name = file_list[file_id]
        file_path = os.path.join(data_dir, file_name)
        d = DataUnit(args, file_path)
        d.refresh_flag = None
        d.file_path = file_path

    def propose_file():
        global file_id, d

        if d.refresh_flag == True:
            print("在这里")
            d = DataUnit_2(args, d.file_path, RH=point[1])
            d.refresh_flag = None

            file_name_new = f"d1={d.structure_raw[0]};d2={d.structure_raw[1]};gx={d.structure_raw[2]};gy={d.structure_raw[3]};F={d.F:.4f};I={d.I:.4f};Q={d.Q:.4f};RH={d.RH:.4f};RH={d.W:.4f}.txt"
            file_path_new = os.path.join(data_dir, file_name_new)
            shutil.copy(d.file_path, file_path_new)
            os.remove(d.file_path)
            d.file_path = file_path_new

        print(d.file_path)
        structure_raw = d.structure_raw
        wavelength_nor = d.wavelength_nor
        spectrum_total = d.spectrum_total
        plt.clf()
        plt.xlim(0.8, 1.3)
        plt.ylim(0, 1)
        plt.ylabel('Reflectance', {'family': 'Times New Roman', 'weight': 'normal', 'size': 20})
        plt.xlabel('Wavelength(nm)', {'family': 'Times New Roman', 'weight': 'normal', 'size': 20})
        plt.plot(wavelength_nor, spectrum_total, label="spectram", linewidth=3, )
        plt.scatter(x=point[0], y=point[1], s=100, c="r")
        plt.plot([point[0] - 0.03, point[0] + 0.03], [point[1], point[1]], 'b--')  # 横线
        plt.plot([point[0], point[0]], [point[1] - 0.08, point[1] + 0.08], 'b--')  # 竖线

        if d.F is None or d.I is None or d.Q is None or d.RH is None :
            plt.title("You can't do this", fontsize=24, fontstyle='italic')
        else:
            plt.title(f"d1:{structure_raw[0]:.4f} d2:{structure_raw[1]:.4f} gx:{structure_raw[2]:.4f} gy:{structure_raw[3]:.4f} \n"
                      f"F:{d.F:.4f} I:{d.I:.4f} Q:{d.Q:.4f} RH:{d.RH:.4f} W:{d.W:.4f}", fontsize=14, fontstyle='italic')
            plt.axvline(x=d.F, color='r', linestyle='--')
            plt.axhline(y=d.RH, linestyle='--')
        plt.legend()
        plt.show(block=True)

    fig.canvas.mpl_connect('button_press_event', button_press)
    fig.canvas.mpl_connect('button_release_event', button_release)
    fig.canvas.mpl_connect('motion_notify_event', motion)
    fig.canvas.mpl_connect('key_press_event', key_press)

    point_move_show()
    plt.show()


if __name__ == '__main__':
    execute()
