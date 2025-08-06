import torch
import argparse
from torch.utils.tensorboard import SummaryWriter


class MyParse(object):
    def __init__(self, debug=True, load=False, save=False):
        # 创建argument解析器
        self.__parser = argparse.ArgumentParser('structure_parser')
        # 向解析器里面添加argument
        self.__parser.add_argument('--root_dir', default=r"D:\DL\AI_for_THz_Code",
                                   help="the root dir of project")
        self.__parser.add_argument('--epochs', default=820, type=int, help="total epoch needed to run")
        self.__parser.add_argument('--epoch', default=0, type=int, help="Current training epoch")
        self.__parser.add_argument('--structure_range',
                                   default=[(0, 40), (0, 40), (0, 40), (0, 40)],
                                   type=list, help="The range of values for each parameter")
        self.__parser.add_argument('--batch_size', default=32, type=int, help="total epoch needed to run")
        self.__parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate of Model")
        self.__parser.add_argument('--seed', default=485316, type=int, help="Random number seeds for each module")
        self.__parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                   type=str, help="available device")
        # 调用解析器的parse_args生成参数器
        self.args = self.__parser.parse_args()
        # 创建管理参数
        self.args.debug = debug
        self.args.load = load
        self.args.save = save


if __name__ == "__main__":
    args = MyParse(debug=False).args
    # 往train_acc事件添加点数据
    if not args.debug:
        args.event_out_dir = "./events.out/"
        args.writer = SummaryWriter(args.event_out_dir)
        args.writer.add_scalar('train_acc', 0.02, 1)
        args.writer.add_scalar('train_acc', 0.01, 2)
