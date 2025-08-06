import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from auxiliary_tools import parseUnit
from auxiliary_tools.dataSet_inverse import DataSet
from bidirectional_forward_design_network.forward_design_network import ForwardNetMlp as F_Net
from inverse_design_network import InverseNet as I_Net
torch.cuda.empty_cache()

title = "bidirectional-inverse-design-network-test"
checkpoint_name = title + "-checkpoint.pth.tar"
event_out_dir = title + "-event.out"

class MainInverseNet(object):

    def __init__(self, args):
        self.args = args
        self.args.seed = 190730
        self.set_seed()
        # 准备数据,创建数据加载器
        # train_set = DataSet(self.args, "train")
        # val_set = DataSet(self.args, "val")
        dataset = torch.load(r"..\data_space\datasets_tar\unloaded_dataset_inverse_for_technology.pth")
        dataset_train = dataset['dataset_train']
        dataset_val = dataset['dataset_val']

        # 创建数据加载器
        self.train_loader = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(dataset=dataset_val, batch_size=64, shuffle=True)

        # 创建模型、损失函数、优化器等,初始化模型参数
        self.f_model = F_Net().to(self.args.device)
        self.i_model = I_Net().to(self.args.device)
        self.paras_initialize()
        self.criterion = self.my_mse_loss
        self.args.lr = 0.0001
        self.optimizer = optim.Adam(self.i_model.parameters(), lr=self.args.lr)

    def my_mse_loss(self, predict_output, expected_output):
        error_squared_sum = torch.sum((predict_output - expected_output) ** 2)
        num_zeros = torch.sum(torch.eq(expected_output, 0)).item()
        mse_loss = error_squared_sum / (expected_output.size(0) * expected_output.size(1) - num_zeros)
        return mse_loss

    def train(self):
        # 获取模型、损失函数、优化器
        i_model = self.i_model
        f_model = self.f_model
        criterion = self.criterion
        optimizer = self.optimizer
        f_checkpoint = torch.load(
            r"../data_space/checkpoints/bidirectional-forward-design-network-checkpoint.pth.tar")
        f_model.load_state_dict(f_checkpoint['state_dict'])

        # 是否读取读取checkpoint
        if args.load:
            if os.path.exists(args.checkpoint_path):
                checkpoint = self.load_checkpoint()
                i_model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_loss_list = checkpoint['train_loss_list']
                val_loss_list = checkpoint['val_loss_list']
                epoch = checkpoint['epoch']
            else:
                train_loss_list = []
                val_loss_list = []
                epoch = 0
                state_info = {
                    "epoch": epoch,
                    "train_loss_list": train_loss_list,
                    "val_loss_list": val_loss_list,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": i_model.state_dict(),
                }
                self.save_checkpoint(state_info)
        else:
            train_loss_list = []
            val_loss_list = []
            start_epoch = 1
            state_info = {
                "epoch": start_epoch - 1,
                "train_loss_list": train_loss_list,
                "val_loss_list": val_loss_list,
                "optimizer": optimizer.state_dict(),
                "state_dict": i_model.state_dict(),
            }
            self.save_checkpoint(state_info)
        print(f"Epoch [{epoch}/{self.args.epochs}] "
              f"- Lr: {optimizer.param_groups[0]['lr']:.4f} "
              f"- train_loss: {train_loss_list} "
              f"- val_loss: {val_loss_list}")

        train_load_num = len(self.train_loader)
        self.args.epochs = 820
        for epoch in range(epoch + 1, self.args.epochs + 1):

            i_model.train()
            running_loss = 0.0
            self.adjust_learning_rate(optimizer, epoch)

            for index, (input_vs, valley_pos, real_structure, mask,
                        reverse_mask, structure_raw, spectrum_total) in enumerate(self.train_loader):

                input_vs = input_vs.to(self.args.device)
                valley_pos = valley_pos.to(self.args.device)
                real_structure = real_structure.to(self.args.device)
                mask = mask.to(self.args.device)
                reverse_mask = reverse_mask.to(self.args.device)

                # 将梯度清零
                optimizer.zero_grad()
                # 逆向传播
                structure_out = i_model(input_vs)
                # 中途处理
                structure_1 = structure_out.masked_fill(mask == 0, 0)
                structure_2 = real_structure.masked_fill(mask == 0, 0)
                loss_structure = criterion(structure_1, structure_2)
                structure_pre = structure_out.masked_fill(reverse_mask == 0, 0)
                structure_in = structure_2 + structure_pre
                # 正向传播
                valley_out = self.f_model(structure_in)
                loss_valley = criterion(valley_pos, valley_out)
                # 计算损失
                loss = loss_valley
                # 反向传播
                loss.backward()
                # 参数优化
                optimizer.step()
                # 累计损失和训练次数
                running_loss += loss.item()
                progress = (index + 1) / train_load_num * 100
                sys.stdout.write(f'\rCurrent_Epoch_Train_Progress: {progress:.2f}%')
                sys.stdout.flush()

            train_loss = running_loss / train_load_num
            train_loss_list.append(train_loss)
            val_loss = self.validation(i_model)
            val_loss_list.append(val_loss)

            if args.debug == False:
                self.args.writer.add_scalar('train_loss', train_loss, epoch)
                self.args.writer.add_scalar('val_loss', val_loss, epoch)
            state_info = {
                "epoch": epoch,
                "train_loss_list": train_loss_list,
                "val_loss_list": val_loss_list,
                "optimizer": optimizer.state_dict(),
                "state_dict": i_model.state_dict(),
            }
            self.save_checkpoint(state_info)
            print(f"Epoch [{epoch}/{self.args.epochs}] "
                  f"- Lr: {optimizer.param_groups[0]['lr']:.8f} "
                  f"- train_loss: {train_loss:.8f} "
                  f"- val_loss: {val_loss:.8f}")
        print(f"Epoch [{epoch}/{self.args.epochs}] "
              f"- Lr: {optimizer.param_groups[0]['lr']:.8f} "
              f"- train_loss: {train_loss_list[-1]:.8f} "
              f"- val_loss: {val_loss_list[-1]:.8f}")

    def load_checkpoint(self):
        print("loading checkpoint...")
        # 如果不存在会报错
        checkpoint = torch.load(args.checkpoint_path)
        return checkpoint

    def save_checkpoint(self, state):
        torch.save(state, args.checkpoint_path)

    def cal_accuracy(self, true, out):
        accuracy = torch.sum(torch.mean(torch.pow(true - out, 2), dim=1))
        return accuracy

    def validation(self, i_model):
        i_model.eval()
        with torch.no_grad():
            running_loss = 0
            val_load_num = len(self.val_loader)
            for index, (input_vs, valley_pos, real_structure, mask,
                        reverse_mask, structure_raw, spectrum_total) in enumerate(self.val_loader):
                input_vs = input_vs.to(self.args.device)
                valley_pos = valley_pos.to(self.args.device)
                real_structure = real_structure.to(self.args.device)
                mask = mask.to(self.args.device)
                reverse_mask = reverse_mask.to(self.args.device)

                # 逆向传播
                structure_out = i_model(input_vs)
                # 中途处理
                structure_1 = structure_out.masked_fill(mask == 0, 0)
                structure_2 = real_structure.masked_fill(mask == 0, 0)
                loss_structure = self.criterion(structure_1, structure_2)
                structure_pre = structure_out.masked_fill(reverse_mask == 0, 0)
                structure_in = structure_2 + structure_pre
                # 正向传播
                valley_out = self.f_model(structure_in)
                loss_valley = self.criterion(valley_pos, valley_out)
                # 计算损失
                loss = loss_valley
                running_loss += loss

                progress = (index + 1) / val_load_num * 100
                sys.stdout.write(f'\rCurrent_Epoch_Val_Progress: {progress:.2f}%')
                sys.stdout.flush()

            sys.stdout.write(f'\r')
            val_loss = running_loss / val_load_num
            return val_loss

    def set_seed(self):
        seed = self.args.seed
        # 设置 Python 随机数种子
        random.seed(seed)
        # 设置 NumPy 随机数种子
        np.random.seed(seed)
        # 设置 PyTorch 随机数种子
        torch.manual_seed(seed)
        # 在 CPU 上使用固定的随机数种子
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 在 CUDA 上使用固定的随机数种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def paras_initialize(self):
        for m in self.i_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.args.lr * np.power(0.8, epoch // 40)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    args = parseUnit.MyParse(debug=False, load=True, save=True).args
    args.checkpoint_path = "../data_space/checkpoints/" + checkpoint_name
    args.event_out_dir = "../data_space/events_out/" + event_out_dir
    if not args.debug:
        args.writer = SummaryWriter(args.event_out_dir)

    Main_Net = MainInverseNet(args)
    Main_Net.train()
