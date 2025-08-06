from spectrum_unloaded_prediction_network.spectrum_prediction_network import ForwardNetMlp as Net


if __name__ == "__main__":
    print("welcome Teddy")
    # 创建模型实例
    model = Net()

    # 计算模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")