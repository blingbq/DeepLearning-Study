import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



def binary_cross_entropy(y_pred,y_true):
    """
    计算二元交叉熵损失函数
    `y_pred`: 模型的原始预测结果（logits，尚未经过sigmoid）
    `y_true`: 真实标签，通常是0或1
    """
    y_pred = 1/ ( 1 + torch.exp(-y_pred) )

    # 为了避免 `log(0)` 的数值不稳定问题
    # 这里将 `y_pred` 限制在 \[1e-7, 1 - 1e-7\] 的范围内，保证对数运算的安全。
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred,min = epsilon,max=1.0 - epsilon)

    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    return loss.mean()

def sigmoid(x):
    """
    Sigmoid 激活函数
    `x`: 输入张量
    """
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(x):
    """
    Sigmoid 函数的导数
    `x`: 输入张量
    """
    sig = sigmoid(x)
    return sig * (1 - sig)


