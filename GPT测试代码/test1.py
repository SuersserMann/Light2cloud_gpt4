import torch
import torch.nn.functional as F

# 假设标签为整数列表
labels = [1, 3, 5]

# 将标签转换为 one-hot 编码的形式
num_classes = 6  # 标签的类别数
labels = torch.LongTensor(labels)  # 将列表转换为 LongTensor 类型
labels = F.one_hot(labels, num_classes=num_classes)  # 将标签转换为 one-hot 编码的形式
print(labels)