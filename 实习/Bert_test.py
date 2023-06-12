import torch
from datasets import load_dataset
from transformers import BertTokenizer
from transformers import AdamW

print(torch.__version__)

# 使用cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果是cuda则调用cuda反之则cpu
print('device=', device)

# 从transformers调用现有的model
from transformers import BertModel

# 加载预训练模型，使用预训练模型可以加快训练的速度
pretrained = BertModel.from_pretrained('bert-base-uncased')
# 需要移动到cuda上
pretrained.to(device)

# pretrained 模型中所有参数的 requires_grad 属性设置为 False，这意味着这些参数在训练过程中将不会被更新，其值将保持不变
for param in pretrained.parameters():
    param.requires_grad_(False)
#fine tune向后传播而不修改之前的参数

# 定义了下游任务模型，包括一个全连接层和forward方法。
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 定义一个全连接层，输入维度为768，输出维度为6
        self.fc = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 将输入传入预训练模型，并记录计算图以计算梯度
        out = pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        # 只保留预训练模型输出的最后一个隐藏状态，并通过全连接层进行分类
        out = self.fc(out.last_hidden_state[:, 0])
        # 对输出进行 softmax 归一化，以得到概率分布
        out = out.softmax(dim=1)
        return out


# 实例化下游任务模型并将其移动到 GPU 上 (如果可用)
model = Model()
model.to(device)

#虚拟一批数据,需要把所有的数据都移动到cuda上
input_ids = torch.ones(16, 500).long().to(device)
attention_mask = torch.ones(16, 500).long().to(device)
token_type_ids = torch.ones(16, 500).long().to(device)
labels = torch.ones(16).long().to(device)

#试算
model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split=split,
                                    verification_mode='no_checks')
        # 从huggingface导入数据集

    def __len__(self):
        return len(self.dataset)
        # 计算数据集长度，方便后面进行一个批量的操作

    def __getitem__(self, i):
        source_code = self.dataset[i]['source_code']
        bytecode = self.dataset[i]['bytecode']
        label = self.dataset[i]['slither']
        # 从数据集遍历数据，比如第一批是16个，那么第二批就可以从17-32
        return source_code, bytecode, label


dataset = Dataset('train')
# 只截取训练集
len(dataset), dataset[0]

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-uncased')


def collate_fn(data):
    source_codes = [i[0] for i in data]
    bytecodes = [i[1] for i in data]
    labels = [i[2][0] for i in data]

    # 编码
    data = token.batch_encode_plus(
        source_codes,
        bytecodes,
        padding='max_length',
        truncation=True,
        max_length=500,
        return_tensors='pt', #返回pytorch模型
        return_length=True)

    # 对于 `data` 字典中的每个键值对：
    # `input_ids`: 编码后的数字表示。
    # `attention_mask`: 表示哪些位置是有效的，哪些位置是补零的（0/1）。
    # `token_type_ids`: BERT 能够区分两个句子的方法（第一句话000..第二句话111...）。
    # `length`: 编码后序列的长度

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    # print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,  # 是每个批次的大小，也就是每次处理的样本数量。
                                     collate_fn=collate_fn,  # 是一个函数，用于对每个批次中的样本进行编码和处理。
                                     shuffle=True,  # 是一个布尔值，表示是否对数据进行随机重排。
                                     drop_last=True)  # 是一个布尔值，表示是否在最后一个批次中舍弃不足一个批次大小的数据

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break
    # 用enumerate函数遍历loader的数据，将它们的值赋给input_ids, attention_mask, token_type_ids,labels

input_ids.shape, attention_mask.shape, token_type_ids.shape, labels



# 训练
optimizer = AdamW(model.parameters(), lr=1e-5)   # 定义优化器，AdamW是一种优化算法，model.parameters()返回模型中所有参数的迭代器
criterion = torch.nn.CrossEntropyLoss()   # 定义损失函数，交叉熵损失用于多分类任务

model.train()   # 将模型设置为训练模式
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    # 遍历数据集，并将数据转移到GPU上
    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # 进行前向传播，得到预测值out
    loss = criterion(out, labels)   # 计算损失
    loss.backward()   # 反向传播，计算梯度
    optimizer.step()   # 更新参数
    optimizer.zero_grad()   # 梯度清零，防止梯度累积

    if i % 5 == 0:   # 每训练5个batch，打印一次训练情况
        out = out.argmax(dim=1)   # 将预测值转化为标签
        accuracy = (out == labels).sum().item() / len(labels)   # 计算准确率

        print(f"第{i}次训练, loss：{loss.item()}, accuracy:{accuracy}")

    if i == 500:   # 训练500个batch后停止
        break

# 保存模型
model_save_path = "../Nlp/model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到：{model_save_path}")
