import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, AdamW

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pretrained = BertModel.from_pretrained('bert-base-uncased')
pretrained.to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = torch.sigmoid(out)  # 添加Sigmoid激活函数
        return out

model = Model()
model.to(device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split=split, verification_mode='no_checks')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        source_code = self.dataset[i]['source_code']
        bytecode = self.dataset[i]['bytecode']
        label = self.dataset[i]['slither']
        return source_code, bytecode, label


train_dataset = Dataset('train')
test_dataset = Dataset('test')
val_dataset = Dataset('validation')

def collate_fn(data):
    source_codes = [i[0] for i in data]
    bytecodes = [i[1] for i in data]
    labels = [i[2] for i in data]
    multi_labels = torch.zeros(len(labels), 6)
    for idx, label_set in enumerate(labels):
        for label in label_set:
            multi_labels[idx][label] = 1

    labels = multi_labels.to(device)

    data = tokenizer.batch_encode_plus(
        source_codes,
        bytecodes,
        padding='max_length',
        truncation=True,
        max_length=500,
        return_tensors='pt',
        return_length=True)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.tensor(labels, dtype=torch.float).to(device)

    return input_ids, attention_mask, token_type_ids, labels


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=16,
                                           collate_fn=collate_fn,
                                           shuffle=True,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=16,
                                          collate_fn=collate_fn,
                                          shuffle=True,
                                          drop_last=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=16,
                                         collate_fn=collate_fn,
                                         shuffle=False,
                                         drop_last=False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in val_loader:
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(out, labels)

            val_loss += loss.item() * len(labels)
            total_samples += len(labels)

            preds = (out > 0.5).float()
            val_corrects += (preds == labels).sum().item()

    val_loss /= total_samples
    val_accuracy = val_corrects / (total_samples * 6)  # 6是类别数量

    return val_loss, val_accuracy

# Training and evaluation
num_epochs = 5

optimizer = AdamW(model.parameters(), lr=1e-5)  # 定义优化器，AdamW是一种优化算法，model.parameters()返回模型中所有参数的迭代器
criterion = torch.nn.BCELoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(labels)
        total_samples += len(labels)

        preds = (out > 0.5).float()
        running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_corrects / (total_samples * 6)

    val_loss, val_accuracy = evaluate(model, val_loader, criterion)

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {epoch_loss:.4f} Train Accuracy: {epoch_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_accuracy:.4f}')

# 测试
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f}')

# 保存模型
model_save_path = "model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到：{model_save_path}")