import re
import pandas as pd
from hexbytes import HexBytes
from datasets import load_dataset
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from torch.utils.data import DataLoader
import torch
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

# Due to a bug in the HuggingFace dataset, at the moment two file checksums do not correspond to what
# is in the dataset metadata, thus we have to load the data splits with the flag ignore_verification
# set to true
train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train',
                         verification_mode='no_checks')
test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test',
                        verification_mode='no_checks')
val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation',
                       verification_mode='no_checks')

print(f"训练集:{train_set}，测试集:{test_set},验证集:{val_set}")
print(f"训练集第一个数据'{train_set[0]}")
print(f"训练集的type'{train_set.features}")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["source_code"], example["bytecode"], padding=True, truncation=True)


tokenized_train_set = train_set.map(tokenize_function, batched=True)
tokenized_test_set = test_set.map(tokenize_function, batched=True)
tokenized_val_set = val_set.map(tokenize_function, batched=True)

print(tokenized_train_set, tokenized_test_set, tokenized_val_set)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_train_set = tokenized_train_set.remove_columns(["address", "source_code", "bytecode"])
tokenized_test_set = tokenized_test_set.remove_columns(["address", "source_code", "bytecode"])
tokenized_val_set = tokenized_val_set.remove_columns(["address", "source_code", "bytecode"])
tokenized_train_set = tokenized_train_set.rename_column("slither", "labels")
tokenized_test_set = tokenized_test_set.rename_column("slither", "labels")
tokenized_val_set = tokenized_val_set.rename_column("slither", "labels")
tokenized_train_set.set_format("torch")
tokenized_test_set.set_format("torch")
tokenized_val_set.set_format("torch")
tokenized_train_set.column_names



train_dataloader = DataLoader(
    tokenized_train_set, shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_val_set, batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break



model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)



optimizer = AdamW(model.parameters(), lr=5e-5)



num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)


device = torch.device("cuda")
model.to(device)
print(device)



progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

import evaluate

metric = evaluate.load("mwritescode/slither-audited-smart-contracts", 'big-multilabel',
                       verification_mode='no_checks')
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
