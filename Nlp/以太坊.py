import json
import torch
import torch.nn as nn
from datasets import load_dataset

train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train',
                         verification_mode='no_checks')
data = []
# 只保留slither第一个标签
for i in range(len(train_set)):
    byte_info = dict()
    byte_info['bytecode'] = train_set[i]['bytecode']
    byte_info['slither'] = train_set[i]['slither'][0]
    data.append(byte_info)
    if i == 5:
        break
bytecodes = [d['bytecode'] for d in data]
slithers = [d['slither'] for d in data]

from ethereum import utils


opcodes = []

for bytecode in bytecodes:
    bytecode = bytecode[2:]  # remove '0x' prefix
    decoded = utils.decode_hex(bytecode)
    opcodes.extend(utils.parse_assembly(decoded))

for opcode in opcodes:
    print(opcode)