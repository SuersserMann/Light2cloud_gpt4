import re
import pandas as pd
import seaborn as sns
from hexbytes import HexBytes
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Due to a bug in the HuggingFace dataset, at the moment two file checksums do not correspond to what
# is in the dataset metadata, thus we have to load the data splits with the flag ignore_verification
# set to true
train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train',
                         verification_mode='no_checks')
test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test',
                        verification_mode='no_checks')
val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation',
                       verification_mode='no_checks')


def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        # if the 2nd group is not None, then we have captured a real comment string.
        if match.group(2) is not None:
            return ""
        else:  # otherwise, we will return the 1st group
            return match.group(1)

    return regex.sub(_replacer, string)


def get_lenghts(example):
    code = remove_comments(example['source_code'])
    example['sourcecode_len'] = len(code.split())
    example['bytecode_len'] = len(HexBytes(example['bytecode']))
    return example


COLS_TO_REMOVE = ['source_code', 'bytecode']

LABELS = {0: 'access-control', 1: 'arithmetic', 2: 'other', 3: 'reentrancy', 4: 'safe', 5: 'unchecked-calls'}

datasets = []
for split in [train_set, test_set, val_set]:
    split_df = pd.DataFrame(split.map(get_lenghts, remove_columns=COLS_TO_REMOVE)).explode('slither')
    split_df['slither'] = split_df['slither'].map(LABELS)
    datasets.append(split_df)

concatenated = pd.concat(
    [split.assign(dataset=split_name) for split, split_name in zip(datasets, ['train', 'test', 'val'])])

print(concatenated.head())



