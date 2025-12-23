import pandas as pd

full_data = pd.read_csv(
    '/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',
    encoding='latin-1'
).drop(
    ["1467810369", "Mon Apr 06 22:19:45 PDT 2009", "NO_QUERY", "_TheSpecialOne_"],
    axis=1
).dropna()

columns_names = list(full_data)
full_data.rename(columns={columns_names[0]: "label", columns_names[1]: "text"}, inplace=True)

NUM_SAMPLES = 30000
negative_samples = full_data[full_data["label"] == 0].sample(NUM_SAMPLES, random_state=42)
positive_samples = full_data[full_data["label"] == 4].sample(NUM_SAMPLES, random_state=42)

positive_samples["label"] = 1  # relabel positives as 1

# combine and shuffle
full_data = pd.concat([negative_samples, positive_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(full_data, test_size=0.3)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
positiv_samples["label"]=[1]*NUM_SAMPLES

full_data = pd.concat([negative_samples,  positiv_samples])

# -------------------------------

print(train_data.columns)

# -------------------------------

# prerequisites
import torch
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# 1) tokenizer init
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

MAX_LEN = 130
batch_size = 32

# 2) Make sure columns are 'label' and 'text' and handle NaNs
# If your labels are 0/4 (Sentiment140), map 4->1 to get binary 0/1
train_data['label'] = train_data['label'].replace({4: 1}).astype(int)
test_data['label']  = test_data['label'].replace({4: 1}).astype(int)

# Replace missing text with empty string and ensure strings
train_data['text'] = train_data['text'].fillna("").astype(str)
test_data['text']  = test_data['text'].fillna("").astype(str)

print("Train empty texts:", (train_data['text'] == "").sum())
print("Test  empty texts:", (test_data['text'] == "").sum())

# 3) Batch tokenization
train_encodings = tokenizer(
    list(train_data['text']),
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_token_type_ids=False
)

test_encodings = tokenizer(
    list(test_data['text']),
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_token_type_ids=False
)

# 4) Convert to PyTorch tensors (labels come from 'label' column)
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_data['label'].values, dtype=torch.long)

test_input_ids = torch.tensor(test_encodings['input_ids'])
test_attention_mask = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_data['label'].values, dtype=torch.long)

# 5) Create TensorDataset + DataLoader
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset  = TensorDataset(test_input_ids, test_attention_mask, test_labels)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
test_dataloader  = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

print("Prepared dataloaders: train batches =", len(train_dataloader), "test batches =", len(test_dataloader))
