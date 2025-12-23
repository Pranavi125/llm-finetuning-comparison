# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
kazanova_sentiment140_path = kagglehub.dataset_download('kazanova/sentiment140')

print('Data source import complete.')

# -------------------------------

!pip install transformers


# ===== MODEL & DATASET SETUP =====

import torch
from transformers import RobertaModel

# Step 1: Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Define the custom Roberta model
class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.classifier = torch.nn.Linear(768, 2)
        self.dropout = torch.nn.Dropout(0.3)  # Optional, helps generalization

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # RobertaModel does not use token_type_ids (unlike BERT)
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]  # last hidden state
        pooler = hidden_state[:, 0]  # take [CLS] token representation
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Step 3: Initialize and move model to device
model = RobertaClass()
model.to(device)

# -------------------------------

import pandas as pd
full_data = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',
                        encoding='latin-1').drop(["1467810369",
                                                  "Mon Apr 06 22:19:45 PDT 2009",
                                                  "NO_QUERY","_TheSpecialOne_"],
                                                 axis=1).dropna()
columns_names = list(full_data)
full_data.rename(columns={columns_names[0]:"label",
                        columns_names[1]:"text"}, inplace= True)

# -------------------------------

NUM_SAMPLES = 30000
negative_samples = full_data[full_data["label"]==0][:NUM_SAMPLES]
positiv_samples = full_data[full_data["label"]==4][:NUM_SAMPLES]

positiv_samples["label"]=[1]*NUM_SAMPLES

full_data = pd.concat([negative_samples,  positiv_samples])
