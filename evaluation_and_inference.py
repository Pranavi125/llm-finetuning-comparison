from torch.utils.data import Dataset, DataLoader
import torch

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE  = 32

class SentimentData(Dataset):
    def __init__(self, data, inputs_tokenized):
        """
        data: pandas DataFrame with columns 'text' and 'label'
        inputs_tokenized: either
          - a HuggingFace BatchEncoding (dict-like with keys 'input_ids', 'attention_mask', ...)
          - OR a list of dicts where each dict has keys 'input_ids', 'attention_mask', (optionally) 'token_type_ids'
          - OR a BatchEncoding with return_tensors='pt' (values are tensors)
        """
        self.texts = data['text'].fillna("").astype(str).tolist()
        self.targets = data['label'].astype(int).tolist()

        # Detect dict-like / BatchEncoding vs list-of-dicts
        # BatchEncoding responds to hasattr(..., 'keys') and contains 'input_ids'
        if (hasattr(inputs_tokenized, 'keys') or hasattr(inputs_tokenized, 'data')) and 'input_ids' in inputs_tokenized:
            self.inputs_is_dict = True
            self.inputs = inputs_tokenized
        else:
            # assume list of dicts
            self.inputs_is_dict = False
            self.inputs = list(inputs_tokenized)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # normalize text
        text = str(self.texts[index])
        text = " ".join(text.split())

        # get tokenized pieces robustly
        if self.inputs_is_dict:
            # Values may be lists of lists or tensors (if return_tensors='pt')
            ids = self.inputs['input_ids'][index]
            mask = self.inputs['attention_mask'][index]

            # token_type_ids might not exist (RoBERTa)
            if 'token_type_ids' in self.inputs:
                ttype = self.inputs['token_type_ids'][index]
            else:
                # create a zero vector with the same length as ids
                # handle tensor / list cases
                if torch.is_tensor(ids):
                    ttype = torch.zeros(ids.shape, dtype=torch.long)
                else:
                    ttype = [0] * (len(ids) if hasattr(ids, '__len__') else 0)
        else:
            # inputs is list of dicts
            item = self.inputs[index]
            ids = item.get('input_ids')
            mask = item.get('attention_mask')
            ttype = item.get('token_type_ids', None)
            if ttype is None:
                if torch.is_tensor(ids):
                    ttype = torch.zeros(ids.shape, dtype=torch.long)
                else:
                    ttype = [0] * (len(ids) if hasattr(ids, '__len__') else 0)

        # convert to tensors if not already
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.long)
        if not torch.is_tensor(ttype):
            ttype = torch.tensor(ttype, dtype=torch.long)

        target = torch.tensor(self.targets[index], dtype=torch.long)

        return {
            'sentence': text,
            'ids': ids,
            'mask': mask,
            'token_type_ids': ttype,
            'targets': target
        }

# -------------------------
# Recreate datasets / loaders
# If you used tokenizer(list(...)) earlier, pass train_encodings/test_encodings (BatchEncoding).
# If you used a list of dicts (old encode_plus loop), pass those lists.
train_dataset = SentimentData(train_data, train_encodings)   # or train_tokenized_data
test_dataset  = SentimentData(test_data, test_encodings)     # or test_tokenized_data

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=TEST_BATCH_SIZE, shuffle=False)

print("Train batches:", len(train_loader), "Test batches:", len(test_loader))

# -------------------------------

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True
                }

train_loader = DataLoader(train_dataset, **train_params)
test_loader = DataLoader(test_dataset, **test_params)

# -------------------------------

from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

train_loss = []
test_loss = []

train_accuracy = []
test_accuracy = []

test_answers = [[[],[]], [[],[]]]

def train_loop(epochs):
  for epoch in range(epochs):
    for phase in ['Train', 'Test']:
      if(phase == 'Train'):
        model.train()
        loader = train_loader
      else:
        model.eval()
        loader = test_loader
      epoch_loss = 0
      epoch_acc = 0
      for steps, data in tqdm(enumerate(loader, 0)):
        sentence = data['sentence']
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model.forward(ids, mask, token_type_ids)

        loss = loss_function(outputs, targets)

        epoch_loss += loss.detach()
        _, max_indices = torch.max(outputs.data, dim=1)
        bath_acc = (max_indices==targets).sum().item()/targets.size(0)
        epoch_acc += bath_acc

        if (phase == 'Train'):
          train_loss.append(loss.detach())
          train_accuracy.append(bath_acc)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        else:
          test_loss.append(loss.detach())
          test_accuracy.append(bath_acc)
          if epoch == epochs-1:
            for i in range(len(targets)):
              test_answers[targets[i].item()][max_indices[i].item()].append([sentence[i],
                                                                 targets[i].item(),
                                                                 max_indices[i].item()])

      print(f"{phase} Loss: {epoch_loss/steps}")
      print(f"{phase} Accuracy: {epoch_acc/steps}")

# -------------------------------

LEARNING_RATE = 1e-05

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

EPOCHS = 4
train_loop(EPOCHS)

# -------------------------------

# Convert to CPU and NumPy or list
train_loss = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in train_loss]
test_loss = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in test_loss]
train_accuracy = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in train_accuracy]
test_accuracy = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in test_accuracy]

# -------------------------------

import matplotlib.pyplot as plt

plt.plot(train_loss, color='blue')
plt.title("Train Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.show()

plt.plot(test_loss, color='orange')
plt.title("Test Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.show()

plt.plot(train_accuracy, color='blue')
plt.title("Train Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.show()

plt.plot(test_accuracy, color='orange')
plt.title("Test Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.show()

# -------------------------------

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

len_num = len(test_dataset)

tp=len(test_answers[1][1])/len_num
fn=len(test_answers[1][0])/len_num
fp=len(test_answers[0][1])/len_num
tn=len(test_answers[0][0])/len_num

array_matrix = [[tp,tn],
                [fp,fn]]
df_cm = pd.DataFrame(array_matrix, index = ['T', 'F'],
                  columns = ['P', 'N'])
plt.figure(figsize = (5,5))
sn.heatmap(df_cm, annot=True)

# -------------------------------

print('False Negative:\n', test_answers[0][0][:3],
      'False Positive:\n', test_answers[0][1][:3])

# -------------------------------

save_path="./"
torch.save(model, save_path+'trained_roberta.pt')
print('All files saved')
