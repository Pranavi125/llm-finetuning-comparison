"""
FINE-TUNING
Author: Mini Project
Description:
This script demonstrates FINE-TUNING of a pretrained FLAN-T5 model.
The model learns from labeled data and updates its weights to improve accuracy.
This approach provides the best performance among the three methods.
"""

# ==========================
# STEP 1: INSTALL LIBRARIES
# ==========================
# pip install transformers datasets torch accelerate pandas

# ==========================
# STEP 2: IMPORTS
# ==========================
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

# ==========================
# STEP 3: LOAD MODEL
# ==========================
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ==========================
# STEP 4: PREPARE DATASET
# ==========================
data = {
    "input_text": [
        "Classify sentiment: I love this phone",
        "Classify sentiment: This application is terrible",
        "Classify sentiment: The movie was average",
        "Classify sentiment: I am very happy with the service",
        "Classify sentiment: I hate waiting so long"
    ],
    "target_text": [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative"
    ]
}

dataset = Dataset.from_pandas(pd.DataFrame(data))

# ==========================
# STEP 5: TOKENIZATION FUNCTION
# ==========================
def preprocess(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )
    labels = tokenizer(
        batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=10
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# ==========================
# STEP 6: TRAINING SETUP
# ==========================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=1,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# ==========================
# STEP 7: TRAIN MODEL
# ==========================
trainer.train()

# ==========================
# STEP 8: TEST MODEL
# ==========================
test_sentence = "Classify sentiment: I am unhappy with the product quality"
inputs = tokenizer(test_sentence, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=20)

prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("FINE-TUNED PREDICTION:", prediction)
