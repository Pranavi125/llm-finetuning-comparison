"""
ZERO-SHOT LEARNING
Author: Mini Project
Description:
This script demonstrates ZERO-SHOT learning using a pretrained FLAN-T5 model.
No examples and no training data are provided. The model relies purely on its
pretrained knowledge to perform sentiment classification.
"""

# ==========================
# STEP 1: INSTALL LIBRARIES
# ==========================
# pip install transformers torch

# ==========================
# STEP 2: IMPORTS
# ==========================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================
# STEP 3: LOAD MODEL
# ==========================
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ==========================
# STEP 4: ZERO-SHOT PROMPT
# ==========================
prompt = (
    "Classify the sentiment of the following sentence as "
    "positive, negative, or neutral:\n"
    "Sentence: I absolutely loved the movie and the acting."
)

# ==========================
# STEP 5: TOKENIZATION
# ==========================
inputs = tokenizer(prompt, return_tensors="pt")

# ==========================
# STEP 6: MODEL INFERENCE
# ==========================
outputs = model.generate(
    **inputs,
    max_length=20,
    num_beams=4,
    early_stopping=True
)

# ==========================
# STEP 7: OUTPUT
# ==========================
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("ZERO-SHOT PREDICTION:", prediction)
