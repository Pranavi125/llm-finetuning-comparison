"""
FEW-SHOT LEARNING
Author: Mini Project
Description:
This script demonstrates FEW-SHOT learning using in-context examples.
The model is not trained; instead, examples are provided inside the prompt
to guide the model toward the correct pattern.
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
# STEP 4: FEW-SHOT PROMPT
# ==========================
prompt = """
Classify the sentiment of each sentence.

Sentence: I love this product.
Sentiment: positive

Sentence: This is the worst experience of my life.
Sentiment: negative

Sentence: The service was okay, nothing special.
Sentiment: neutral

Sentence: I am extremely happy with the support team.
Sentiment:
"""

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
print("FEW-SHOT PREDICTION:", prediction)
