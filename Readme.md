# Comparative Study of Zero-Shot, Few-Shot, and PEFT Fine-Tuning
### Using RoBERTa and FLAN-T5 with Explainability

---

## 📌 Overview
This project implements and compares **Zero-Shot**, **Few-Shot**, and **Fine-Tuning** approaches for NLP tasks using **FLAN-T5** and **RoBERTa** models.  
The study evaluates performance improvements through **Parameter-Efficient Fine-Tuning (PEFT)** and provides **model explainability using LIME** across all approaches.

The goal is to analyze accuracy, efficiency, and interpretability of modern transformer-based models under different learning paradigms.

---

## 🧠 Models Used
- **FLAN-T5**  
  - Zero-Shot Learning  
  - Few-Shot Learning  
- **RoBERTa**
  - Fine-Tuned using **PEFT**

---

## 🎯 Learning Approaches Implemented
- **Zero-Shot Learning**
- **Few-Shot Learning**
- **Parameter-Efficient Fine-Tuning (PEFT)**

---

## 🔍 Explainability
- **LIME (Local Interpretable Model-Agnostic Explanations)**  
  - Applied to Zero-Shot, Few-Shot, and Fine-Tuned models  
  - Helps interpret model predictions and feature importance

---

## 📊 Results & Performance

| Approach        | Model     | Accuracy (%) |
|-----------------|-----------|--------------|
| Zero-Shot       | FLAN-T5   | **81.2**     |
| Few-Shot        | FLAN-T5   | **81.4**     |
| Fine-Tuning     | RoBERTa + PEFT | **86.9** |

🔹 Fine-tuning with PEFT shows a **significant improvement** over zero-shot and few-shot learning.

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Models & Libraries:**  
  - Hugging Face Transformers  
  - RoBERTa  
  - FLAN-T5  
  - PEFT  
  - LIME  
- **ML Tools:**  
  - PyTorch  
  - NumPy, Pandas  
  - Scikit-learn  
- **Development Tools:**  
  - Jupyter Notebook  
  - VS Code  
  - Git & GitHub  
