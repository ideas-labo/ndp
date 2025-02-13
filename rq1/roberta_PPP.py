import os
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import KFold
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def preprocess_data(dataset, tokenizer):
    def tokenize_function(dataset):
        return tokenizer(dataset["text"], padding=True, truncation=True, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

repo_name = "./roberta-trainer-pp"
modelpath = "./model_pp"
datasetpath = "./data_pp/cppdatasets"

model_save_path = "./model_pp"

tokenizer = AutoTokenizer.from_pretrained(modelpath)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

type_list = ['NONE', 'Optimization', 'Function', 'Resource', 'Tradeoff']

# Inference on the test set
texts = []
true_labels = []
predictions = []

with open("./data_pp/cppp.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for line in lines:       
        texts.append(line)
            

# Initialize the model for inference
model = RobertaForSequenceClassification.from_pretrained(model_save_path, num_labels=5)
model.eval()

with open("./data_pp/predictions.txt", "w", encoding="utf-8") as out_file:
    for text in texts:
        with torch.no_grad():
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            first_part = text.split(' ')[0]
            out_file.write(f"{first_part}\t{type_list[prediction]}\n")
