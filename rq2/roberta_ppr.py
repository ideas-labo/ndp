import numpy as np
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define metrics computation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def get_segment_probabilities1(text, model, tokenizer, device='cpu'):
    segments = split_into_segments1(text, tokenizer)
    all_probabilities = []
    for segment in segments:
        inputs = tokenizer.decode(segment, skip_special_tokens=True)
        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
        all_probabilities.append(probabilities[0])
    return np.mean(all_probabilities, axis=0)

def split_into_segments1(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer.encode(text, truncation=False)
    segments = []
    for i in range(0, len(tokens), stride):
        segment = tokens[i:i+max_length]
        segments.append(segment)
        if len(segment) < max_length:
            break
    return segments

# Load model and tokenizer
modelpath = "./model_pr_mysql"
tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = RobertaForSequenceClassification.from_pretrained(modelpath, num_labels=2)

# Move model to device for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
texts = []
true_labels = []

# Read test data
with open("./data_pr/train_cpr_sppp.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            text, label = parts
            texts.append(text)
            label = int(label)
            true_labels.append(label)

print(len(texts))

# Get probabilities and display texts for test dataset
all_probabilities = []

for text in texts:
    probabilities = get_segment_probabilities1(text, model, tokenizer, device='cuda:0')
    all_probabilities.append(probabilities)

# Convert probabilities to numpy array
all_probabilities = np.array(all_probabilities)

# Calculate final probabilities for each text
predictions = np.argmax(all_probabilities, axis=1)

# Sort texts by probability
sorted_lists = sorted(zip(texts, all_probabilities), key=lambda x: x[1][1], reverse=True)
sorted_text, sorted_probabilities = zip(*sorted_lists)

# Display text content
display_texts = [text.split(' ')[0] if ' ' in text else text for text in sorted_text]

with open("./data_pr/sorted_results.txt", "w", encoding="utf-8") as output_file:
    for index, (text, prob) in enumerate(zip(display_texts, sorted_probabilities), start=1):
        output_file.write(f"Rank {index}: Text: {text}, Probability: {prob[1]}\n")
        print(f"Rank {index}: Text: {text}, Probability: {prob[1]}")
