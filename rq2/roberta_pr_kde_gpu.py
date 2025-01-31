import numpy as np
import torch
import os
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, RobertaForSequenceClassification, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

repo_name = "./roberta-trainer-pr-gcc&clang"
modelpath = "./roberta_pretrain"
datasetpath = "./data_pr/cprdatasets_gcc&clang"

temp_dir = "/dev/shm"
os.environ['TMPDIR'] = temp_dir

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set the fraction of GPU memory to be used (e.g., 80%)
memory_fraction = 0.80
torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)


# Define metrics computation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Define oversampling function
def oversample(dataset):
    pos_indices = [i for i, label in enumerate(dataset['label']) if label == 1]
    neg_indices = [i for i, label in enumerate(dataset['label']) if label == 0]
    diff = len(neg_indices) - len(pos_indices)
    if diff > 0:
        oversample_indices = np.random.choice(pos_indices, size=diff, replace=True)
        texts = dataset["text"]
        labels = dataset["label"]
        for idx in oversample_indices:
            texts.append(dataset[int(idx)]["text"])
            labels.append(dataset[int(idx)]["label"])
        dataset = Dataset.from_dict({"text": texts, "label": labels})
    return dataset.shuffle(seed=42)

# Define function to split long texts into segments
def split_into_segments(text, max_length=512, stride=256):
    tokens = tokenizer(text, truncation=False)["input_ids"]
    segments = []
    for i in range(0, len(tokens), stride):
        segment = tokens[i:i+max_length]
        if len(segment) < max_length:
            segment = segment + [tokenizer.pad_token_id] * (max_length - len(segment))
        segments.append(tokenizer.decode(segment))
        if len(segment) < max_length:
            break
    return segments

# Apply segmentation and tokenization to dataset
def tokenize_and_segment(dataset):
    tokenized_texts = []
    labels = []
    for text, label in zip(dataset["text"], dataset["label"]):
        segments = split_into_segments(text)
        for segment in segments:
            tokenized_texts.append(segment)
            labels.append(label)
    return {"text": tokenized_texts, "label": labels}

def tokenize_and_segment1(dataset, tokenizer):
    tokenized_texts = []
    labels = []
    for text, label in zip(dataset["text"], dataset["label"]):
        segments = split_into_segments1(text, tokenizer)
        for segment in segments:
            tokenized_texts.append(tokenizer.decode(segment, skip_special_tokens=True))
            labels.append(label)
    return Dataset.from_dict({"text": tokenized_texts, "label": labels})

def split_into_segments1(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer.encode(text, truncation=False)
    segments = []
    for i in range(0, len(tokens), stride):
        segment = tokens[i:i+max_length]
        segments.append(segment)
        if len(segment) < max_length:
            break
    return segments

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

raw_datasets = DatasetDict.load_from_disk(datasetpath)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

# Apply oversampling to training dataset
raw_train_dataset = raw_datasets['train']
oversampled_train_dataset = oversample(raw_train_dataset)

# Apply segmentation and tokenization to datasets
train_dataset = Dataset.from_dict(tokenize_and_segment(oversampled_train_dataset))
validation_dataset = Dataset.from_dict(tokenize_and_segment(raw_datasets["validation"]))
test_dataset = Dataset.from_dict(tokenize_and_segment(raw_datasets["test"]))

raw_datasets = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding=True, truncation=True, max_length=512)

def tokenize_and_segment(dataset, tokenizer):
    tokenized_texts = []
    labels = []
    for text, label in zip(dataset["text"], dataset["label"]):
        segments = split_into_segments(text, tokenizer)
        for segment in segments:
            tokenized_texts.append(segment)
            labels.append(label)
    return {"text": tokenized_texts, "label": labels}

def get_segment_probabilities(text, model, tokenizer, device='cpu'):
    segments = split_into_segments(text, tokenizer)
    all_probabilities = []
    for segment in segments:
        inputs = tokenizer(segment, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
        all_probabilities.append(probabilities[0])
    return np.mean(all_probabilities, axis=0)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = RobertaForSequenceClassification.from_pretrained(modelpath, num_labels=2)

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,  # Limit the total number of saved checkpoints
    evaluation_strategy="epoch",
    logging_dir=f'{repo_name}/logs',
    logging_steps=10,
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Define the early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=10,  # Number of evaluation steps with no improvement after which training will be stopped
    early_stopping_threshold=0.01  # Minimum change to qualify as an improvement
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]  # Add the early stopping callback
)


checkpoint_dir = repo_name
last_checkpoint = None
if os.path.isdir(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(checkpoints) > 0:
        last_checkpoint = max(checkpoints, key=os.path.getctime)

if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting training from scratch")
    trainer.train()
trainer.evaluate()

# predictions = []
# texts = raw_datasets["test"]["text"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


model.eval()
texts = []
true_labels = []
probabilities = []

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

sorted_lists = sorted(zip(texts, all_probabilities), key=lambda x: x[1][1], reverse=True)
sorted_text, sorted_probabilities = zip(*sorted_lists)

display_texts = [text.split(' ')[0] if ' ' in text else text for text in sorted_text]

with open("./data_pr/sorted_results.txt", "w", encoding="utf-8") as output_file:
    for index, (text, prob) in enumerate(zip(display_texts, sorted_probabilities), start=1):
        output_file.write(f"Rank {index}: Text: {text}, Probability: {prob[1]}\n")
        print(f"Rank {index}: Text: {text}, Probability: {prob[1]}")

X_plot = np.linspace(0, 1, 100)[:, np.newaxis]
colors = ['blue']
kernels = ['gaussian']
lw = 2

ncpp, cpp = zip(*all_probabilities)
p = np.array(cpp)
X = p.reshape(-1, 1)
bandwidths = np.logspace(-1, 1, 20)

grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=5)
grid.fit(X)

print("Best Bandwidth:", grid.best_estimator_.bandwidth)

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=grid.best_estimator_.bandwidth).fit(X)
    log_dens = kde.score_samples(X_plot)
    arr = np.exp(log_dens)
    np.save('data.npy', arr)


data = np.load('data.npy')
maxn = argrelextrema(data, np.greater)
minn = argrelextrema(data, np.less)
print(maxn)
print(minn)


