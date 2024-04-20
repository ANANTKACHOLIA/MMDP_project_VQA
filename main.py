import torch
import torchvision.transforms as transforms
from train import train_model
from utils import load_statements, split_df, get_answer_space
from datasets import QADataset
from model import VQAModel
import torch, nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.corpus import wordnet
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import pandas as pd
import os, re, json
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import torchvision
from torch import nn
from tqdm import tqdm
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths and configurations
input_path = '/kaggle/input/daquar'
data_files = {
    "train": os.path.join(input_path, 'train.csv'),
    "test": os.path.join(input_path, 'test.csv')
}
image_root = os.path.join(input_path, "images")

# Load dataset
train_df, test_df = pd.read_csv(data_files['train']), pd.read_csv(data_files['test'])

# Split dataset
train_csv_path = 'train.csv'
test_csv_path = 'test.csv'
train_df, test_df = split_df(train_df, test_size=0.2, train_out=train_csv_path, test_out=test_csv_path)

# Load statements
statements_file = 'qa-full.txt'
statements = load_statements(statements_file)

# Get answer space
answer_space = get_answer_space(train_df)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Initialize dataset and dataloaders
train_dataset = QADataset(train_csv_path, image_root=image_root, transform=transform)
test_dataset = QADataset(test_csv_path, image_root=image_root, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

word_2_idx, idx_2_word = {}, {}
answer_2_idx, idx_2_answer = {}, {}
max_length_question = 0
words_counter, answer_counter = 0, 0
for line in train_df['question']:
    line = word_tokenize(line.lower())
    max_length_question = max(max_length_question, len(line))
    for i in line:
        if i not in word_2_idx:
            word_2_idx[i], idx_2_word[words_counter] = words_counter, i
            words_counter += 1

for line in test_df['question']:
    line = word_tokenize(line.lower())
    max_length_question = max(max_length_question, len(line))
    for i in line:
        if i not in word_2_idx:
            word_2_idx[i], idx_2_word[words_counter] = words_counter, i
            words_counter += 1
        
for line in train_df['answer']:
    line = word_tokenize(line.lower())
    for i in line:
        if i not in answer_2_idx:
            answer_2_idx[i], idx_2_answer[answer_counter] = answer_counter, i
            answer_counter += 1
        
for line in test_df['answer']:
    line = word_tokenize(line.lower())
    for i in line:
        if i not in answer_2_idx:
            answer_2_idx[i], idx_2_answer[answer_counter] = answer_counter, i
            answer_counter += 1


words = [None] * len(word_2_idx)

for word, index in word_2_idx.items():
    words[index] = word

answers = [None] * len(answer_2_idx)
for word, index in answer_2_idx.items():
    answers[index] = word
    
with open('dictionaries.txt', 'w') as fp:
    fp.write('\n'.join(words))
    
with open('answers-index.txt', 'w') as fp:
    fp.write('\n'.join(answers))
    

max_length_question = max_length_question * 4 // 3
answer_space_length = len(answer_2_idx)
words_length = len(word_2_idx)
# Initialize model
model = VQAModel(vocab_size=words_length + 1, sequence_length=max_length_question, hidden_size=512, lstm_layers=2, classes=answer_space_length + 1, embedding_size=512)
model.to(device)

# Train model
train_model(model, train_loader, test_loader, device, epochs=200, initial_lr=1e-5)

