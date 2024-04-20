import torch, nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.corpus import wordnet
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import pandas as pd
import os, re, json
from torchvision import models
import torchvision
from torch import nn


class VQAModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(VQAModel, self).__init__()

        self.image_embedding = torchvision.models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
        self.image_embedding.classifier = self.image_embedding.classifier[:-1]
        self.fc1 = nn.Linear(4096, 1024)
        
        
        self.vocab_size = kwargs.pop("vocab_size", 2048)
        self.embedding_size = kwargs.pop("embedding_size", 300)
        self.hidden_size = kwargs.pop("hidden_size", 1024)
        self.lstm_layers = kwargs.pop("lstm_layers", 1)
        self.classes = kwargs.pop("classes", 1024)
        self.sequence_length = kwargs.pop("sequence_length", 30)
        
        self.text_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(input_size = self.embedding_size, hidden_size = self.hidden_size, num_layers  = self.lstm_layers, batch_first = True)
        self.fc2 = nn.Linear(self.hidden_size * self.sequence_length, 1024)
        
        self.classifier = nn.Linear(1024, self.classes)
        
    def forward(self, images, prompts):
        image_features = self.image_embedding(images) # (batch_size, 3, 224, 224)
        image_features = self.fc1(image_features) # (batch_size, 4096)
        
        text_features = self.text_embedding(prompts) # (batch_size, max_length, embedding_size)
        text_features, (_, _) = self.lstm(text_features) # (batch_size, max_length, hidden_size)
        
        text_features = text_features.contiguous().view(text_features.size(0), -1)

        text_features = self.fc2(text_features) # (batch_size, max_length, 1024)
        
        features = torch.mul(image_features, text_features)
        features = self.classifier(features)
        
        return features
