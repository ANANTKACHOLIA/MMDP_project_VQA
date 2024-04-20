import torch
import os
from PIL import Image
import pandas as pd

class QADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset
        self.transform = kwargs.pop("transform", None)
        self.question_preprocess = kwargs.pop("question_preprocess", None)
        self.answer_preprocess = kwargs.pop("answer_preprocess", None)
        self.image_root = kwargs.pop("image_root", None)
        assert self.image_root is not None, "Image root is not defined"
        self.images_path, self.question, self.answer = self.load()
        self.to_tensor = torchvision.transforms.ToTensor()
        
    def load(self):
        images_path, question, answer = [], [], []
        
        for item in self.dataset:
            images_path += [item["image_id"]]
            question += [item["question"]]
            answer += [item["answer"]]
            
        images_path = [os.path.join(self.image_root, f'{id}.png') for id in images_path]

        if self.question_preprocess:
            question = [self.question_preprocess(item) for item in question]

        if self.answer_preprocess:
            answer = [self.answer_preprocess(item) for item in answer]
                    
        return images_path, question, answer
    
    def __len__(self):
        return len(self.images_path)
    
    def max_length(self):
        return self.max_lengthmax
    
    def __getitem__(self, index):
        image_data = Image.open(self.images_path[index]).convert("RGB")
        
        if self.transform:
            image_data = self.transform(image_data)
            
        if type(image_data) is not torch.Tensor:
            image_data = self.to_tensor(image_data)
                
        return image_data, torch.tensor(self.question[index], dtype=torch.long), torch.tensor(self.answer[index], dtype=torch.float)
