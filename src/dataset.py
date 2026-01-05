import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO

class MultimodalDataset(Dataset):
    def __init__(self, texts, image_urls, labels, tokenizer, transform):
        self.texts = texts
        self.image_urls = image_urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Text
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        # Image
        url = self.image_urls[idx].split(",")[0]
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros(3,224,224)

        return {
            "input_ids": encoding['input_ids'].squeeze(),
            "attention_mask": encoding['attention_mask'].squeeze(),
            "image": image,
            "labels": torch.tensor(self.labels[idx])
        }
