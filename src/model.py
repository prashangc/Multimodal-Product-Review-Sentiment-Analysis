import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Text
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(768, 256)
        
        # Image
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 256)
        
        # Fusion
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask, images):
        # Text
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1] # pooled output
        text_feat = self.text_fc(text_feat)
        
        # Image
        img_feat = self.resnet(images)
        
        # Concatenate
        fused = torch.cat((text_feat, img_feat), dim=1)
        x = self.dropout(torch.relu(self.fc1(fused)))
        return self.out(x)
