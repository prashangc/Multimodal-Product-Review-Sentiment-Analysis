import torch
from torch.utils.data import DataLoader
from dataset import MultimodalDataset
from model import MultimodalSentimentModel
from transformers import BertTokenizer
from torchvision import transforms
import pandas as pd

# Load test dataset
df = pd.read_csv("../data/amazon_fashion_reviews.csv")
df = df.dropna(subset=['reviewText','imageURLs'])
df['sentiment'] = df['overall'].apply(lambda x: 0 if x<=2 else 1 if x==3 else 2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_dataset = MultimodalDataset(df['reviewText'].tolist(), df['imageURLs'].tolist(), df['sentiment'].tolist(), tokenizer, transform)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalSentimentModel().to(device)
model.load_state_dict(torch.load("model.pt"))  # After training

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask, images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import classification_report, accuracy_score
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["Negative","Neutral","Positive"]))
