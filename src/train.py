import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from dataset import MultimodalDataset
from model import MultimodalSentimentModel
from torchvision import transforms
from transformers import BertTokenizer

# Load your CSV and split
# Example code:
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/amazon_fashion_reviews.csv")
df = df.dropna(subset=['reviewText','imageURLs'])
df['sentiment'] = df['overall'].apply(lambda x: 0 if x<=2 else 1 if x==3 else 2)

train_texts, test_texts, train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    df['reviewText'], df['imageURLs'], df['sentiment'], test_size=0.3, random_state=42
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = MultimodalDataset(train_texts.tolist(), train_imgs.tolist(), train_labels.tolist(), tokenizer, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalSentimentModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop (1 epoch example)
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    images = batch['image'].to(device)
    labels = batch['labels'].to(device)
    
    outputs = model(input_ids, attention_mask, images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print("Batch Loss:", loss.item())
