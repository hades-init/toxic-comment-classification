import pandas as pd
import numpy as np
from tqdm import tqdm
import config
from pathlib import Path
from src.data import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from src.data.dataset_wrapper import ToxicCommentsDataset
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def train_model(checkpoint, dataset_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_df = pd.read_csv(dataset_path / 'train.csv')
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    texts = train_df['comment_text'].tolist()
    labels = train_df[toxicity_types].to_numpy()

    # Split data into train and validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, random_state=42)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Wrap dataset in `DataLoader`
    train_dataset = ToxicCommentsDataset(train_encodings, train_labels)
    val_dataset = ToxicCommentsDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Load model
    model = AutoModelForSequenceClassification(checkpoint)
    model.to(device)

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training and evaluation loop
    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()    # reset gradients
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            # back propagation
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Train Loss: {train_loss / len(train_loader)}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
            print(f"Validation Loss: {val_loss / len(val_loader)}")

    # Save model
    model_path = Path(config.MODELS_DIR) / 'pretrained/toxicity-classification'
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return {'train_loss': train_loss, 
            'validation_loss': val_loss, 
            'model_path': model_path}


if __name__ == '__main__':
    dataset_path = Path(config.DATA_DIR) / 'processed/jigsaw-toxic-comment-classification-challenge'
    checkpoint = 'cardiffnlp/twitter-roberta-base'
    train_results = train_model(checkpoint, dataset_path)
