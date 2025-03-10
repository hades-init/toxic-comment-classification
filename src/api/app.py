from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from src.data import preprocessing
from config import MODELS_DIR
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Request model
class Comment(BaseModel):
    text: str

# Load model and tokenizer
model_path = MODELS_DIR / 'pretrained' / 'toxicity-classification'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Prediction function
def predict_toxicity(text: str):
    embeddings = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    input_ids = embeddings['input_ids'].to(device)
    attention_mask = embeddings['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prob_scores = torch.sigmoid(logits).cpu().numpy()[0]
    return {label: round(float(score), 4) for label, score in zip(toxicity_labels, prob_scores)}


@app.post("/predict")
async def predict(comment: Comment):
    text = comment.text
    text = preprocessing.clean_text(text)
    return predict_toxicity(text)