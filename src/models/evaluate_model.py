import pandas as pd
from src.data import preprocessing
import config
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def evaluate_model(checkpoint, dataset_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_df = pd.read_csv(dataset_path / 'test.csv')
    target_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    test_texts = test_df['comment_text'].tolist()
    test_labels = test_df[target_names].to_numpy()

    # Tokenize
    test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Wrap dataset in `DataLoader`
    test_dataset = ToxicCommentsDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Load model
    model = AutoModelForSequenceClassification(checkpoint)
    model.to(device)

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Evaluation loop
    prob_scores = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            test_loss += loss.item()
            pred_prob = torch.sigmoid(logits).cpu().numpy()
            prob_scores.extend(pred_prob)
            true_labels.extend(labels.cpu().numpy())


    prob_scores = np.array(prob_scores)
    true_labels = np.array(true_labels, dtype=int)

    # Predict labels
    # [0.75, 0.3, 0.55, 0.15, 0.55, 0.45] - after fine tuning on validation set
    thresholds = [0.75, 0.3, 0.65, 0.15, 0.55, 0.65]
    pred_labels = (prob_scores > thresholds).astype(int)

    # Classification report
    eval_report = classification_report(true_labels, pred_labels, target_names=target_names)
    return {'prob_scores': prob_scores, 
            'pred_labels': pred_labels, 
            'classification_report': eval_report}


if __name__ == '__main__':
    dataset_path = Path(config.DATA_DIR) / 'processed/jigsaw-toxic-comment-classification-challenge/'
    checkpoint = Path(config.MODELS_DIR) / 'pretrained/toxicity-classification'
    eval_report = evaluate_model(checkpoint, dataset_path)
