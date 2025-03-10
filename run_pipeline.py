from pathlib import Path
import config
from src.data.make_dataset import fetch_competition_data
from src.data import preprocessing
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model


def run_pipeline():
    # Step 1: Fetch dataset from kaggle
    download_dir = Path(config.DATA_DIR) / 'raw' / 'competitions'
    dataset_path = fetch_competition_data('jigsaw-toxic-comment-classification-challenge', path=download_dir)

    # Step 2: Preprocess train and test data
    process_train_data(dataset_path)
    process_test_data(dataset_path)

    # Step 3: Model training and validation
    checkpoint = 'cardiffnlp/twitter-roberta-base'
    train_results = train_model(checkpoint, dataset_path)

    # Step 4: Model evaluation
    pretrained_model = train_results['model_path']
    eval_results = evaluate_model(pretrained_model, dataset_path)

    # Step 5: Generate report
    with open('reports/model_report.txt', 'w') as file:
        print("========== Training Results ==========", file=file)
        print(f"Train Loss: {train_results['train_loss']}", file=file)
        print(f"Validation Loss {train_results['validation_loss']}", file=file)
        print("========= Evaluation Results =========", file=file)
        print(f"Classification Report:", file=file)
        print(eval_results['classification_report'], file=file)
