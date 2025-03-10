import re
import pandas as pd
import config
from typing import List, Union
from pathlib import Path

_email_pattern = re.compile(r"[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_url_pattern = re.compile(r"http[s]?://\S+|www\.\S+")
_username_pattern = re.compile(r"\B@[a-zA-Z0-9_-]+")
_extra_spaces = re.compile(r"\s+")

# Replace usernames and links for placeholders: "@user" and "http"
def clean_text(text: str, url=True, email=True, username=True) -> str:
    if url:
        text = _url_pattern.sub('http', text)
    if email:
        text = _email_pattern.sub('<email>', text)
    if username:
        text = _username_pattern.sub('@user', text)
    text = _extra_spaces.sub(' ', text)
    return text


def process_dataset(dataset_path: Union[str, Path], target_names: List) -> Path:
    processed_path = Path(config.DATA_DIR) / 'processed' / Path(dataset_path).stem
    processed_path.mkdir(parents=True, exist_ok=True)
    # Process train dataset
    train_df = pd.read_csv(dataset_path / 'train.csv')
    train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
    train_df.to_csv(processed_path / 'train.csv')
    # Process test dataset
    test_comments = pd.read_csv(dataset_path / 'test.csv')
    test_labels = pd.read_csv(dataset_path / 'test_labels.csv')
    test_df = test_comments.merge(test_labels, on='id')
    test_df = test_df[~(test_df[target_names] == -1).any(axis=1)]   # remove rows where any `label == -1`
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
    test_df.to_csv(processed_path / 'test.csv')
    return processed_path


if __name__ == '__main__':
    dataset_path = Path(config.DATA_DIR) / 'raw/competitions/jigsaw-toxic-comment-classification-challenge'
    target_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    process_dataset(dataset_path, target_names)
