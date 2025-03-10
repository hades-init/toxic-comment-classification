import kaggle
import config
import zipfile
from typing import Union
from pathlib import Path
from src.data.preprocessing import clean_text
import pandas as pd

# Download dataset
def fetch_competition_data(competition: str, path: Union[str, Path]) -> Path:
    dataset_path = Path(path) / competition
    dataset_path.mkdir(parents=True, exist_ok=True)
    kaggle.api.competition_download_files(competition, path=dataset_path)
    archive_path = Path(dataset_path, competition).with_suffix('.zip')
    unarchive_dataset(archive_path)
    return dataset_path

# Unarchive dataset files
def unarchive_dataset(dataset_archive: Union[str, Path]):
    dataset_path = Path(dataset_archive).parent.resolve()
    tmp_dir = dataset_path / 'tmp'
    with zipfile.ZipFile(dataset_archive, 'r') as archive:
        archive.extractall(tmp_dir)

    for item in tmp_dir.iterdir():
        if zipfile.is_zipfile(item):
            with zipfile.ZipFile(item, 'r') as archive:
                archive.extractall(dataset_path)
            # remove inner zip files after extraction
            item.unlink()
            
    # remove 'tmp' folder
    tmp_dir.rmdir()
    # remove zip file after extraction
    dataset_archive.unlink()


if __name__ == '__main__':
    dataset_dir = Path(config.DATA_DIR) / 'raw' / 'competitions'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    fetch_competition_data('jigsaw-toxic-comment-classification-challenge', path=dataset_dir)