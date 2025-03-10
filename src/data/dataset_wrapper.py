from typing import List, Union
from torch.utils.data import Dataset

# Dataset
class ToxicCommentsDataset(Dataset):
    def __init__(self, encodings: dict, labels: Union[List, np.ndarray]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item