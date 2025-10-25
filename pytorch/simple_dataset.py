import torch
from torch.utils.data.dataset import Dataset
from typing import Tuple

class SimpleDataset(Dataset):
    def __init__(self):
        self.n = 22
        self.training_data = [torch.tensor([1.0 + i, 1.0 - i]) for i in range(self.n)]
        self.label_data = [torch.tensor([101.0 + i]) for i in range(self.n)]

    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.training_data[idx], self.label_data[idx]


def main():
    sample_dataset = SimpleDataset()
    for i, (sample_x, sample_y) in enumerate(sample_dataset):
        print(i, sample_x, sample_y)


if __name__ == "__main__":
    main()