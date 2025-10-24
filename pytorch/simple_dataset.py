import torch
from torch.utils.data.dataset import Dataset

class SimpleDataset(Dataset):
    def __init__(self):
        self.data = [torch.tensor([32 + i, 32 - i]) for i in range(22)]

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    



def main():
    sample_dataset = SimpleDataset()
    for i, sample in enumerate(sample_dataset):
        print(i, sample, sample.shape)


if __name__ == "__main__":
    main()