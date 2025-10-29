import torch
from torch.utils.data.dataset import Dataset
from typing import Optional, Tuple
import math

class SimpleDataset(Dataset):
    def __init__(self, seq_length: int, file_path: Optional[str] = None):
        if file_path is not None:
            with open(file_path, 'r') as f:
                self.data = f.read()
        else:
            raise ValueError("file_path must be provided")

        self.chars = sorted(set(self.data))
        # print(self.chars)
        self.charToIndex = {ch: i for i, ch in enumerate(self.chars)}

        ids = []
        for ch in self.data:
            # print(f"{ch},")
            ids.append(self.charToIndex[ch])
        
        
        self.seq_length = seq_length
        # chosen_data = ids
        # #discard the remainder
        # num_records = len(ids) // self.seq_length
        # chosen_data = ids[:(num_records * self.seq_length)]

        self.tensor_data = torch.Tensor(ids)
        print(len(self.tensor_data))
        # print(len(self.data))
        # self.tensor_data_reshaped = self.tensor_data.view(self.seq_length, num_records)
        # print(self.tensor_data_reshaped.shape)
        # print(self.tensor_data_reshaped)
        

    def __len__(self) -> int:
        return max(0, len(self.tensor_data) - self.seq_length)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tensor_data[idx: idx + self.seq_length], \
            self.tensor_data[idx + 1: idx + 1 + self.seq_length]


def main():
    sample_dataset = SimpleDataset(4, "./data.txt")
    for i, (sample_x, sample_y) in enumerate(sample_dataset):
        if (i >= 10):
            break
        print(i, sample_x, sample_y)


if __name__ == "__main__":
    main()