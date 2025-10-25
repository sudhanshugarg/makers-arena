import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from simple_model import SimpleModel


class Trainer():
    def __init__(self):
        pass

    def train(self, model: nn.Module, dataloader: DataLoader, max_epoch: int = 5):
        device = torch.accelerator.current_accelerator.type if torch.accelerator.is_available() else "cpu"
        
        model = SimpleModel(4, 2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        with torch.enable_grad():
            for _ in range(max_epoch):
                for batch_idx, (X, y) in enumerate(dataloader):
                    X_device = batch.to(X)
                    output = model(X_device)
                    loss = self.compute_loss(output, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


    def compute_loss(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        pass