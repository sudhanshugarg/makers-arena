import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from simple_model import SimpleModel
from simple_dataset import SimpleDataset
from torch.utils.data.dataloader import DataLoader


class Trainer():
    def __init__(self):
        pass

    def train(self, model: nn.Module, dataloader: DataLoader, max_epoch: int = 5):
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        with torch.enable_grad():
            for _ in range(max_epoch):
                for batch_idx, (X, y) in enumerate(dataloader):
                    print(f"batch {batch_idx}: {X}, {y}")
                    X_device = X.to(device)
                    pred = model(X_device)
                    print(f"pred = {pred}")
                    #loss = self.compute_loss(pred, y)
                    y_device = y.to(device)
                    loss = F.mse_loss(pred, y_device)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


    def compute_loss(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        pass


def main():
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    trainer = Trainer()
    model = SimpleModel(2, 1)
    trainer.train(model=model, dataloader=dataloader)

if __name__ == "__main__":
    main()