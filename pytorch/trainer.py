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
        step = 0
        with torch.enable_grad():
            for _ in range(max_epoch):
                for batch_idx, (X, y) in enumerate(dataloader):
                    # print(f"batch {batch_idx}: {X.shape}, {y.shape}")
                    X_device = X.to(device)
                    pred = model(X_device)
                    #print(f"pred = {pred}")
                    #loss = self.compute_loss(pred, y)
                    y_device = y.to(device)
                    loss = F.mse_loss(pred, y_device)
                    if step % 10 == 0:
                        print(f"step: {step}, loss = {loss}")

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1


    def compute_loss(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        pass


def main():
    input_size = 2
    dataset = SimpleDataset(seq_length=input_size, file_path="./data.txt")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    trainer = Trainer()
    model = SimpleModel(input_size, input_size, hidden_layers=5, hidden_layer_dim=16)
    trainer.train(model=model, dataloader=dataloader, max_epoch=15)

if __name__ == "__main__":
    main()