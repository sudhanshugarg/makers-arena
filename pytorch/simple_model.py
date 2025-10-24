import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from typing import List

class SimpleModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int = 2, hidden_layer_dim: int = 32):
        is_bias = False
        self.layers = nn.ModuleList()
        input_layer: nn.Linear = nn.Linear(in_features=input_dim, out_features=hidden_layer_dim, bias=is_bias)
        self.layers.append(input_layer)

        for _ in range(hidden_layers):
            next_layer = nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim, bias=is_bias)
            self.layers.append(next_layer)

        output_layer = nn.Linear(in_features=hidden_layer_dim, out_features=output_dim)
        
        self.layers.append(output_layer)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x
        n = len(self.layers)
        for i in range(n - 1):
            layer = self.layers[i]
            output = F.relu(layer(output))

        return self.layers[n-1](output)
    
    def backward(self) -> None:
        pass


class ModelTrain():
    def __init__(self):
        pass


    def train(self, model: SimpleModel, train_dataset: DataLoader, max_epoch: int = 5):

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        with torch.enable_grad():
            for i in range(max_epoch):
                # break the dataset into batches
                for b_features, b_target in get_next_batch(train_dataset):
                    optimizer.zero_grad()
                    output = model(x)
                    loss = F.mse_loss(b_target, output)
                    loss.backward()
                    optimizer.step()


        # DONE!


    def eval(self):
        with torch.no_grad():
            pass
