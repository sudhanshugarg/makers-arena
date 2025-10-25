import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int = 2, hidden_layer_dim: int = 32):
        super(SimpleModel, self).__init__()
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

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print (f"Using device: {device}")
    model = SimpleModel(4, 2).to(device)
    print(model)

    input_tensor = torch.randn(3, 4).to(device)
    output_tensor = model(input_tensor)
    print("Input Tensor:", input_tensor)
    print("Output Tensor:", output_tensor)


if __name__ == "__main__":
    main()