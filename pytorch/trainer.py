import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from simple_model import SimpleModel
from simple_dataset import SimpleDataset
from self_attention_module import SelfAttentionModule
from torch.utils.data.dataloader import DataLoader


class Trainer():
    def __init__(self):
        pass

    def train(self, model: nn.Module, token_embedding: nn.Embedding, dataloader: DataLoader, max_epoch: int = 5):
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        
        model = model.to(device)
        token_embedding = token_embedding.to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(token_embedding.parameters()), lr=1e-4)

        model.train()
        step = 0
        with torch.enable_grad():
            for _ in range(max_epoch):
                for batch_idx, (X, y) in enumerate(dataloader):
                    #print(f"batch {batch_idx}: {X.shape}, {y.shape}")
                    #with token dims
                    #print(f"X_token: {X_token.shape}, X: {X.shape}")

                    X_moved = X.to(device) #[batch, seq]
                    y_moved = y.to(device)

                    X_device = token_embedding(X_moved.to(torch.int64)) #[batch, seq, d_model]
                    y_device = token_embedding(y_moved.to(torch.int64)) #[batch, seq, d_model]

                    pred = model(X_device)
                    #print(f"pred = {pred}")
                    #loss = self.compute_loss(pred, y)
                    loss = F.mse_loss(pred, y_device)
                    if step % 10 == 0:
                        print(f"step: {step}, loss = {loss}")

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1


    def compute_loss(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        pass


def attention():
    input_size = 3
    dataset = SimpleDataset(seq_length=input_size, file_path="./data.txt")
    print(f"num_chars={dataset.__len__()}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    vocab_size = 32
    d_model = 1024
    token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    
    trainer = Trainer()
    model = SelfAttentionModule(d_model=d_model, query_key_dim=128, seq_length=input_size)

    print(token_embedding.weight[0:5, 0:3])
    trainer.train(model=model, token_embedding=token_embedding, dataloader=dataloader, max_epoch=50)
    print(token_embedding.weight[0:5, 0:3])

def feedForward():
    input_size = 2
    dataset = SimpleDataset(seq_length=input_size, file_path="./data.txt")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    trainer = Trainer()
    model = SimpleModel(input_size, input_size, hidden_layers=5, hidden_layer_dim=16)

    trainer.train(model=model, dataloader=dataloader, max_epoch=20)

if __name__ == "__main__":
    # feedForward()
    attention()