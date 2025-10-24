from torch.utils.data.dataloader import DataLoader
from simple_dataset import SimpleDataset

class SimpleDataLoader(DataLoader):
    def __init__(self):
        pass


def main():
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, batch in enumerate(dataloader):
        print(i, batch.shape, batch)

if __name__ == "__main__":
    main()
