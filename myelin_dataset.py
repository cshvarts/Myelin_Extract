from torch.utils.data import Dataset

class MyelinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx]["image"]
        label = self.data.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label
