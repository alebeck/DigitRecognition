from torch.utils.data import Dataset


class CustomDataset(Dataset):
    
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y