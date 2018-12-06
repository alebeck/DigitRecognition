from torch.utils.data import Dataset


class CustomDataset(Dataset):
    
    def __init__(self, X, y, transform=None, test=False):
        self.X = X
        self.y = y
        self.transform = transform
        self.test = test
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index].astype('float32')
        
        if self.transform:
            x = self.transform(x)
            
        if self.test:
            return x
        
        y = self.y[index].astype('int64')
        
        return x, y