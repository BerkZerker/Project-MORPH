import torch


class TestDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=100, feature_dim=10):
        self.data = torch.randn(size, feature_dim)
        self.targets = torch.randint(0, 5, (size,))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
