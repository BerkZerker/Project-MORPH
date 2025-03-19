import torch
import torch.nn as nn
from unittest.mock import MagicMock


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.experts = nn.ModuleList([nn.Linear(10, 5) for _ in range(3)])
        self.gating = nn.Linear(10, 3)
        self.step_count = 0
        self.sleep_module = MagicMock()
        self.knowledge_graph = MagicMock()
        self.config = MagicMock()
        self.config.output_size = 5
        self.config.expert_k = 2
        self.config.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        self.device = torch.device("cuda:0")
        self.expert_device_map = {i: torch.device("cuda:0") for i in range(3)}
        self.enable_mixed_precision = False
        self.scaler = None
        
    def forward(self, x, training=True):
        return self.fc(x)
    
    def train_step(self, batch, optimizer, criterion):
        inputs, targets = batch
        outputs = self(inputs)
        loss = criterion(outputs, targets)
        return {'loss': loss.item(), 'accuracy': 95.0, 'num_experts': len(self.experts)}
    
    def evaluate(self, data_loader, criterion, device=None):
        return {'loss': 0.1, 'accuracy': 96.0, 'num_experts': len(self.experts)}
    
    def sleep(self):
        pass
