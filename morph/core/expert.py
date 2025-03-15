import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Base expert network that specializes in a particular subset of the data.
    
    Each expert is a simple feed-forward network that can be customized.
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """
        Initialize an expert network.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Size of hidden layers
            output_size: Dimension of output features
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        # Expert ID (will be set when added to MoE)
        self.expert_id = None
        
        # Activation history for tracking utilization
        self.activation_count = 0
        self.last_activated = 0  # Will store training step
        
        # Build layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the expert.
        
        Args:
            x: Input tensor
            
        Returns:
            Expert output
        """
        self.activation_count += 1
        return self.network(x)
    
    def clone(self):
        """
        Create a clone of this expert with the same architecture but 
        re-initialized weights.
        
        Returns:
            A new Expert instance
        """
        input_size = self.network[0].in_features
        hidden_size = self.network[0].out_features
        output_size = self.network[-1].out_features
        num_layers = (len(self.network) - 1) // 2
        
        return Expert(input_size, hidden_size, output_size, num_layers)
    
    def get_parameter_similarity(self, other_expert):
        """
        Compute cosine similarity between this expert's parameters and another expert.
        
        Args:
            other_expert: Another Expert instance to compare with
            
        Returns:
            Similarity score between 0 and 1
        """
        # Flatten parameters into vectors
        params1 = torch.cat([p.view(-1) for p in self.parameters()])
        params2 = torch.cat([p.view(-1) for p in other_expert.parameters()])
        
        # Compute cosine similarity
        return F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))[0]
