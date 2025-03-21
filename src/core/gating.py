import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatingNetwork(nn.Module):
    """
    Gating network that determines which experts to use for a given input.
    
    The gating network routes inputs to the most appropriate experts based on
    the input features.
    """
    
    def __init__(self, input_size, num_experts, k=2, routing_type="top_k"):
        """
        Initialize the gating network.
        
        Args:
            input_size: Dimension of input features
            num_experts: Number of experts to route between
            k: Number of experts to activate per input (for top-k routing)
            routing_type: Type of routing mechanism ("top_k" or "noisy_top_k")
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.routing_type = routing_type
        
        # Device tracking
        self.device = torch.device('cpu')  # Default to CPU, will be updated when moved
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, num_experts)
        )
        
        # Expert uncertainty threshold (for dynamic expert creation)
        self.uncertainty_threshold = 0.3  # Will trigger new expert creation
    
    def forward(self, x, training=True):
        """
        Compute routing probabilities for each expert.
        
        Args:
            x: Input tensor [batch_size, input_size]
            training: Whether in training mode (affects routing)
            
        Returns:
            Tuple of (routing_weights, expert_indices, uncertainty)
            - routing_weights: Tensor of shape [batch_size, k]
            - expert_indices: Tensor of shape [batch_size, k]
            - uncertainty: Scalar representing routing uncertainty
        """
        # Update device tracking
        if x.device != self.device:
            self.device = x.device
            
        batch_size = x.shape[0]
        
        # Forward pass through router network
        router_logits = self.router(x)  # [batch_size, num_experts]
        
        # Add noise for exploration during training if using noisy top-k routing
        if self.routing_type == "noisy_top_k" and training:
            # Generate noise directly on the same device as router_logits
            noise = torch.randn_like(router_logits) * 0.3
            router_logits = router_logits + noise
        
        # Apply temperature scaling to sharpen the distribution
        # This helps with task specialization in continual learning
        temperature = 0.8 if training else 1.0
        scaled_logits = router_logits / temperature
        
        # Compute softmax probabilities once and reuse
        routing_probs = F.softmax(scaled_logits, dim=1)
        
        # Get top-k routing weights and corresponding expert indices
        # This is a vectorized operation that runs efficiently on GPU
        routing_weights, expert_indices = torch.topk(routing_probs, self.k, dim=1)
        
        # Normalize the routing weights to sum to 1 (vectorized operation)
        routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)
        
        # Compute uncertainty as the entropy of the routing distribution
        # Reuse the full probability distribution we already computed
        full_probs = F.softmax(router_logits, dim=1)
        
        # Use stable computation for entropy with proper handling of small values
        log_probs = torch.log(full_probs + 1e-10)  # Add small epsilon for numerical stability
        entropy = -torch.sum(full_probs * log_probs, dim=1).mean()
        
        # Normalize by log(num_experts) to get a value between 0 and 1
        # Move the denominator tensor to the same device as entropy
        log_num_experts = torch.log(torch.tensor(self.num_experts, dtype=torch.float, device=self.device))
        uncertainty = entropy / log_num_experts
        
        return routing_weights, expert_indices, uncertainty
        
    def to(self, *args, **kwargs):
        """
        Override the to() method to track device changes.
        
        Args:
            *args, **kwargs: Arguments to pass to the parent to() method
            
        Returns:
            Self, after moving to the specified device
        """
        # Call the parent to() method
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        
        if device is not None:
            # Update device tracking
            self.device = device
            
        # Call the parent to() method
        return super().to(*args, **kwargs)
    
    def should_create_new_expert(self, uncertainty):
        """
        Determine if a new expert should be created based on uncertainty.
        
        Args:
            uncertainty: Routing uncertainty score
            
        Returns:
            Boolean indicating whether to create a new expert
        """
        return uncertainty > self.uncertainty_threshold
    
    def update_num_experts(self, num_experts):
        """
        Update the gating network when the number of experts changes.
        
        Args:
            num_experts: New number of experts
        """
        if num_experts <= self.num_experts:
            return
        
        # Create new router with more outputs
        old_router = self.router
        self.num_experts = num_experts
        self.k = min(self.k, num_experts)
        
        # Create new router on the same device as the old router
        new_router = nn.Sequential(
            nn.Linear(self.input_size, self.input_size // 2),
            nn.ReLU(),
            nn.Linear(self.input_size // 2, num_experts)
        ).to(self.device)  # Ensure new router is on the same device
        
        # Copy existing weights for existing experts
        with torch.no_grad():
            # Copy weights and biases for the first layer
            new_router[0].weight.copy_(old_router[0].weight)
            new_router[0].bias.copy_(old_router[0].bias)
            
            # Copy weights for existing experts in the output layer
            old_num_experts = old_router[2].weight.size(0)
            new_router[2].weight[:old_num_experts].copy_(old_router[2].weight)
            new_router[2].bias[:old_num_experts].copy_(old_router[2].bias)
            
            # Initialize new expert weights with small random values
            # This helps with faster specialization of new experts
            if old_num_experts < num_experts:
                # Calculate standard deviation of existing weights for proper scaling
                weight_std = old_router[2].weight.std().item()
                # Initialize new rows with scaled random values
                nn.init.normal_(
                    new_router[2].weight[old_num_experts:], 
                    mean=0.0, 
                    std=weight_std * 0.1  # Smaller than existing weights to encourage specialization
                )
        
        # Replace the router
        self.router = new_router
