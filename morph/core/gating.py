import torch
import torch.nn as nn
import torch.nn.functional as F


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
        batch_size = x.shape[0]
        router_logits = self.router(x)  # [batch_size, num_experts]
        
        if self.routing_type == "noisy_top_k" and training:
            # Add noise for exploration during training
            noise = torch.randn_like(router_logits) * 0.2
            router_logits = router_logits + noise
        
        # Get top-k routing weights and corresponding expert indices
        routing_weights, expert_indices = torch.topk(
            F.softmax(router_logits, dim=1), self.k, dim=1
        )
        
        # Normalize the routing weights to sum to 1
        routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)
        
        # Compute uncertainty as the entropy of the routing distribution
        full_probs = F.softmax(router_logits, dim=1)
        entropy = -torch.sum(full_probs * torch.log(full_probs + 1e-10), dim=1).mean()
        uncertainty = entropy / torch.log(torch.tensor(self.num_experts, dtype=torch.float))
        
        return routing_weights, expert_indices, uncertainty
    
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
        
        # Create new router
        new_router = nn.Sequential(
            nn.Linear(self.input_size, self.input_size // 2),
            nn.ReLU(),
            nn.Linear(self.input_size // 2, num_experts)
        )
        
        # Copy existing weights for existing experts
        with torch.no_grad():
            new_router[0].weight.copy_(old_router[0].weight)
            new_router[0].bias.copy_(old_router[0].bias)
            new_router[2].weight[:self.num_experts-1].copy_(old_router[2].weight)
            new_router[2].bias[:self.num_experts-1].copy_(old_router[2].bias)
        
        self.router = new_router
