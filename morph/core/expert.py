import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        
        # Specialization tracking
        self.specialization_score = 0.5  # 0.0 = general, 1.0 = specialized
        self.input_feature_history = {}  # Map of feature hash to count
        self.input_feature_centroid = None  # Representative centroid of inputs
        
        # Performance tracking
        self.performance_history = []  # List of (step, loss) tuples
        self.confidence_score = 0.5  # 0.0 = unconfident, 1.0 = confident
        
        # Device tracking
        self.device = torch.device('cpu')  # Default to CPU, will be updated when moved
        
        # Build layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Store architecture parameters for cloning
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
    
    def forward(self, x, update_stats=True):
        """
        Forward pass through the expert.
        
        Args:
            x: Input tensor
            update_stats: Whether to update expert statistics
            
        Returns:
            Expert output
        """
        # Update device tracking and ensure network is on the same device as input
        if x.device != self.device:
            self.device = x.device
            # Move the network to the same device as the input
            self.network = self.network.to(x.device)
            
        if update_stats:
            self.activation_count += x.size(0)  # Count each sample in the batch
            
            # Update input feature statistics if not in sleep mode
            if x.size(0) <= 32:  # Don't track large batch statistics for memory efficiency
                with torch.no_grad():
                    # Compute mean input feature vector using vectorized operations
                    # Keep on device as long as possible to reduce transfers
                    mean_features = torch.mean(x, dim=0).detach()
                    
                    # Only move to CPU for storage if needed
                    mean_features_cpu = mean_features.cpu()
                    
                    # Update running centroid with vectorized operations
                    if self.input_feature_centroid is None:
                        self.input_feature_centroid = mean_features_cpu
                    else:
                        # Vectorized update using tensor operations
                        self.input_feature_centroid = 0.9 * self.input_feature_centroid + 0.1 * mean_features_cpu
                    
                    # Update specialization score based on input consistency
                    if hasattr(self, 'last_inputs') and self.last_inputs is not None:
                        # Calculate similarity between current and previous inputs
                        # Use vectorized operations for better performance
                        similarity = F.cosine_similarity(
                            mean_features_cpu.unsqueeze(0),
                            self.last_inputs.unsqueeze(0)
                        )[0].item()
                        
                        # Use vectorized update with threshold-based logic
                        if similarity > 0.7:  # High similarity threshold
                            # Increase specialization score (more specialized)
                            self.specialization_score = min(1.0, self.specialization_score + 0.01)
                        elif similarity < 0.3:  # Low similarity threshold
                            # Decrease specialization score (more general)
                            self.specialization_score = max(0.0, self.specialization_score - 0.005)
                    
                    # Store current inputs for next comparison
                    self.last_inputs = mean_features_cpu
        
        # Forward pass through the network
        return self.network(x)
        
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
    
    def clone(self):
        """
        Create a clone of this expert with the same architecture but 
        re-initialized weights.
        
        Returns:
            A new Expert instance
        """
        new_expert = Expert(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        
        # Copy centroid (to preserve specialization starting point)
        if self.input_feature_centroid is not None:
            new_expert.input_feature_centroid = self.input_feature_centroid.clone()
            
        return new_expert
    
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
        
        # Ensure both parameter vectors are on the same device
        if params1.device != params2.device:
            # Move params2 to the same device as params1
            params2 = params2.to(params1.device)
        
        # Compute cosine similarity
        return F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))[0]
    
    def get_specialization_score(self):
        """
        Calculate a specialization score for this expert.
        
        Higher scores indicate more specialized experts (focused on specific input patterns).
        Lower scores indicate more general experts.
        
        Returns:
            Specialization score between 0 and 1
        """
        return self.specialization_score
    
    def get_centroid_similarity(self, other_expert):
        """
        Compute similarity between this expert's input centroid and another expert's.
        
        This helps determine if experts are specializing in similar input domains.
        
        Args:
            other_expert: Another Expert instance to compare with
            
        Returns:
            Similarity score between 0 and 1, or None if centroids not available
        """
        if self.input_feature_centroid is None or other_expert.input_feature_centroid is None:
            return None
        
        # Get centroids and ensure they're on the same device
        centroid1 = self.input_feature_centroid.unsqueeze(0)
        centroid2 = other_expert.input_feature_centroid.unsqueeze(0)
        
        # Ensure both centroids are on the same device
        if centroid1.device != centroid2.device:
            # Move centroid2 to the same device as centroid1
            centroid2 = centroid2.to(centroid1.device)
            
        # Compute cosine similarity between centroids
        return F.cosine_similarity(centroid1, centroid2)[0]
    
    def update_confidence(self, loss_value):
        """
        Update the expert's confidence score based on loss values.
        
        Args:
            loss_value: The loss value from a recent forward pass
            
        Returns:
            Updated confidence score
        """
        # Keep history limited to recent performance
        MAX_HISTORY = 100
        self.performance_history.append(loss_value)
        if len(self.performance_history) > MAX_HISTORY:
            self.performance_history = self.performance_history[-MAX_HISTORY:]
            
        # Calculate confidence based on recent loss trend
        if len(self.performance_history) >= 10:
            recent_losses = self.performance_history[-10:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            
            # Normalize to a confidence score (lower loss = higher confidence)
            # Assuming loss values are typically in [0, 2] range
            self.confidence_score = max(0.0, min(1.0, 1.0 - (avg_loss / 2.0)))
        
        return self.confidence_score
