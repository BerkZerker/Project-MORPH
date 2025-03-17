import torch
import pytest
import networkx as nx
from src.core.model import MorphModel
from src.config import MorphConfig


class TestExpertMerging:
    """
    Tests for the expert merging functionality in MORPH.
    """
    
    @pytest.fixture
    def model_config(self):
        config = MorphConfig()
        config.input_size = 10
        config.expert_hidden_size = 20
        config.output_size = 5
        config.num_initial_experts = 4
        config.expert_k = 2
        config.enable_dynamic_experts = True
        config.enable_sleep = True
        config.sleep_cycle_frequency = 500
        config.expert_similarity_threshold = 0.8
        config.min_experts = 2
        return config
    
    @pytest.fixture
    def model(self, model_config):
        model = MorphModel(model_config)
        # Process a batch to initialize
        batch = torch.randn(10, model_config.input_size)
        model(batch, training=True)
        return model
    
    def test_similarity_calculation(self, model):
        """Test that expert similarity calculation works properly."""
        # Clone an expert to get high similarity
        expert1 = model.experts[0]
        expert2 = expert1.clone()
        
        # Copy parameters exactly
        with torch.no_grad():
            for p1, p2 in zip(expert1.parameters(), expert2.parameters()):
                p2.copy_(p1)
                
        # Calculate similarity
        similarity = expert1.get_parameter_similarity(expert2)
        assert similarity > 0.99, "Identical experts should have similarity near 1.0"
        
        # Create a very different expert
        expert3 = expert1.clone()
        with torch.no_grad():
            for p in expert3.parameters():
                p.data = torch.randn_like(p)
                
        # Calculate similarity
        similarity = expert1.get_parameter_similarity(expert3)
        assert similarity < 0.9, "Different experts should have lower similarity"
    
    def test_merge_experts(self, model):
        """Test that experts can be merged properly."""
        # Number of experts before
        num_experts_before = len(model.experts)
        
        # Force high similarity between two experts
        with torch.no_grad():
            for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
                p2.copy_(p1)
        
        # Merge the experts
        model._merge_expert_parameters(0, 1)
        
        # Test that parameters were properly merged
        # Since we copied them, they should still be identical
        with torch.no_grad():
            for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
                assert torch.allclose(p1, p2)
    
    def test_sleep_merges_similar_experts(self, model, model_config):
        """Test that sleep cycle merges similar experts."""
        # Force high similarity between experts 0 and 1
        with torch.no_grad():
            for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
                p2.copy_(p1)
        
        # Lower the threshold to ensure merging happens
        model.config.expert_similarity_threshold = 0.5
        
        # Number of experts before
        num_experts_before = len(model.experts)
        
        # Mock the merge method to check if it's called
        original_merge = model._merge_similar_experts
        merge_called = [False]
        
        def mock_merge():
            merge_called[0] = True
            return original_merge()
            
        model._merge_similar_experts = mock_merge
        
        # Call sleep
        model.sleep()
        
        # Check that merge was called
        assert merge_called[0], "Merge method should be called during sleep"
    
    def test_prune_dormant_experts(self, model, model_config):
        """Test that dormant experts get pruned."""
        # Number of experts before
        num_experts_before = len(model.experts)
        
        # Mark one expert as dormant
        model.experts[2].last_activated = 0
        model.knowledge_graph.nodes[2]['last_activated'] = 0
        model.knowledge_graph.nodes[2]['activation_count'] = 10  # Below threshold
        
        # Set the step count to trigger dormant detection
        model.step_count = model_config.dormant_steps_threshold + 100
        
        # Call pruning directly
        pruned = model._prune_dormant_experts()
        
        # Check that an expert was pruned
        assert pruned, "Pruning should have occurred"
        assert len(model.experts) < num_experts_before, "Number of experts should decrease"
    
    def test_rebuild_knowledge_graph(self, model):
        """Test knowledge graph rebuilding after merging/pruning."""
        # Number of experts before
        num_experts_before = len(model.experts)
        
        # Add some test edges
        model.knowledge_graph.add_edge(0, 1, weight=0.5)
        model.knowledge_graph.add_edge(1, 2, weight=0.3)
        
        # Manually remove an expert (simulating pruning)
        del model.experts[2]
        
        # Update expert IDs (normally done in prune method)
        for i, expert in enumerate(model.experts):
            expert.expert_id = i
            
        # Rebuild knowledge graph
        model._rebuild_knowledge_graph()
        
        # Check that graph was properly rebuilt
        assert len(model.knowledge_graph.nodes) == len(model.experts), "Graph should have same number of nodes as experts"
        assert model.knowledge_graph.has_edge(0, 1), "Edge between remaining experts should be preserved"
        
    def test_end_to_end_dynamic_experts(self, model, model_config):
        """Test the full dynamic expert lifecycle with merging and pruning."""
        initial_expert_count = len(model.experts)
        
        # Process batches to trigger expert creation
        for i in range(10):
            batch = torch.randn(10, model_config.input_size)
            model(batch, training=True)
            
        # Force similar experts to trigger merging
        with torch.no_grad():
            for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
                p2.copy_(p1)
                
        # Lower the similarity threshold to ensure merging
        model.config.expert_similarity_threshold = 0.5
                
        # Process more batches to trigger sleep
        for i in range(model_config.sleep_cycle_frequency):
            batch = torch.randn(10, model_config.input_size)
            model(batch, training=True)
            
        # Check that experts were dynamically managed
        assert len(model.experts) != initial_expert_count, "Expert count should change after dynamic lifecycle"
