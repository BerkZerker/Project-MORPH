import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_knowledge_graph_initialization():
    """Test that knowledge graph initializes correctly."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Check graph is initialized
    assert isinstance(kg.graph, nx.Graph)
    assert len(kg.graph.nodes) == 0
    assert len(kg.graph.edges) == 0
    
    # Check concept structures
    assert isinstance(kg.concept_embeddings, dict)
    assert isinstance(kg.expert_concepts, dict)
    
    # Check concept hierarchy
    assert isinstance(kg.concept_hierarchy, nx.DiGraph)
    

@visualize_test
def test_add_expert():
    """Test adding an expert to the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add an expert with visualization
    with capture_test_state(kg, "Add Expert"):
        kg.add_expert(
            expert_id=0,
            specialization_score=0.6,
            adaptation_rate=0.8
        )
    
    # Check expert was added
    assert 0 in kg.graph.nodes
    assert kg.graph.nodes[0]['type'] == 'expert'
    assert kg.graph.nodes[0]['specialization_score'] == 0.6
    assert kg.graph.nodes[0]['adaptation_rate'] == 0.8
    assert kg.graph.nodes[0]['activation_count'] == 0
    
    # Check expert-concept mapping was initialized
    assert 0 in kg.expert_concepts
    assert len(kg.expert_concepts[0]) == 0


@visualize_test
def test_add_edge():
    """Test adding edges between experts in the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add two experts
    kg.add_expert(0)
    kg.add_expert(1)
    
    # Add an edge between them with visualization
    with capture_test_state(kg, "Add Edge"):
        kg.add_edge(
            expert1_id=0,
            expert2_id=1,
            weight=0.75,
            relation_type='similarity'
        )
    
    # Check edge was added
    assert kg.graph.has_edge(0, 1)
    edge_data = kg.graph.get_edge_data(0, 1)
    assert edge_data['weight'] == 0.75
    assert edge_data['relation_type'] == 'similarity'
    
    # Test with invalid relation type
    kg.add_edge(0, 1, relation_type='invalid_type')
    # Should default to the valid relation type
    assert kg.graph.get_edge_data(0, 1)['relation_type'] in config.knowledge_relation_types


@visualize_test
def test_update_expert_activation():
    """Test updating expert activation information."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add an expert
    kg.add_expert(0)
    assert kg.graph.nodes[0]['activation_count'] == 0
    assert kg.graph.nodes[0]['last_activated'] == 0
    
    # Update activation with visualization
    with capture_test_state(kg, "Update Expert Activation"):
        kg.update_expert_activation(0, step=42)
    assert kg.graph.nodes[0]['activation_count'] == 1
    assert kg.graph.nodes[0]['last_activated'] == 42
    
    # Update again
    kg.update_expert_activation(0, step=100)
    assert kg.graph.nodes[0]['activation_count'] == 2
    assert kg.graph.nodes[0]['last_activated'] == 100
    
    # Test with non-existent expert
    kg.update_expert_activation(999, step=1)  # Should not raise error
    

@visualize_test
def test_update_expert_specialization():
    """Test updating expert specialization and adaptation rate."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add an expert
    kg.add_expert(0, specialization_score=0.5, adaptation_rate=1.0)
    
    # Update specialization with visualization
    with capture_test_state(kg, "Update Expert Specialization"):
        kg.update_expert_specialization(0, specialization_score=0.8)
    
    # Check specialization was updated
    assert kg.graph.nodes[0]['specialization_score'] == 0.8
    
    # Check adaptation rate was updated based on specialization
    # Higher specialization should lead to lower adaptation rate
    assert kg.graph.nodes[0]['adaptation_rate'] < 1.0
    

@visualize_test
def test_add_concept():
    """Test adding concepts to the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add a concept with visualization
    embedding = torch.randn(10)
    with capture_test_state(kg, "Add Concept"):
        kg.add_concept("concept1", embedding)
    
    # Check concept was added
    assert "concept1" in kg.concept_embeddings
    assert torch.equal(kg.concept_embeddings["concept1"], embedding)
    assert "concept1" in kg.concept_hierarchy.nodes
    
    # Add a child concept
    child_embedding = torch.randn(10)
    kg.add_concept("concept1.1", child_embedding, parent_concept="concept1")
    
    # Check parent-child relationship
    assert kg.concept_hierarchy.has_edge("concept1", "concept1.1")


@visualize_test
def test_link_expert_to_concept():
    """Test linking experts to concepts."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add expert and concept
    kg.add_expert(0)
    embedding = torch.randn(10)
    kg.add_concept("concept1", embedding)
    
    # Link expert to concept with visualization
    with capture_test_state(kg, "Link Expert to Concept"):
        kg.link_expert_to_concept(0, "concept1", strength=0.9)
    
    # Check linking worked
    assert "concept1" in kg.expert_concepts[0]
    
    # Test with non-existent concept
    kg.link_expert_to_concept(0, "nonexistent", strength=0.5)  # Should not raise error
    

@visualize_test
def test_get_similar_experts():
    """Test finding similar experts in the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add several experts
    for i in range(5):
        kg.add_expert(i)
    
    # Add similarity edges
    kg.add_edge(0, 1, weight=0.9, relation_type='similarity')
    kg.add_edge(0, 2, weight=0.3, relation_type='similarity')
    kg.add_edge(0, 3, weight=0.8, relation_type='similarity')
    kg.add_edge(0, 4, weight=0.5, relation_type='similarity')
    
    # Find similar experts with high threshold and visualization
    with capture_test_state(kg, "Get Similar Experts"):
        similar = kg.get_similar_experts(0, threshold=0.7)
    
    # Should return experts 1 and 3
    assert len(similar) == 2
    assert (1, 0.9) in similar
    assert (3, 0.8) in similar
    
    # Check sorting (highest similarity first)
    assert similar[0][1] >= similar[1][1]
    
    # Test with non-existent expert
    assert kg.get_similar_experts(999) == []


@visualize_test
def test_decay_edges():
    """Test edge decay functionality."""
    config = MorphConfig(
        knowledge_edge_decay=0.5,  # Aggressive decay for testing
        knowledge_edge_min=0.2
    )
    kg = KnowledgeGraph(config)
    
    # Add experts and edges
    kg.add_expert(0)
    kg.add_expert(1)
    kg.add_expert(2)
    
    kg.add_edge(0, 1, weight=1.0)
    kg.add_edge(0, 2, weight=0.3)  # Just above min threshold
    
    # Apply decay with visualization
    with capture_test_state(kg, "Decay Edges"):
        kg.decay_edges()
    
    # First edge should be decayed but still exist
    assert kg.graph.has_edge(0, 1)
    assert kg.graph.get_edge_data(0, 1)['weight'] == 0.5  # 1.0 * 0.5
    
    # Second edge should be decayed below threshold and removed
    assert not kg.graph.has_edge(0, 2)  # 0.3 * 0.5 = 0.15 < 0.2
    

@visualize_test
def test_prune_isolated_experts():
    """Test identifying isolated experts in the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add several experts
    for i in range(4):
        kg.add_expert(i)
    
    # Connect some experts
    kg.add_edge(0, 1, weight=0.8)
    kg.add_edge(1, 2, weight=0.6)
    # Expert 3 remains isolated
    
    # Find isolated experts with visualization
    with capture_test_state(kg, "Prune Isolated Experts"):
        isolated = kg.prune_isolated_experts()
    
    # Only expert 3 should be isolated
    assert len(isolated) == 1
    assert 3 in isolated


@visualize_test
def test_get_dormant_experts():
    """Test identifying dormant experts."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add experts with different activation patterns
    for i in range(4):
        kg.add_expert(i)
        
    # Update activation history
    kg.update_expert_activation(0, step=100)  # Recently activated
    kg.update_expert_activation(1, step=10)   # Dormant
    kg.update_expert_activation(2, step=20)   # Dormant
    # Expert 3 never activated
    
    # Make expert 0 have many activations (not dormant despite inactivity)
    kg.graph.nodes[0]['activation_count'] = 200
    
    # Set expert 2 to have few activations (dormant due to both inactivity and few activations)
    kg.graph.nodes[2]['activation_count'] = 5
    
    # Get dormant experts with visualization
    current_step = 100
    with capture_test_state(kg, "Get Dormant Experts"):
        dormant = kg.get_dormant_experts(
            current_step=current_step,
            dormancy_threshold=50,  # Inactive for 50+ steps
            min_activations=10      # Fewer than 10 activations
        )
    
    # Experts 1 and 2 should be dormant (inactive + few activations)
    # Expert 0 has too many activations to be dormant
    # Expert 3 was never activated so should be dormant
    assert 1 in dormant
    assert 2 in dormant
    assert 3 in dormant
    assert 0 not in dormant
    

@visualize_test
def test_merge_expert_connections():
    """Test merging expert connections when removing an expert."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add experts
    for i in range(5):
        kg.add_expert(i)
    
    # Add connections to expert 0 (to be removed)
    kg.add_edge(0, 1, weight=0.8, relation_type='similarity')
    kg.add_edge(0, 2, weight=0.6, relation_type='dependency')
    kg.add_edge(0, 3, weight=0.9, relation_type='specialization')
    # No connection to expert 4
    
    # Merge connections from expert 0 to experts 1 and 3 with visualization
    with capture_test_state(kg, "Merge Expert Connections"):
        kg.merge_expert_connections(0, target_ids=[1, 3])
    
    # Check that expert 0's connections were transferred
    # Expert 1 should now be connected to experts 2 and 3
    assert kg.graph.has_edge(1, 2)
    assert kg.graph.get_edge_data(1, 2)['relation_type'] == 'dependency'
    assert kg.graph.get_edge_data(1, 2)['weight'] < 0.6  # Should be reduced
    
    # Expert 3 should now be connected to expert 2
    assert kg.graph.has_edge(3, 2)
    assert kg.graph.get_edge_data(3, 2)['relation_type'] == 'dependency'
    assert kg.graph.get_edge_data(3, 2)['weight'] < 0.6  # Should be reduced
    
    # No connection should be created to expert 4
    assert not kg.graph.has_edge(1, 4)
    assert not kg.graph.has_edge(3, 4)
    
    # Test with non-existent source expert
    kg.merge_expert_connections(999, [1, 3])  # Should not raise error
    

@visualize_test
def test_rebuild_graph():
    """Test rebuilding the knowledge graph after expert count changes."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add experts
    for i in range(4):
        kg.add_expert(i)
        kg.graph.nodes[i]['custom_field'] = f"expert_{i}"
    
    # Add edges
    kg.add_edge(0, 1, weight=0.8)
    kg.add_edge(1, 2, weight=0.6)
    kg.add_edge(2, 3, weight=0.7)
    
    # Track original state
    original_node_count = len(kg.graph.nodes)
    original_edge_count = len(kg.graph.edges)
    
    # Rebuild with fewer experts (removing expert 3) with visualization
    with capture_test_state(kg, "Rebuild Graph"):
        kg.rebuild_graph(expert_count=3)
    
    # Check node count changed
    assert len(kg.graph.nodes) == 3
    assert original_node_count == 4
    
    # Check edges were preserved for remaining experts
    assert kg.graph.has_edge(0, 1)
    assert kg.graph.has_edge(1, 2)
    assert not kg.graph.has_edge(2, 3)  # This edge should be gone
    
    # Check node attributes were preserved
    assert kg.graph.nodes[0]['custom_field'] == "expert_0"
    
    # Check expert concepts mapping was updated
    assert len(kg.expert_concepts) == 3
    assert 3 not in kg.expert_concepts
