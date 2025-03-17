# Expert management module
from src.core.expert_management.expert_lifecycle import create_new_expert, merge_expert_parameters, merge_similar_experts
from src.core.expert_management.pruning import prune_dormant_experts
from src.core.expert_management.rebuilding import rebuild_knowledge_graph

__all__ = [
    'create_new_expert',
    'merge_expert_parameters',
    'merge_similar_experts',
    'prune_dormant_experts',
    'rebuild_knowledge_graph'
]
