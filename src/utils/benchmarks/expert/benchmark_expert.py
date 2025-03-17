"""
Expert utilization tracking for continual learning benchmarks.
"""

import numpy as np
from typing import Dict, Set, Tuple


class BenchmarkExpert:
    """
    Expert utilization tracking for continual learning benchmarks.
    
    This class provides methods for tracking and analyzing expert utilization
    in MORPH models during continual learning.
    """
    
    def calculate_expert_utilization_metrics(self, model_name: str):
        """
        Calculate metrics related to expert utilization for MORPH models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of expert utilization metrics
        """
        if model_name not in self.expert_utilization:
            return {}
            
        metrics = {
            'expert_utilization': {},
            'expert_specialization': {},
            'task_expert_overlap': {}
        }
        
        # Calculate expert utilization across tasks
        for task_id, utilization in self.expert_utilization.items():
            metrics['expert_utilization'][task_id] = utilization
            
        # Calculate expert specialization across tasks
        for task_id, specialization in self.expert_specialization.items():
            metrics['expert_specialization'][task_id] = specialization
            
        # Calculate task-expert overlap (which experts are shared between tasks)
        task_ids = sorted(self.expert_utilization.keys())
        for i, task_i in enumerate(task_ids):
            for j, task_j in enumerate(task_ids[i+1:], i+1):
                # Experts used in both tasks
                experts_i = set(self.expert_utilization[task_i].keys())
                experts_j = set(self.expert_utilization[task_j].keys())
                
                common_experts = experts_i & experts_j
                
                # Calculate Jaccard similarity
                if experts_i or experts_j:
                    overlap = len(common_experts) / len(experts_i | experts_j)
                else:
                    overlap = 0.0
                    
                metrics['task_expert_overlap'][(task_i, task_j)] = overlap
        
        return metrics
