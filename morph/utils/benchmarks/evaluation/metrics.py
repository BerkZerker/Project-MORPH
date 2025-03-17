"""
Metrics for evaluating continual learning performance.
"""

from typing import Dict


def calculate_forgetting(accuracy_history: Dict[int, Dict[int, float]]):
    """
    Calculate forgetting metrics for each task.
    
    Forgetting measures how much performance on previous tasks decreases
    after learning new tasks.
    
    Args:
        accuracy_history: Dictionary mapping task IDs to accuracy history
        
    Returns:
        Dictionary of forgetting metrics
    """
    forgetting = {}
    
    for task_id in accuracy_history.keys():
        if task_id in accuracy_history:
            # Get maximum performance on this task before learning subsequent tasks
            history = accuracy_history[task_id]
            
            if len(history) > 1:
                # Calculate forgetting as the difference between max performance and final performance
                max_performance = max(list(history.values())[:-1])
                final_performance = list(history.values())[-1]
                forgetting[task_id] = max_performance - final_performance
            else:
                forgetting[task_id] = 0.0
    
    return forgetting


def calculate_forward_transfer(accuracy_history: Dict[int, Dict[int, float]], task_similarities=None):
    """
    Calculate forward transfer metrics.
    
    Forward transfer measures how learning a task improves performance on future tasks.
    
    Args:
        accuracy_history: Dictionary mapping task IDs to accuracy history
        task_similarities: Optional dictionary mapping task pairs (i,j) to similarity scores
        
    Returns:
        Dictionary of forward transfer metrics
    """
    forward_transfer = {}
    
    # Use task similarities if available
    if task_similarities:
        for (task_i, task_j), similarity in task_similarities.items():
            if task_i < task_j and task_i in accuracy_history and task_j in accuracy_history:
                # Get accuracy on task_j before and after seeing it
                pre_accuracy = accuracy_history[task_j].get(task_i, 0.0)
                post_accuracy = list(accuracy_history[task_j].values())[-1]
                
                # Weight transfer by task similarity
                forward_transfer[(task_i, task_j)] = (post_accuracy - pre_accuracy) * similarity
    
    return forward_transfer


def calculate_knowledge_retention(accuracy_history: Dict[int, Dict[int, float]]):
    """
    Calculate knowledge retention metrics.
    
    Knowledge retention measures how well the model retains knowledge of previous tasks.
    
    Args:
        accuracy_history: Dictionary mapping task IDs to accuracy history
        
    Returns:
        Dictionary of knowledge retention metrics
    """
    retention = {}
    
    for task_id in sorted(accuracy_history.keys()):
        if task_id in accuracy_history:
            history = accuracy_history[task_id]
            
            if len(history) > 1:
                # Calculate retention as ratio of final to maximum performance
                max_performance = max(list(history.values())[:-1])
                final_performance = list(history.values())[-1]
                
                if max_performance > 0:
                    retention[task_id] = final_performance / max_performance
                else:
                    retention[task_id] = 1.0  # No change
            else:
                retention[task_id] = 1.0  # Perfect retention
    
    return retention
