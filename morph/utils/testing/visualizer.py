"""
Test visualization framework for MORPH models.

This module provides the core functionality for capturing and visualizing
the state of MORPH models during test execution.
"""

import os
import time
import inspect
import functools
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

import torch
import numpy as np
import networkx as nx

from morph.core.model import MorphModel
from morph.utils.visualization.knowledge_graph import visualize_knowledge_graph
from morph.utils.visualization.expert_activation import (
    plot_expert_activations,
    visualize_expert_lifecycle,
    visualize_expert_specialization_over_time
)


class TestVisualizer:
    """
    Captures and visualizes the state of MORPH models during test execution.
    
    This class provides methods to track model state changes, capture key metrics,
    and generate visualizations that help understand what's happening during tests.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the test visualizer.
        
        Args:
            output_dir: Directory to save visualizations. If None, a default
                        directory will be created in the current working directory.
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'test_visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # State tracking
        self.test_name = None
        self.start_time = None
        self.end_time = None
        self.state_snapshots = []
        self.model_snapshots = []
        self.current_test_dir = None
        
    def start_test(self, test_name: str):
        """
        Start tracking a new test.
        
        Args:
            test_name: Name of the test being executed
        """
        self.test_name = test_name
        self.start_time = time.time()
        self.state_snapshots = []
        self.model_snapshots = []
        
        # Create test-specific output directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.current_test_dir = os.path.join(self.output_dir, f"{test_name}_{timestamp}")
        os.makedirs(self.current_test_dir, exist_ok=True)
        
        print(f"\n📊 Test visualization enabled for: {test_name}")
        print(f"📁 Visualizations will be saved to: {self.current_test_dir}\n")
    
    def end_test(self):
        """
        End tracking for the current test and generate summary visualizations.
        """
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        print(f"\n✅ Test completed in {duration:.2f}s")
        print(f"📊 Generated {len(self.state_snapshots)} visualization snapshots")
        
        # Generate summary visualizations if we have snapshots
        if self.state_snapshots:
            self._generate_summary_visualizations()
    
    def capture_state(self, model: MorphModel, step_name: str, 
                     additional_data: Optional[Dict[str, Any]] = None):
        """
        Capture the current state of the model.
        
        Args:
            model: The MorphModel instance being tested
            step_name: Name of the current test step
            additional_data: Any additional data to capture
        """
        if not self.test_name:
            raise RuntimeError("Must call start_test before capturing state")
        
        # Create a snapshot of the current state
        snapshot = {
            'step_name': step_name,
            'timestamp': time.time(),
            'step_number': len(self.state_snapshots),
            'model_state': self._capture_model_state(model),
            'additional_data': additional_data or {}
        }
        
        self.state_snapshots.append(snapshot)
        
        # Generate visualizations for this state
        self._visualize_snapshot(snapshot, model)
        
        return snapshot
    
    def _capture_model_state(self, model: MorphModel) -> Dict[str, Any]:
        """
        Capture key aspects of the model's current state.
        
        Args:
            model: The MorphModel instance
            
        Returns:
            Dictionary containing the captured state
        """
        state = {
            'num_experts': len(model.experts),
            'sleep_cycles_completed': getattr(model, 'sleep_cycles_completed', 0),
            'step_count': getattr(model, 'step_count', 0),
            'next_sleep_step': getattr(model, 'next_sleep_step', None),
            'adaptive_sleep_frequency': getattr(model, 'adaptive_sleep_frequency', None),
            'activation_buffer_size': len(getattr(model, 'activation_buffer', [])),
            'knowledge_graph_nodes': len(model.knowledge_graph.graph.nodes),
            'knowledge_graph_edges': len(model.knowledge_graph.graph.edges),
        }
        
        # Capture expert states
        expert_states = {}
        for i, expert in enumerate(model.experts):
            expert_states[i] = {
                'expert_id': expert.expert_id,
                'activation_count': expert.activation_count,
                'last_activated': expert.last_activated,
            }
        
        state['expert_states'] = expert_states
        
        # Capture knowledge graph node attributes
        kg_node_attrs = {}
        for node in model.knowledge_graph.graph.nodes:
            kg_node_attrs[node] = dict(model.knowledge_graph.graph.nodes[node])
        
        state['kg_node_attrs'] = kg_node_attrs
        
        return state
    
    def _visualize_snapshot(self, snapshot: Dict[str, Any], model: MorphModel):
        """
        Generate visualizations for a state snapshot.
        
        Args:
            snapshot: The state snapshot to visualize
            model: The MorphModel instance
        """
        step_name = snapshot['step_name']
        step_number = snapshot['step_number']
        
        # Create step-specific directory
        step_dir = os.path.join(self.current_test_dir, f"{step_number:02d}_{step_name}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Visualize knowledge graph
        kg_path = os.path.join(step_dir, "knowledge_graph.png")
        visualize_knowledge_graph(model, output_path=kg_path)
        
        # Visualize expert states
        self._visualize_expert_states(model, step_dir)
        
        # Visualize model metrics
        self._visualize_model_metrics(snapshot, step_dir)
        
        # Create HTML summary for this step
        self._create_step_html(snapshot, step_dir)
    
    def _visualize_expert_states(self, model: MorphModel, output_dir: str):
        """
        Visualize the current state of experts.
        
        Args:
            model: The MorphModel instance
            output_dir: Directory to save visualizations
        """
        # Create a bar chart of expert activation counts
        plt.figure(figsize=(10, 6))
        expert_ids = [e.expert_id for e in model.experts]
        activation_counts = [e.activation_count for e in model.experts]
        
        plt.bar(range(len(expert_ids)), activation_counts)
        plt.xlabel('Expert Index')
        plt.ylabel('Activation Count')
        plt.title('Expert Activation Counts')
        plt.xticks(range(len(expert_ids)), expert_ids)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "expert_activations.png"))
        plt.close()
        
        # If we have specialization scores, visualize them
        if all('specialization_score' in model.knowledge_graph.graph.nodes[e.expert_id] 
               for e in model.experts if e.expert_id is not None):
            
            plt.figure(figsize=(10, 6))
            spec_scores = []
            
            for e in model.experts:
                if e.expert_id is not None:
                    score = model.knowledge_graph.graph.nodes[e.expert_id].get('specialization_score', 0)
                    spec_scores.append(score)
                else:
                    spec_scores.append(0)
            
            plt.bar(range(len(expert_ids)), spec_scores)
            plt.xlabel('Expert Index')
            plt.ylabel('Specialization Score')
            plt.title('Expert Specialization Scores')
            plt.xticks(range(len(expert_ids)), expert_ids)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "expert_specialization.png"))
            plt.close()
    
    def _visualize_model_metrics(self, snapshot: Dict[str, Any], output_dir: str):
        """
        Visualize key model metrics.
        
        Args:
            snapshot: The state snapshot
            output_dir: Directory to save visualizations
        """
        model_state = snapshot['model_state']
        
        # Create a summary metrics figure
        plt.figure(figsize=(10, 6))
        
        metrics = [
            ('Number of Experts', model_state['num_experts']),
            ('Sleep Cycles', model_state['sleep_cycles_completed']),
            ('Step Count', model_state['step_count']),
            ('KG Nodes', model_state['knowledge_graph_nodes']),
            ('KG Edges', model_state['knowledge_graph_edges']),
            ('Buffer Size', model_state['activation_buffer_size'])
        ]
        
        labels, values = zip(*metrics)
        
        plt.bar(labels, values)
        plt.ylabel('Value')
        plt.title('Model Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "model_metrics.png"))
        plt.close()
    
    def _create_step_html(self, snapshot: Dict[str, Any], step_dir: str):
        """
        Create an HTML summary for a test step.
        
        Args:
            snapshot: The state snapshot
            step_dir: Directory to save the HTML file
        """
        step_name = snapshot['step_name']
        model_state = snapshot['model_state']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Step: {step_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric {{ 
                    background: #f5f5f5; 
                    border-radius: 5px; 
                    padding: 10px; 
                    margin: 10px;
                    min-width: 150px;
                }}
                .images {{ display: flex; flex-wrap: wrap; }}
                .image-container {{ margin: 10px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Test Step: {step_name}</h1>
            <p>Step {snapshot['step_number']} at {time.strftime('%H:%M:%S', time.localtime(snapshot['timestamp']))}</p>
            
            <h2>Model Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Experts</h3>
                    <p>{model_state['num_experts']}</p>
                </div>
                <div class="metric">
                    <h3>Sleep Cycles</h3>
                    <p>{model_state['sleep_cycles_completed']}</p>
                </div>
                <div class="metric">
                    <h3>Step Count</h3>
                    <p>{model_state['step_count']}</p>
                </div>
                <div class="metric">
                    <h3>Next Sleep</h3>
                    <p>{model_state['next_sleep_step']}</p>
                </div>
                <div class="metric">
                    <h3>KG Nodes</h3>
                    <p>{model_state['knowledge_graph_nodes']}</p>
                </div>
                <div class="metric">
                    <h3>KG Edges</h3>
                    <p>{model_state['knowledge_graph_edges']}</p>
                </div>
                <div class="metric">
                    <h3>Buffer Size</h3>
                    <p>{model_state['activation_buffer_size']}</p>
                </div>
            </div>
            
            <h2>Visualizations</h2>
            <div class="images">
                <div class="image-container">
                    <h3>Knowledge Graph</h3>
                    <img src="knowledge_graph.png" alt="Knowledge Graph">
                </div>
                <div class="image-container">
                    <h3>Expert Activations</h3>
                    <img src="expert_activations.png" alt="Expert Activations">
                </div>
        """
        
        # Add specialization image if it exists
        if os.path.exists(os.path.join(step_dir, "expert_specialization.png")):
            html_content += """
                <div class="image-container">
                    <h3>Expert Specialization</h3>
                    <img src="expert_specialization.png" alt="Expert Specialization">
                </div>
            """
        
        # Add model metrics image
        html_content += """
                <div class="image-container">
                    <h3>Model Metrics</h3>
                    <img src="model_metrics.png" alt="Model Metrics">
                </div>
            </div>
        """
        
        # Add additional data if present
        if snapshot['additional_data']:
            html_content += """
            <h2>Additional Data</h2>
            <pre>
            """
            for key, value in snapshot['additional_data'].items():
                html_content += f"{key}: {value}\n"
            
            html_content += """
            </pre>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(os.path.join(step_dir, "index.html"), 'w') as f:
            f.write(html_content)
    
    def _generate_summary_visualizations(self):
        """
        Generate summary visualizations for the entire test.
        """
        # Create index.html that links to all step visualizations
        self._create_test_index_html()
        
        # Generate timeline visualization
        self._create_test_timeline()
    
    def _create_test_index_html(self):
        """
        Create an index.html file that links to all step visualizations.
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Visualization: {self.test_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ 
                    background: #f5f5f5; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin-bottom: 20px;
                }}
                .steps {{ 
                    display: flex; 
                    flex-direction: column;
                }}
                .step {{ 
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 5px 0;
                }}
                .step a {{ 
                    text-decoration: none;
                    color: #0066cc;
                    font-weight: bold;
                }}
                .step a:hover {{ 
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <h1>Test Visualization: {self.test_name}</h1>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p>Duration: {self.end_time - self.start_time:.2f} seconds</p>
                <p>Steps: {len(self.state_snapshots)}</p>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Test Timeline</h2>
            <img src="timeline.png" alt="Test Timeline" style="max-width: 100%; margin-bottom: 20px;">
            
            <h2>Test Steps</h2>
            <div class="steps">
        """
        
        # Add links to each step
        for snapshot in self.state_snapshots:
            step_name = snapshot['step_name']
            step_number = snapshot['step_number']
            step_dir = f"{step_number:02d}_{step_name}"
            
            html_content += f"""
                <div class="step">
                    <a href="{step_dir}/index.html">Step {step_number}: {step_name}</a>
                    <p>Timestamp: {time.strftime('%H:%M:%S', time.localtime(snapshot['timestamp']))}</p>
                </div>
            """
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(os.path.join(self.current_test_dir, "index.html"), 'w') as f:
            f.write(html_content)
    
    def _create_test_timeline(self):
        """
        Create a timeline visualization of the test steps.
        """
        if not self.state_snapshots:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Extract step names and timestamps
        step_names = [s['step_name'] for s in self.state_snapshots]
        timestamps = [s['timestamp'] - self.start_time for s in self.state_snapshots]
        
        # Plot timeline
        plt.plot(timestamps, range(len(timestamps)), 'o-', markersize=10)
        
        # Add step labels
        for i, (t, name) in enumerate(zip(timestamps, step_names)):
            plt.text(t + 0.1, i, name, verticalalignment='center')
        
        plt.yticks([])  # Hide y-axis
        plt.xlabel('Time (seconds)')
        plt.title('Test Timeline')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.current_test_dir, "timeline.png"))
        plt.close()


# Global instance for use by decorators
_default_visualizer = TestVisualizer()

def get_default_visualizer() -> TestVisualizer:
    """Get the default TestVisualizer instance."""
    return _default_visualizer
