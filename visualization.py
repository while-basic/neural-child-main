# visualization.py
# Description: Module for visualizing emotional states and neural network architectures
# Created by: Christopher Celaya

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import networkx as nx
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import HTML
import pandas as pd

# Set the style globally
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

class EmotionalStateVisualizer:
    def __init__(self):
        self.emotional_history = []
        self.time_points = []
    
    def add_emotional_state(self, emotional_state, timestamp):
        if isinstance(emotional_state, torch.Tensor):
            emotional_state = emotional_state.cpu().detach().numpy()
        self.emotional_history.append(emotional_state)
        self.time_points.append(timestamp)
    
    def plot_emotional_timeline(self, save_path=None):
        if not self.emotional_history:
            print("No emotional data to plot.")
            return
        
        data = np.array(self.emotional_history)
        time_points = np.array(self.time_points)
        
        plt.figure(figsize=(15, 8))
        emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
        
        for i, emotion in enumerate(emotions):
            plt.plot(time_points, data[:, i], label=emotion, linewidth=2)
        
        plt.title('Emotional State Timeline', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Emotional Intensity', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_emotional_heatmap(self, save_path=None):
        if not self.emotional_history:
            print("No emotional data to plot.")
            return
        
        data = np.array(self.emotional_history)
        emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
        correlations = np.corrcoef(data.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, 
                   annot=True, 
                   cmap='coolwarm', 
                   xticklabels=emotions,
                   yticklabels=emotions)
        plt.title('Emotional State Correlations', fontsize=16)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_3d_emotional_trajectory(self, save_path=None):
        if not self.emotional_history:
            print("No emotional data to plot.")
            return
        
        data = np.array(self.emotional_history)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='lines+markers',
            marker=dict(
                size=4,
                color=data[:, 3],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Surprise Level')
            ),
            line=dict(color='darkblue', width=2),
            text=[f'Time: {t}<br>Joy: {d[0]:.2f}<br>Trust: {d[1]:.2f}<br>Fear: {d[2]:.2f}<br>Surprise: {d[3]:.2f}'
                  for t, d in zip(self.time_points, data)],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title='3D Emotional State Trajectory',
            scene=dict(
                xaxis_title='Joy',
                yaxis_title='Trust',
                zaxis_title='Fear'
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        return fig

class NeuralNetworkVisualizer:
    def __init__(self, model):
        self.model = model
    
    def visualize_architecture(self, save_path=None):
        """Visualize network architecture using matplotlib instead of Graphviz"""
        G = nx.DiGraph()
        
        # Add nodes for each layer
        layer_sizes = []
        layer_names = []
        
        # Extract layer information
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_sizes.append(module.in_features)
                layer_names.append(f"{name}\n({module.in_features}→{module.out_features})")
                if len(layer_sizes) == 1:  # First layer
                    G.add_node(0, name=layer_names[-1], size=module.in_features)
                G.add_node(len(layer_sizes), name=f"{name}_out\n({module.out_features})", size=module.out_features)
                G.add_edge(len(layer_sizes)-1, len(layer_sizes))
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=[1000 + size*10 for size in [G.nodes[node]['size'] for node in G.nodes]],
                             alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Add labels
        labels = {node: G.nodes[node]['name'] for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title("Neural Network Architecture", pad=20)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_weight_distributions(self, save_path=None):
        """Plot weight distributions for each layer"""
        plt.figure(figsize=(15, 10))
        num_layers = len(list(self.model.named_parameters()))
        current_plot = 1
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                plt.subplot(num_layers//2 + num_layers%2, 2, current_plot)
                weights = param.cpu().detach().numpy().flatten()
                sns.histplot(weights, bins=50, kde=True)
                plt.title(f'Weight Distribution - {name}')
                plt.xlabel('Weight Value')
                plt.ylabel('Count')
                current_plot += 1
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def create_activation_heatmap(self, input_data, save_path=None):
        """Create activation heatmap for each layer"""
        self.model.eval()
        activations = {}
        
        def hook_fn(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                handles.append(module.register_forward_hook(hook_fn(name)))
        
        with torch.no_grad():
            self.model(input_data)
        
        for handle in handles:
            handle.remove()
        
        plt.figure(figsize=(15, 10))
        current_plot = 1
        num_layers = len(activations)
        
        for name, activation in activations.items():
            plt.subplot(num_layers//2 + num_layers%2, 2, current_plot)
            act_data = activation.cpu().numpy().mean(axis=0)
            sns.heatmap(act_data.reshape(-1, 1), cmap='viridis')
            plt.title(f'Activation Heatmap - {name}')
            current_plot += 1
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def plot_learning_metrics(telemetry, save_path=None):
    metrics = ['loss', 'emotional_stability', 'conversation_quality']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5*len(metrics)))
    
    for i, metric in enumerate(metrics):
        if metric in telemetry and telemetry[metric]:
            axes[i].plot(telemetry[metric], label=metric)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_psychological_metrics(telemetry, save_path=None):
    if 'psychological' not in telemetry:
        print("No psychological metrics found in telemetry.")
        return
    
    metrics = telemetry['psychological']
    categories = list(metrics.keys())
    values = [np.mean(metrics[cat]) if metrics[cat] else 0 for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Psychological Development'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Psychological Development Metrics'
    )
    
    if save_path:
        fig.write_html(save_path)
    return fig 