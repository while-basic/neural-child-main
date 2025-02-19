# visualization.py
# Created by Christopher Celaya
# Real-time visualization module for monitoring neural child development

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class DevelopmentVisualizer:
    def __init__(self):
        # Set up seaborn style
        sns.set_theme(style="darkgrid")
        sns.set_context("talk")
        
        # Create figure with custom style
        self.fig = plt.figure(figsize=(15, 10))
        self.plots = {}
        self.last_update = datetime.now()
        self.update_interval = 5  # seconds
        
        # Set custom color palette
        self.colors = sns.color_palette("husl", 8)
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.colors)
        
    def initialize_plots(self):
        """Initialize the visualization dashboard"""
        self.fig.clear()
        
        # Create subplot grid
        gs = self.fig.add_gridspec(3, 3)
        
        # Emotional Development Plot
        self.plots['emotional'] = self.fig.add_subplot(gs[0, 0])
        self.plots['emotional'].set_title('Emotional Development')
        self.plots['emotional'].set_ylabel('Stability')
        
        # Learning Progress Plot
        self.plots['learning'] = self.fig.add_subplot(gs[0, 1])
        self.plots['learning'].set_title('Learning Progress')
        self.plots['learning'].set_ylabel('Loss')
        
        # Memory Usage Plot
        self.plots['memory'] = self.fig.add_subplot(gs[0, 2])
        self.plots['memory'].set_title('Memory Usage')
        self.plots['memory'].set_ylabel('Usage %')
        
        # Psychological Development Plot
        self.plots['psychological'] = self.fig.add_subplot(gs[1, :])
        self.plots['psychological'].set_title('Psychological Development')
        self.plots['psychological'].set_ylabel('Score')
        
        # Stage Progression Plot
        self.plots['stages'] = self.fig.add_subplot(gs[2, :])
        self.plots['stages'].set_title('Developmental Stage Progression')
        self.plots['stages'].set_ylabel('Stage')
        
        plt.tight_layout()
        
    def create_plotly_dashboard(self, telemetry):
        """Create an interactive Plotly dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Emotional Development',
                'Learning Progress',
                'Psychological Metrics',
                'Memory Usage',
                'Stage Progression',
                'Social Development'
            )
        )
        
        # Emotional Development
        fig.add_trace(
            go.Scatter(y=telemetry['emotional_stability'], name='Emotional Stability'),
            row=1, col=1
        )
        
        # Learning Progress
        fig.add_trace(
            go.Scatter(y=telemetry['loss'], name='Loss'),
            row=1, col=2
        )
        
        # Psychological Metrics
        for metric in telemetry['psychological']:
            if len(telemetry['psychological'][metric]) > 0:
                fig.add_trace(
                    go.Scatter(y=telemetry['psychological'][metric], name=metric),
                    row=2, col=1
                )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(y=telemetry['memory_usage'], name='Memory Usage'),
            row=2, col=2
        )
        
        # Stage Progression
        if telemetry['stage_transitions']:
            stages = [t['stage'] for t in telemetry['stage_transitions']]
            times = [t['time'] for t in telemetry['stage_transitions']]
            fig.add_trace(
                go.Scatter(x=times, y=stages, name='Stage'),
                row=3, col=1
            )
        
        # Social Development
        fig.add_trace(
            go.Scatter(y=telemetry['social_awareness'], name='Social Awareness'),
            row=3, col=2
        )
        
        fig.update_layout(height=1000, showlegend=True, title_text="Neural Child Development Dashboard")
        return fig
    
    def update_plots(self, telemetry):
        """Update all visualization plots with new telemetry data"""
        current_time = datetime.now()
        if (current_time - self.last_update).total_seconds() < self.update_interval:
            return
            
        self.last_update = current_time
        
        # Clear all plots
        for plot in self.plots.values():
            plot.clear()
            
        # Update Emotional Development
        self.plots['emotional'].plot(telemetry['emotional_stability'])
        self.plots['emotional'].set_title('Emotional Development')
        
        # Update Learning Progress
        self.plots['learning'].plot(telemetry['loss'])
        self.plots['learning'].set_title('Learning Progress')
        
        # Update Memory Usage
        self.plots['memory'].plot(telemetry['memory_usage'])
        self.plots['memory'].set_title('Memory Usage')
        
        # Update Psychological Development
        for i, (metric, values) in enumerate(telemetry['psychological'].items()):
            if values:  # Only plot if we have data
                self.plots['psychological'].plot(values, label=metric)
        self.plots['psychological'].legend()
        
        # Update Stage Progression
        if telemetry['stage_transitions']:
            stages = [t['stage'] for t in telemetry['stage_transitions']]
            times = [t['time'] for t in telemetry['stage_transitions']]
            self.plots['stages'].plot(times, stages, 'ro-')
            
        plt.draw()
        plt.pause(0.1)
    
    def save_visualization(self, telemetry, path):
        """Save current visualization state to file"""
        # Create Plotly dashboard
        fig = self.create_plotly_dashboard(telemetry)
        
        # Save as HTML for interactive viewing
        fig.write_html(f"{path}/dashboard.html")
        
        # Save static plots
        plt.figure(figsize=(20, 15))
        self.update_plots(telemetry)
        plt.savefig(f"{path}/development_summary.png")
        
    def generate_report(self, telemetry, path):
        """Generate a comprehensive development report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'emotional_stability': np.mean(telemetry['emotional_stability']),
                'learning_progress': {
                    'mean_loss': np.mean(telemetry['loss']),
                    'loss_trend': np.polyfit(range(len(telemetry['loss'])), telemetry['loss'], 1)[0]
                },
                'memory_efficiency': np.mean(telemetry['memory_usage']),
                'social_development': np.mean(telemetry['social_awareness'])
            },
            'stage_progress': len(telemetry['stage_transitions']),
            'psychological_summary': {
                metric: np.mean(values) if values else 0
                for metric, values in telemetry['psychological'].items()
            }
        }
        
        # Save report as JSON
        import json
        with open(f"{path}/development_report.json", 'w') as f:
            json.dump(report, f, indent=4)
            
        return report 