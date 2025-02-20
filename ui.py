# ui.py
# Description: Gradio interface for the neural child development system
# Created by: Christopher Celaya

import gradio as gr
import torch
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import logging
import sys
import signal

from main import DigitalChild, DevelopmentalStage, LanguageStage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeuralChildUI:
    def __init__(self):
        self.child = DigitalChild()
        self.conversation_history = []
        
    def get_development_status(self):
        """Get current development status as formatted text"""
        try:
            age = self.child.age()
            stage = self.child.curriculum.current_stage.name
            progress = self.child.curriculum.stage_progress * 100
            
            status = f"""
            ## Development Status
            - Age: {age} months
            - Stage: {stage}
            - Progress: {progress:.1f}%
            
            ### Metrics:
            """
            
            for metric, value in self.child.metrics.items():
                status += f"- {metric.replace('_', ' ').title()}: {value:.2f}\n"
                
            return status
        except Exception as e:
            logger.error(f"Error in get_development_status: {str(e)}")
            return "Error: Could not fetch development status"
        
    def plot_emotional_state(self):
        """Create radar plot of emotional state"""
        try:
            emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
            values = self.child.emotional_state.cpu().numpy().tolist()
            values.append(values[0])  # Close the polygon
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=emotions + [emotions[0]],
                fill='toself',
                name='Emotional State'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Emotional State"
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error in plot_emotional_state: {str(e)}")
            return None
        
    def plot_network_structure(self):
        """Create network visualization"""
        try:
            network_data = self.child.get_network_structure()
            G = nx.DiGraph()
            
            # Add brain network nodes
            G.add_node("Input", layer="input", size=network_data['brain']['layers']['input_size'])
            for i, size in enumerate(network_data['brain']['layers']['hidden_sizes']):
                G.add_node(f"H{i}", layer="hidden", size=size)
                if i == 0:
                    G.add_edge("Input", f"H{i}")
                else:
                    G.add_edge(f"H{i-1}", f"H{i}")
                    
            # Create visualization
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=1000, arrowsize=20, font_size=10)
            
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64 string
            img_str = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error in plot_network_structure: {str(e)}")
            return None
        
    def update_conversation(self, user_input):
        """Process user input and update conversation"""
        try:
            if not user_input:
                return ""
                
            # Get child's emotional response
            emotion_change = torch.randn(4) * 0.1  # Random emotional change
            self.child.update_emotions(self.child.emotional_state + emotion_change)
            
            # Process through language development
            response = self.child.language.process_input(user_input)
            
            # Add to conversation history
            self.conversation_history.append({
                "user": user_input,
                "child": response,
                "emotion": self.child.express_feeling()
            })
            
            # Format conversation history
            formatted_history = ""
            for entry in self.conversation_history[-5:]:  # Show last 5 exchanges
                formatted_history += f"You: {entry['user']}\n"
                formatted_history += f"Child [{entry['emotion']}]: {entry['child']}\n\n"
                
            return formatted_history
        except Exception as e:
            logger.error(f"Error in update_conversation: {str(e)}")
            return "Error: Could not process conversation"
        
    def update_development(self, stage_name, progress):
        """Update development stage and progress"""
        try:
            if stage_name and isinstance(stage_name, str):
                stage = DevelopmentalStage[stage_name]
                self.child.set_stage(stage, float(progress))
                return self.get_development_status()
            return "Error: Invalid stage selection"
        except Exception as e:
            logger.error(f"Error in update_development: {str(e)}")
            return f"Error: Could not update development stage - {str(e)}"
        
    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info("Shutting down gracefully...")
        sys.exit(0)
        
    def launch_ui(self):
        """Launch the Gradio interface"""
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        try:
            with gr.Blocks(
                title="Neural Child Development System",
                theme=gr.themes.Soft(),  # Use a more modern theme
                css=".gradio-container {max-width: 1200px; margin: auto;}"  # Custom CSS
            ) as interface:
                gr.Markdown("# Neural Child Development System")
                
                with gr.Tab("Development Status"):
                    status_output = gr.Markdown()
                    refresh_btn = gr.Button("Refresh Status", variant="primary")
                    refresh_btn.click(self.get_development_status, outputs=status_output)
                    # Initialize status
                    status_output.value = self.get_development_status()
                    
                with gr.Tab("Emotional State"):
                    emotion_plot = gr.Plot()
                    emotion_btn = gr.Button("Update Emotional State", variant="primary")
                    emotion_btn.click(self.plot_emotional_state, outputs=emotion_plot)
                    # Initialize plot
                    emotion_plot.value = self.plot_emotional_state()
                    
                with gr.Tab("Conversation"):
                    chat_output = gr.Markdown()
                    with gr.Row():
                        txt_input = gr.Textbox(
                            label="Talk to the child",
                            placeholder="Type your message here...",
                            lines=2
                        )
                        send_btn = gr.Button("Send", variant="primary")
                    
                    # Handle conversation updates
                    txt_input.submit(self.update_conversation, inputs=txt_input, outputs=chat_output)
                    send_btn.click(self.update_conversation, inputs=txt_input, outputs=chat_output)
                    
                with gr.Tab("Neural Network"):
                    network_img = gr.Image(label="Neural Network Visualization")
                    network_btn = gr.Button("Visualize Network", variant="primary")
                    network_btn.click(self.plot_network_structure, outputs=network_img)
                    # Initialize visualization
                    network_img.value = self.plot_network_structure()
                    
                with gr.Tab("Development Controls"):
                    with gr.Row():
                        stage_dropdown = gr.Dropdown(
                            choices=[stage.name for stage in DevelopmentalStage],
                            label="Set Development Stage",
                            value=DevelopmentalStage.NEWBORN.name,
                            info="Select the developmental stage"
                        )
                        progress_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.0,
                            label="Stage Progress",
                            info="Set progress within the current stage (0-100%)"
                        )
                    update_btn = gr.Button("Update Development", variant="primary")
                    update_status = gr.Markdown()
                    
                    update_btn.click(
                        self.update_development,
                        inputs=[stage_dropdown, progress_slider],
                        outputs=update_status
                    )
                
            # Launch with specific server settings
            interface.launch(
                server_name="127.0.0.1",  # Localhost only
                server_port=7860,  # Default Gradio port
                share=False,  # Disable sharing for security
                debug=True,  # Enable debug mode
                show_error=True,  # Show detailed error messages
                enable_queue=True,  # Enable request queuing
                max_threads=4  # Limit concurrent threads
            )
            
        except Exception as e:
            logger.error(f"Error launching UI: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        ui = NeuralChildUI()
        ui.launch_ui()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1) 