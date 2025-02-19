# streamlit_app.py
# Created by Christopher Celaya
# Streamlit interface for Neural Child Development System

import streamlit as st
import time
from child_model import DynamicNeuralChild
import torch
from curriculum_manager import DevelopmentalStage, DevelopmentalSystem
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(
    page_title="Neural Child Development System",
    page_icon="👶",
    layout="wide"
)

# Initialize session state
if 'child' not in st.session_state:
    # Initialize with NEWBORN stage and developmental system
    device = torch.device('cpu')  # Use CPU for Streamlit
    child = DynamicNeuralChild()
    child.to(device)  # Move model to CPU
    st.session_state.child = child
    st.session_state.interaction_count = 0

# Title and description
st.title("👶 Neural Child Development System")
st.markdown("---")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Current Stage Information
    st.subheader("Current Developmental Stage")
    current_stage = st.session_state.child.developmental_system.current_stage
    st.markdown(f"### {current_stage.name}")
    
    # Progress to Next Stage
    if current_stage.value < len(DevelopmentalStage) - 1:
        next_stage = DevelopmentalStage(current_stage.value + 1)
        st.markdown(f"Progress towards {next_stage.name}")
        
        # Get current performance metrics from drives and emotional state
        metrics = {
            'curiosity': float(st.session_state.child.drives.drives['curiosity']),
            'attention_span': float(st.session_state.child.drives.drives['mastery']),
            'emotional_stability': float(torch.mean(st.session_state.child.emotional_state)),
            'social_awareness': float(st.session_state.child.drives.drives['social_need']),
            'abstraction': float(st.session_state.child.drives.drives['autonomy']),
            'language': float(st.session_state.child.drives.drives['mastery']),
            'memory': float(st.session_state.child.drives.drives['mastery']),
            'problem_solving': float(st.session_state.child.drives.drives['mastery']),
            'motor_skills': float(st.session_state.child.drives.drives['mastery']),
            'self_awareness': float(st.session_state.child.drives.drives['autonomy'])
        }
        
        # Calculate overall progress
        performance = sum(metrics.values()) / len(metrics)  # Average of all metrics
        progress = min(performance / 0.75 * 100, 100)  # 0.75 is the threshold for advancement
        
        # Display progress bar
        st.progress(progress / 100)
        st.markdown(f"**Overall Progress**: {progress:.1f}%")

with col2:
    # Emotional State
    st.subheader("Emotional State")
    emotions = st.session_state.child.emotional_state.cpu()  # Move to CPU for display
    
    # Create gauge charts for emotions
    fig = go.Figure()
    
    # Add gauge charts for each emotion
    emotions_data = [
        ("Joy", emotions[0], "yellow"),
        ("Trust", emotions[1], "green"),
        ("Fear", emotions[2], "red"),
        ("Surprise", emotions[3], "purple")
    ]
    
    for i, (emotion, value, color) in enumerate(emotions_data):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=float(value) * 100,
            domain={'row': i//2, 'column': i%2},
            title={'text': emotion},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ]
            }
        ))
    
    fig.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Developmental Metrics
st.subheader("Developmental Metrics")
metrics_col1, metrics_col2 = st.columns(2)

with metrics_col1:
    st.markdown("### Cognitive Metrics")
    st.metric("Curiosity", f"{float(st.session_state.child.drives.drives['curiosity']):.2f}")
    st.metric("Attention Span", f"{float(st.session_state.child.drives.drives['mastery']):.2f}")
    st.metric("Problem Solving", f"{float(st.session_state.child.drives.drives['mastery']):.2f}")
    st.metric("Memory Retention", f"{float(st.session_state.child.drives.drives['mastery']):.2f}")
    st.metric("Abstraction Level", f"{float(st.session_state.child.drives.drives['autonomy']):.2f}")

with metrics_col2:
    st.markdown("### Social-Emotional Metrics")
    st.metric("Emotional Stability", f"{float(torch.mean(st.session_state.child.emotional_state.cpu())):.2f}")
    st.metric("Social Awareness", f"{float(st.session_state.child.drives.drives['social_need']):.2f}")
    st.metric("Language Complexity", f"{float(st.session_state.child.drives.drives['mastery']):.2f}")
    st.metric("Motor Skills", f"{float(st.session_state.child.drives.drives['mastery']):.2f}")
    st.metric("Self Awareness", f"{float(st.session_state.child.drives.drives['autonomy']):.2f}")

# Interaction History
st.subheader("Recent Interactions")
if hasattr(st.session_state.child, 'memory') and st.session_state.child.memory.experiences:
    for experience in st.session_state.child.memory.experiences[-5:]:
        timestamp = experience['timestamp']
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        
        with st.expander(f"Experience at {formatted_time}"):
            if 'emotional_state' in experience:
                emotional_state = experience['emotional_state'].cpu()
                st.markdown("**Emotional State:**")
                cols = st.columns(4)
                emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
                colors = ['#FFD700', '#90EE90', '#FFB6C6', '#DDA0DD']
                
                for i, (emotion, color) in enumerate(zip(emotions, colors)):
                    with cols[i]:
                        value = float(emotional_state[i])
                        st.markdown(
                            f"<div style='text-align: center; padding: 10px; background-color: {color}; border-radius: 5px;'>"
                            f"<b>{emotion}</b><br>{value:.3f}</div>",
                            unsafe_allow_html=True
                        )
            
            if 'context_influence' in experience:
                st.markdown("**Context Influence:**")
                st.progress(float(experience['context_influence']))
            
            if 'memory_influence' in experience:
                st.markdown("**Memory Influence:**")
                st.progress(float(experience['memory_influence']))

# Interaction Controls
st.markdown("---")
st.subheader("Interaction Controls")
if st.button("Start New Interaction"):
    st.session_state.interaction_count += 1
    # Trigger new interaction cycle with tensor on CPU
    mother_vector = torch.randn(4, device='cpu')  # Create tensor on CPU
    st.session_state.child.update_emotions(mother_vector)
    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

# Add requirements to requirements.txt
requirements = """
streamlit>=1.30.0
plotly>=5.14.0
"""

with open("requirements.txt", "a") as f:
    f.write(requirements) 