# logger.py
# Created by Christopher Celaya
# Comprehensive logging system for neural child development

import logging
import os
from datetime import datetime
import json
from pathlib import Path
import traceback
import sys
import torch
import numpy as np

def convert_to_serializable(obj):
    """Convert PyTorch tensors and other non-serializable types to Python native types"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj

class DevelopmentLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{self.session_timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Setup different loggers for different aspects
        self.setup_loggers()
        
        # Keep track of important events
        self.important_events = []
        
    def setup_loggers(self):
        """Setup different loggers for various aspects of the system"""
        # Main development logger
        self.dev_logger = logging.getLogger('development')
        self.dev_logger.setLevel(logging.INFO)
        
        # Error logger
        self.error_logger = logging.getLogger('errors')
        self.error_logger.setLevel(logging.ERROR)
        
        # Performance logger
        self.perf_logger = logging.getLogger('performance')
        self.perf_logger.setLevel(logging.INFO)
        
        # Setup handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup handlers for each logger"""
        # Development log handler
        dev_handler = logging.FileHandler(self.session_dir / 'development.log')
        dev_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.dev_logger.addHandler(dev_handler)
        
        # Error log handler
        error_handler = logging.FileHandler(self.session_dir / 'errors.log')
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n')
        )
        self.error_logger.addHandler(error_handler)
        
        # Performance log handler
        perf_handler = logging.FileHandler(self.session_dir / 'performance.log')
        perf_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        self.perf_logger.addHandler(perf_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('\033[1m%(levelname)s\033[0m - %(message)s')
        )
        self.dev_logger.addHandler(console_handler)
        self.error_logger.addHandler(console_handler)
        
    def log_development(self, message, level=logging.INFO):
        """Log development-related messages"""
        self.dev_logger.log(level, message)
        
    def log_error(self, error, context=""):
        """Log errors with full traceback"""
        error_message = f"{context}\nError: {str(error)}\n{traceback.format_exc()}"
        self.error_logger.error(error_message)
        
    def log_performance(self, metrics):
        """Log performance metrics"""
        serializable_metrics = convert_to_serializable(metrics)
        self.perf_logger.info(json.dumps(serializable_metrics))
        
    def log_stage_transition(self, old_stage, new_stage, metrics):
        """Log developmental stage transitions"""
        serializable_metrics = convert_to_serializable(metrics)
        message = (
            f"Stage Transition: {old_stage} -> {new_stage}\n"
            f"Metrics at transition: {json.dumps(serializable_metrics, indent=2)}"
        )
        self.dev_logger.info(message)
        self.important_events.append({
            'type': 'stage_transition',
            'timestamp': datetime.now().isoformat(),
            'old_stage': old_stage,
            'new_stage': new_stage,
            'metrics': serializable_metrics
        })
        
    def log_emotional_event(self, emotion_type, intensity, context):
        """Log significant emotional events"""
        serializable_intensity = convert_to_serializable(intensity)
        message = f"Emotional Event: {emotion_type} (Intensity: {serializable_intensity})\nContext: {context}"
        self.dev_logger.info(message)
        self.important_events.append({
            'type': 'emotional_event',
            'timestamp': datetime.now().isoformat(),
            'emotion_type': emotion_type,
            'intensity': serializable_intensity,
            'context': context
        })
        
    def log_learning_milestone(self, milestone, details):
        """Log learning milestones"""
        serializable_details = convert_to_serializable(details)
        message = f"Learning Milestone: {milestone}\nDetails: {json.dumps(serializable_details, indent=2)}"
        self.dev_logger.info(message)
        self.important_events.append({
            'type': 'learning_milestone',
            'timestamp': datetime.now().isoformat(),
            'milestone': milestone,
            'details': serializable_details
        })
        
    def log_memory_consolidation(self, stats):
        """Log memory consolidation events"""
        serializable_stats = convert_to_serializable(stats)
        message = f"Memory Consolidation\nStats: {json.dumps(serializable_stats, indent=2)}"
        self.perf_logger.info(message)
        
    def log_interaction(self, stimulus, response, feedback):
        """Log mother-child interactions"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'stimulus': convert_to_serializable(stimulus),
            'response': convert_to_serializable(response),
            'feedback': convert_to_serializable(feedback)
        }
        with open(self.session_dir / 'interactions.jsonl', 'a') as f:
            f.write(json.dumps(interaction) + '\n')
            
    def save_session_summary(self):
        """Save a summary of the session"""
        summary = {
            'session_id': self.session_timestamp,
            'start_time': self.session_timestamp,
            'end_time': datetime.now().isoformat(),
            'important_events': convert_to_serializable(self.important_events)
        }
        
        with open(self.session_dir / 'session_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
    def cleanup(self):
        """Cleanup and close all handlers"""
        self.save_session_summary()
        for logger in [self.dev_logger, self.error_logger, self.perf_logger]:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler) 