# logger.py
# Description: Logging system for neural child development
# Created by: Christopher Celaya

import logging
import os
from datetime import datetime
from typing import Dict, Any

class DevelopmentLogger:
    def __init__(self, log_dir: str = "logs"):
        """Initialize the development logger."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file handler
        log_file = os.path.join(log_dir, f"development_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters and add them to the handlers
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Get the logger
        self.logger = logging.getLogger('NeuralChild')
        self.logger.setLevel(logging.INFO)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics tracking
        self.metrics = {
            'emotional_states': [],
            'language_progress': [],
            'memory_stats': [],
            'interactions': []
        }
    
    def log_emotional_state(self, state: Dict[str, float]):
        """Log emotional state changes."""
        self.metrics['emotional_states'].append({
            'timestamp': datetime.now().isoformat(),
            'state': state
        })
        self.logger.info(f"Emotional State: {state}")
    
    def log_language_progress(self, progress: Dict[str, Any]):
        """Log language development progress."""
        self.metrics['language_progress'].append({
            'timestamp': datetime.now().isoformat(),
            'progress': progress
        })
        self.logger.info(f"Language Progress: {progress}")
    
    def log_memory_stats(self, stats: Dict[str, Any]):
        """Log memory system statistics."""
        self.metrics['memory_stats'].append({
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        })
        self.logger.info(f"Memory Stats: {stats}")
    
    def log_interaction(self, interaction: Dict[str, Any]):
        """Log an interaction with the child."""
        self.metrics['interactions'].append({
            'timestamp': datetime.now().isoformat(),
            'interaction': interaction
        })
        self.logger.info(f"Interaction: {interaction}")
    
    def log_error(self, error: str, context: Dict[str, Any] = None):
        """Log an error with optional context."""
        if context:
            self.logger.error(f"Error: {error} - Context: {context}")
        else:
            self.logger.error(f"Error: {error}")
    
    def log_warning(self, warning: str, context: Dict[str, Any] = None):
        """Log a warning with optional context."""
        if context:
            self.logger.warning(f"Warning: {warning} - Context: {context}")
        else:
            self.logger.warning(f"Warning: {warning}")
    
    def log_milestone(self, milestone: str, details: Dict[str, Any] = None):
        """Log a developmental milestone."""
        if details:
            self.logger.info(f"Milestone Reached - {milestone}: {details}")
        else:
            self.logger.info(f"Milestone Reached: {milestone}")
    
    def get_metrics(self) -> Dict[str, list]:
        """Get all tracked metrics."""
        return self.metrics
    
    def save_metrics(self, filepath: str):
        """Save metrics to a file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_metrics(self, filepath: str):
        """Load metrics from a file."""
        import json
        with open(filepath, 'r') as f:
            self.metrics = json.load(f) 