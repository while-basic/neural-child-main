# development_logger.py
# Description: Logger for tracking child development progress
# Created by: Christopher Celaya

import os
import json
from datetime import datetime
from typing import Dict, Any

class DevelopmentLogger:
    def __init__(self, log_dir: str = "logs"):
        """Initialize development logger"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"development_log_{timestamp}.json")
        self.log_entries = []
        
    def log_vision_development(self, stage: Any, metrics: Dict[str, float]):
        """Log vision development progress"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'vision_development',
            'stage': stage.name if hasattr(stage, 'name') else str(stage),
            'metrics': metrics
        }
        self.log_entries.append(entry)
        self._save_log()
        
    def log_error(self, error_msg: str, context: Dict[str, Any] = None):
        """Log error messages with context"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'message': error_msg,
            'context': context or {}
        }
        self.log_entries.append(entry)
        self._save_log()
        
    def _save_log(self):
        """Save log entries to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_entries, f, indent=2)
        except Exception as e:
            print(f"Error saving log: {str(e)}")
            
    def get_latest_logs(self, n: int = 10) -> list:
        """Get the latest n log entries"""
        return self.log_entries[-n:] 