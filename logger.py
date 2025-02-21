# logger.py
# Description: Development logger for the neural child system
# Created by: Christopher Celaya

import json
from datetime import datetime
from typing import Dict, Any, Optional
import os
from pathlib import Path

class DevelopmentLogger:
    def __init__(self, log_dir: str = "logs"):
        """Initialize the development logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log files
        self.development_log = self.log_dir / "development.log"
        self.interaction_log = self.log_dir / "interactions.json"
        self.error_log = self.log_dir / "errors.log"
        self.vision_log = self.log_dir / "vision.log"
        self.training_log = self.log_dir / "training.json"
        
    def log_development(self, data: Dict[str, Any]):
        """Log development-related events"""
        self._write_log(self.development_log, {
            'timestamp': datetime.now().isoformat(),
            'type': 'development',
            'data': data
        })
        
    def log_interaction(self, interaction_data: Dict[str, Any], action: int, reward: float, state: Any) -> None:
        """Log interaction data"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'interaction_data': interaction_data,
            'action': int(action),
            'reward': float(reward),
            'state': str(state)
        }
        
        # Append to log file
        with open(self.interaction_log, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
        
    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """Log error events"""
        self._write_log(self.error_log, {
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'error': str(error),
            'context': context or {}
        })
        
    def log_vision(self, data: Dict[str, Any]):
        """Log vision processing events"""
        self._write_log(self.vision_log, {
            'timestamp': datetime.now().isoformat(),
            'type': 'vision',
            'data': data
        })
        
    def log_training(self, metrics: Dict[str, float]) -> None:
        """Log training metrics"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        # Append to log file
        with open(self.training_log, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
        
    def _write_log(self, log_file: Path, data: Dict[str, Any]):
        """Write log entry to file"""
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(data, f)
                f.write('\n')
        except Exception as e:
            print(f"Error writing to log: {str(e)}")
            
    def get_development_history(self, start_time: Optional[datetime] = None) -> list:
        """Get development history from logs"""
        history = []
        try:
            with open(self.development_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if start_time:
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            if entry_time >= start_time:
                                history.append(entry)
                        else:
                            history.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return history
        
    def get_vision_history(self, start_time: Optional[datetime] = None) -> list:
        """Get vision processing history from logs"""
        history = []
        try:
            with open(self.vision_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if start_time:
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            if entry_time >= start_time:
                                history.append(entry)
                        else:
                            history.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return history
        
    def get_error_history(self, start_time: Optional[datetime] = None) -> list:
        """Get error history from logs"""
        history = []
        try:
            with open(self.error_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if start_time:
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            if entry_time >= start_time:
                                history.append(entry)
                        else:
                            history.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return history
        
    def clear_logs(self):
        """Clear all log files"""
        log_files = [
            self.development_log,
            self.interaction_log,
            self.error_log,
            self.vision_log,
            self.training_log
        ]
        
        for log_file in log_files:
            try:
                if os.path.exists(log_file):
                    os.remove(log_file)
            except Exception as e:
                print(f"Error clearing log file {log_file}: {str(e)}")
                
    def get_latest_development_metrics(self) -> Dict[str, Any]:
        """Get the latest development metrics from logs"""
        history = self.get_development_history()
        if not history:
            return {}
            
        latest = history[-1]
        return latest.get('data', {}).get('metrics', {}) 