"""
Memory Context for Neural Child Development
Created by Christopher Celaya
"""

from datetime import datetime
from typing import Dict, Optional

class MemoryContext:
    """Context for memory retrieval operations"""
    
    def __init__(self,
                query: str,
                emotional_state: Dict[str, float],
                brain_state: Dict[str, float],
                developmental_stage: str,
                age_months: int,
                timestamp: Optional[datetime] = None):
        """Initialize memory context"""
        self.query = query
        self.emotional_state = emotional_state
        self.brain_state = brain_state
        self.developmental_stage = developmental_stage
        self.age_months = age_months
        self.timestamp = timestamp or datetime.now()
        
    def get(self, key: str, default=None):
        """Get context attribute with default value"""
        return getattr(self, key, default)
        
    def to_dict(self) -> Dict:
        """Convert context to dictionary"""
        return {
            'query': self.query,
            'emotional_state': self.emotional_state,
            'brain_state': self.brain_state,
            'developmental_stage': self.developmental_stage,
            'age_months': self.age_months,
            'timestamp': self.timestamp.isoformat()
        } 