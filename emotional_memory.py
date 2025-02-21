"""
emotional_memory.py - Defines the EmotionalMemoryEntry class for managing emotional memories
Made by Christopher Celaya
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class EmotionalMemoryEntry:
    """Class representing an emotional memory entry with associated metadata."""
    
    content: str
    emotional_state: Dict[str, float]
    context: str
    intensity: float
    valence: float
    arousal: float
    timestamp: datetime
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate and process the emotional memory entry after initialization."""
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        
        # Ensure emotional state values are within valid range [0.0, 1.0]
        for emotion, value in self.emotional_state.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Emotional state value for {emotion} must be between 0.0 and 1.0")
        
        # Validate intensity, valence, and arousal ranges
        for value_name, value in [("intensity", self.intensity), 
                                ("valence", self.valence), 
                                ("arousal", self.arousal)]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{value_name} must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory entry to a dictionary format."""
        return {
            "content": self.content,
            "emotional_state": self.emotional_state,
            "context": self.context,
            "intensity": self.intensity,
            "valence": self.valence,
            "arousal": self.arousal,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalMemoryEntry':
        """Create an EmotionalMemoryEntry instance from a dictionary."""
        return cls(
            content=data["content"],
            emotional_state=data["emotional_state"],
            context=data["context"],
            intensity=data["intensity"],
            valence=data["valence"],
            arousal=data["arousal"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data["metadata"]
        )

    def get_emotional_intensity(self) -> float:
        """Calculate the overall emotional intensity of the memory."""
        return sum(self.emotional_state.values()) / len(self.emotional_state)

    def is_significant(self, threshold: float = 0.7) -> bool:
        """Determine if the memory is emotionally significant."""
        return (self.intensity >= threshold or 
                any(value >= threshold for value in self.emotional_state.values())) 