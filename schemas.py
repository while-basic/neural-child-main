from pydantic import BaseModel, Field
from typing import Optional

class MotherResponse(BaseModel):
    """Structured response schema for mother-child interactions"""
    content: str = Field(default="", description="The response message")
    joy: float = Field(default=0.5, ge=0.0, le=1.0, description="Joy emotion level")
    trust: float = Field(default=0.5, ge=0.0, le=1.0, description="Trust emotion level")
    fear: float = Field(default=0.2, ge=0.0, le=1.0, description="Fear emotion level")
    surprise: float = Field(default=0.3, ge=0.0, le=1.0, description="Surprise emotion level")
    reward_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Reward score for the interaction")
    success_metric: float = Field(default=0.5, ge=0.0, le=1.0, description="Success metric for the development stage")
    complexity_rating: float = Field(default=0.3, ge=0.0, le=1.0, description="Complexity rating of the interaction")
    self_critique_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Self-awareness score")
    cognitive_labels: Optional[list] = []
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "content": "That's a good attempt! [HUG]",
                "joy": 0.8,
                "trust": 0.6,
                "fear": 0.05,
                "surprise": 0.1,
                "reward_score": 0.85,
                "success_metric": 0.7,
                "complexity_rating": 0.4,
                "self_critique_score": 0.3,
                "cognitive_labels": ["encouragement", "basic_concept"]
            }]
        }
    }
