# development_stages.py
# Description: Development stages for the neural child system
# Created by: Christopher Celaya

from enum import Enum, auto

class DevelopmentStage(Enum):
    NEWBORN = auto()         # 0-3 months
    INFANT = auto()          # 3-6 months
    EARLY_TODDLER = auto()   # 6-12 months
    LATE_TODDLER = auto()    # 12-18 months
    EARLY_PRESCHOOL = auto() # 18-24 months
    LATE_PRESCHOOL = auto()  # 2-3 years
    EARLY_CHILDHOOD = auto() # 3-4 years
    MIDDLE_CHILDHOOD = auto() # 4-5 years
    LATE_CHILDHOOD = auto()  # 5-6 years
    PRE_ADOLESCENT = auto()  # 6-12 years
    EARLY_TEEN = auto()      # 12-14 years
    MID_TEEN = auto()        # 14-16 years
    LATE_TEEN = auto()       # 16-18 years
    YOUNG_ADULT = auto()     # 18-21 years
    EARLY_TWENTIES = auto()  # 21-25 years
    LATE_TWENTIES = auto()   # 25-30 years

class DevelopmentMetrics:
    """Metrics for tracking development progress"""
    def __init__(self):
        self.physical = 0.0      # Physical development and motor skills
        self.cognitive = 0.0     # Cognitive development and learning
        self.social = 0.0        # Social skills and interaction
        self.emotional = 0.0     # Emotional regulation and awareness
        self.language = 0.0      # Language acquisition and communication
        self.vision = 0.0        # Visual processing and perception
        
    def update(self, metrics: dict):
        """Update development metrics"""
        for key, value in metrics.items():
            if hasattr(self, key):
                setattr(self, key, min(1.0, max(0.0, value)))
                
    def get_overall_progress(self) -> float:
        """Calculate overall development progress"""
        metrics = [self.physical, self.cognitive, self.social,
                  self.emotional, self.language, self.vision]
        return sum(metrics) / len(metrics)
        
    def to_dict(self) -> dict:
        """Convert metrics to dictionary"""
        return {
            'physical': self.physical,
            'cognitive': self.cognitive,
            'social': self.social,
            'emotional': self.emotional,
            'language': self.language,
            'vision': self.vision,
            'overall': self.get_overall_progress()
        }

class DevelopmentProfile:
    """Profile containing stage-specific development information"""
    def __init__(self, stage: DevelopmentStage):
        self.stage = stage
        self.metrics = DevelopmentMetrics()
        self.milestones = []
        self.skills = set()
        self.interests = set()
        self.social_connections = set()
        
    def add_milestone(self, milestone: str):
        """Add a developmental milestone"""
        self.milestones.append({
            'description': milestone,
            'achieved': False,
            'date_achieved': None
        })
        
    def achieve_milestone(self, milestone: str):
        """Mark a milestone as achieved"""
        for m in self.milestones:
            if m['description'] == milestone:
                m['achieved'] = True
                m['date_achieved'] = datetime.now().isoformat()
                
    def add_skill(self, skill: str):
        """Add a new skill"""
        self.skills.add(skill)
        
    def add_interest(self, interest: str):
        """Add a new interest"""
        self.interests.add(interest)
        
    def add_social_connection(self, connection: str):
        """Add a social connection"""
        self.social_connections.add(connection)
        
    def to_dict(self) -> dict:
        """Convert profile to dictionary"""
        return {
            'stage': self.stage.name,
            'metrics': self.metrics.to_dict(),
            'milestones': self.milestones,
            'skills': list(self.skills),
            'interests': list(self.interests),
            'social_connections': list(self.social_connections)
        } 