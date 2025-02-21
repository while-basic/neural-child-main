# milestone_demo.py
# Description: Demonstration of the milestone tracking system
# Created by: Christopher Celaya

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from digital_child import DigitalChild
from developmental_stages import DevelopmentalStage

def print_milestone_status(status):
    """Print milestone status in a formatted way"""
    print("\n=== Development Status ===")
    print(f"Name: {status['name']}")
    print(f"Age: {status['age_months']} months")
    print(f"Stage: {status['development_stage']}")
    print("\nMetrics:")
    for metric, value in status['metrics'].items():
        print(f"- {metric}: {value:.2f}")
    
    report = status['development_report']
    print(f"\nAchieved Milestones: {report['achieved_milestones']} / {report['total_milestones']}")
    
    print("\nProgress by Domain:")
    for domain, progress in report['progress_by_domain'].items():
        print(f"- {domain}: {progress:.2%}")
        
    if report['recent_achievements']:
        print("\nRecent Achievements:")
        for achievement in report['recent_achievements']:
            print(f"- {achievement['description']} ({achievement['id']})")

def main():
    # Create a new digital child
    child = DigitalChild(name="Test Child", age_months=0)
    
    # Print initial status
    print("\n=== Initial Status ===")
    status = child.get_development_status()
    print_milestone_status(status)
    
    # Simulate some development in pattern recognition
    print("\n=== Simulating Pattern Recognition Development ===")
    interaction_data = {
        "pattern_recognition": 0.5,  # Above the 0.3 requirement for cog_1
        "vocabulary_size": 3,
        "emotional_range": 0.25,
        "social_interaction": 0.3
    }
    
    # Update development
    update_result = child.update_development(interaction_data)
    
    # Print results
    if update_result["milestone_updates"]["new_achievements"]:
        print("\nNew Achievements:")
        for milestone in update_result["milestone_updates"]["new_achievements"]:
            print(f"- {milestone.description} ({milestone.id})")
    
    if update_result["milestone_updates"]["in_progress"]:
        print("\nMilestones in Progress:")
        for milestone in update_result["milestone_updates"]["in_progress"]:
            print(f"- {milestone.description} ({milestone.id})")
    
    # Print updated status
    print("\n=== Updated Status ===")
    status = child.get_development_status()
    print_milestone_status(status)
    
    # Get intervention suggestions
    suggestions = child.get_intervention_suggestions()
    if suggestions:
        print("\n=== Intervention Suggestions ===")
        for suggestion in suggestions:
            print(f"\nDomain: {suggestion['domain']}")
            print(f"Concern: {suggestion['concern']}")
            print(f"Suggestion: {suggestion['suggestion']}")

if __name__ == "__main__":
    main() 