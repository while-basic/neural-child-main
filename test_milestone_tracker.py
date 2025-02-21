# test_milestone_tracker.py
# Description: Test file for milestone tracking system
# Created by: Christopher Celaya

import unittest
from datetime import datetime
from digital_child import DigitalChild
from developmental_stages import DevelopmentalStage
from milestone_tracker import MilestoneTracker, DomainType

class TestMilestoneTracker(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.child = DigitalChild(name="Test Child", age_months=0)
        
    def test_milestone_initialization(self):
        """Test milestone tracker initialization"""
        tracker = self.child.milestone_tracker
        self.assertIsNotNone(tracker.milestones)
        self.assertEqual(len(tracker.achieved_milestones), 0)
        
    def test_development_progress(self):
        """Test development progress tracking"""
        # Simulate interaction data
        interaction_data = {
            "pattern_recognition": 0.4,
            "vocabulary_size": 6,
            "expression_level": 0.35,
            "emotional_range": 0.3,
            "social_interaction": 0.25,
            "motor_skills": 0.2,
            "visual_acuity": 0.3,
            "memory_retention": 0.2
        }
        
        # Update development
        update_result = self.child.update_development(interaction_data)
        
        # Check update result structure
        self.assertIn("curriculum_update", update_result)
        self.assertIn("milestone_updates", update_result)
        self.assertIn("development_report", update_result)
        
    def test_milestone_achievement(self):
        """Test milestone achievement tracking"""
        tracker = self.child.milestone_tracker
        
        # Print all available milestones
        print("\nAvailable milestones:")
        for m_id, milestone in tracker.milestones.items():
            print(f"{m_id}: {milestone.description} (stage: {milestone.stage}, requirements: {milestone.requirements})")
        
        # Simulate high performance in pattern recognition
        metrics = {
            "pattern_recognition": 0.5,  # Above the 0.3 requirement
            "object_permanence": 0.2
        }
        
        # Update progress
        updates = tracker.update_progress(metrics, DevelopmentalStage.NEWBORN)
        
        # Check if pattern recognition milestone was achieved
        achieved_milestones = [m.id for m in updates["new_achievements"]]
        self.assertIn("cog_1", achieved_milestones)
        
    def test_development_report(self):
        """Test development report generation"""
        # Generate initial report
        report = self.child.milestone_tracker.generate_development_report()
        
        # Check report structure
        self.assertIn("timestamp", report)
        self.assertIn("achieved_milestones", report)
        self.assertIn("total_milestones", report)
        self.assertIn("progress_by_domain", report)
        self.assertIn("recent_achievements", report)
        self.assertIn("development_trajectory", report)
        self.assertIn("intervention_suggestions", report)
        
    def test_intervention_suggestions(self):
        """Test intervention suggestion generation"""
        # Get intervention suggestions
        suggestions = self.child.get_intervention_suggestions()
        
        # Verify suggestions structure
        self.assertTrue(isinstance(suggestions, list))
        if suggestions:
            suggestion = suggestions[0]
            self.assertIn("domain", suggestion)
            self.assertIn("concern", suggestion)
            self.assertIn("suggestion", suggestion)
            
    def test_progress_persistence(self):
        """Test saving and loading progress"""
        tracker = self.child.milestone_tracker
        
        # Simulate some progress
        metrics = {
            "pattern_recognition": 0.5,
            "vocabulary_size": 10
        }
        tracker.update_progress(metrics, DevelopmentalStage.NEWBORN)
        
        # Save progress
        tracker.save_progress()
        
        # Create new tracker
        new_tracker = MilestoneTracker()
        new_tracker.load_progress()
        
        # Verify progress was restored
        self.assertEqual(
            len(tracker.achieved_milestones),
            len(new_tracker.achieved_milestones)
        )
        
    def test_domain_progress(self):
        """Test domain-specific progress tracking"""
        tracker = self.child.milestone_tracker
        
        # Get initial domain progress
        initial_progress = tracker._calculate_domain_progress()
        
        # Simulate progress in cognitive domain
        metrics = {
            "pattern_recognition": 0.5,
            "object_permanence": 0.6
        }
        tracker.update_progress(metrics, DevelopmentalStage.NEWBORN)
        
        # Get updated domain progress
        updated_progress = tracker._calculate_domain_progress()
        
        # Verify cognitive progress increased
        self.assertGreaterEqual(
            updated_progress.get(DomainType.COGNITIVE.value, 0),
            initial_progress.get(DomainType.COGNITIVE.value, 0)
        )

if __name__ == '__main__':
    unittest.main() 