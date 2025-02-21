# dream_system.py
# Description: Dream simulation system for neural child development
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import random
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from dream_generator import DreamGenerator

@dataclass
class DreamContent:
    """Represents the content of a dream"""
    narrative: str
    emotional_intensity: float
    primary_emotion: str
    secondary_emotions: Dict[str, float]
    source_memories: List[str]
    dream_symbols: List[str]
    timestamp: datetime
    duration_minutes: float

class DreamType(Enum):
    """Types of dreams that can be experienced"""
    PROCESSING = auto()  # Processing daily experiences
    EMOTIONAL = auto()   # Processing emotional events
    CREATIVE = auto()    # Creative problem-solving
    DEVELOPMENTAL = auto()  # Related to developmental stage
    SYMBOLIC = auto()    # Symbolic/abstract processing
    LUCID = auto()      # Self-aware dreaming

class DreamSystem:
    """Neural dream simulation system"""
    def __init__(self, 
                brain,
                obsidian_api,
                sentence_transformer_model: str = "all-MiniLM-L6-v2"):
        """Initialize the dream system"""
        self.brain = brain
        self.obsidian_api = obsidian_api
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model)
        self.dream_generator = DreamGenerator(sentence_transformer_model)
        
        # Create dream folder in Obsidian if it doesn't exist
        self.dream_folder = Path(obsidian_api.vault_path) / "Dreams"
        self.dream_folder.mkdir(exist_ok=True)
        
        # Initialize dream state
        self.is_dreaming = False
        self.current_dream = None
        self.dream_history = []
        
        # Q-Learning parameters for dream generation
        self.q_table = {}  # State-action pairs for dream decisions
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
        # Dream generation parameters
        self.min_dream_duration = 5  # minutes
        self.max_dream_duration = 45  # minutes
        
        # Symbol dictionary for dream interpretation
        self.dream_symbols = self.dream_generator.dream_symbols
    
    def generate_dream(self) -> DreamContent:
        """Generate a dream based on recent memories and emotional state"""
        try:
            # Get recent emotional memories
            recent_memories = []
            results = self.brain.emotional_collection.get()
            print("\nChromaDB results:", results)
            if results and 'documents' in results:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    recent_memories.append({
                        'content': doc,
                        'id': metadata['id'],
                        'emotional_intensity': metadata.get('emotional_intensity', 0.5),
                        'timestamp': metadata.get('timestamp', datetime.now().isoformat())
                    })
            print("Recent memories:", recent_memories)
            
            # Select dream type using Q-Learning
            current_state = self._get_current_state()
            dream_type = self._select_dream_type(current_state)
            print("Selected dream type:", dream_type)
            
            # Generate dream narrative based on type
            if dream_type == DreamType.PROCESSING:
                narrative, sources = self._generate_processing_dream(recent_memories)
            elif dream_type == DreamType.EMOTIONAL:
                narrative, sources = self._generate_emotional_dream(recent_memories)
            elif dream_type == DreamType.CREATIVE:
                narrative, sources = self._generate_creative_dream(recent_memories)
            elif dream_type == DreamType.DEVELOPMENTAL:
                narrative, sources = self._generate_developmental_dream()
            elif dream_type == DreamType.SYMBOLIC:
                narrative, sources = self._generate_symbolic_dream(recent_memories)
            else:  # LUCID
                narrative, sources = self._generate_lucid_dream()
            print("Generated narrative:", narrative)
            print("Sources:", sources)
            
            # Calculate emotional components
            emotional_intensity = random.uniform(0.3, 1.0)
            primary_emotion = self._determine_primary_emotion(narrative)
            print("Primary emotion:", primary_emotion)
            secondary_emotions = self._analyze_emotional_content(narrative)
            print("Secondary emotions:", secondary_emotions)
            
            # Extract dream symbols
            dream_symbols = self._extract_dream_symbols(narrative)
            print("Dream symbols:", dream_symbols)
            
            # Create dream content
            dream = DreamContent(
                narrative=narrative,
                emotional_intensity=emotional_intensity,
                primary_emotion=primary_emotion,
                secondary_emotions=secondary_emotions,
                source_memories=sources,
                dream_symbols=dream_symbols,
                timestamp=datetime.now(),
                duration_minutes=random.uniform(self.min_dream_duration, self.max_dream_duration)
            )
            
            # Store dream in Obsidian
            self._store_dream_in_obsidian(dream, dream_type)
            
            # Update Q-Learning
            reward = self._calculate_dream_reward(dream)
            self._update_q_table(current_state, dream_type, reward)
            
            return dream
            
        except Exception as e:
            print(f"Error generating dream: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _get_current_state(self) -> str:
        """Get current state for Q-Learning"""
        brain_state = self.brain.get_brain_state()
        
        # Handle different brain state formats
        emotional_valence = 0.5  # Default values
        arousal = 0.5
        
        # Try to extract values from different possible structures
        if 'emotion_state' in brain_state:
            emotional_valence = brain_state['emotion_state'].get('emotional_valence', 0.5)
            arousal = brain_state['emotion_state'].get('arousal', 0.5)
        elif 'emotional_valence' in brain_state:
            emotional_valence = brain_state.get('emotional_valence', 0.5)
            arousal = brain_state.get('arousal', 0.5)
        
        # Discretize continuous values for state space
        emotional_state = "high" if emotional_valence > 0.5 else "low"
        arousal_state = "high" if arousal > 0.5 else "low"
        development_stage = brain_state.get('development_stage', 'NEWBORN')
        
        return f"{emotional_state}_{arousal_state}_{development_stage}"
        
    def _select_dream_type(self, state: str) -> DreamType:
        """Select dream type using epsilon-greedy Q-Learning"""
        if state not in self.q_table:
            self.q_table[state] = {dream_type: 0.0 for dream_type in DreamType}
            
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            return random.choice(list(DreamType))
        else:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
            
    def _update_q_table(self, state: str, action: DreamType, reward: float):
        """Update Q-table based on reward"""
        if state not in self.q_table:
            self.q_table[state] = {dream_type: 0.0 for dream_type in DreamType}
            
        # Q-Learning update
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[state].values())
        
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
                   
        self.q_table[state][action] = new_value
        
    def _calculate_dream_reward(self, dream: DreamContent) -> float:
        """Calculate reward for dream generation"""
        # Base reward
        reward = 0.5
        
        # Reward for emotional processing
        if dream.emotional_intensity > 0.7:
            reward += 0.2
            
        # Reward for incorporating multiple memories
        if len(dream.source_memories) > 2:
            reward += 0.1
            
        # Reward for symbolic content
        if len(dream.dream_symbols) > 3:
            reward += 0.1
            
        # Reward for developmental relevance
        if any(symbol in ['learning', 'growth', 'development'] for symbol in dream.dream_symbols):
            reward += 0.1
            
        return min(1.0, reward)
        
    def _generate_dream_title(self, dream: DreamContent, dream_type: DreamType) -> Tuple[str, str]:
        """Generate a meaningful title for the dream using Ollama/Llama3 if available, otherwise use fallback method.
        Returns a tuple of (display_title, filename_title)"""
        try:
            # Check if Ollama is available
            test_cmd = 'ollama -v'
            version_check = self.obsidian_api.run_command(test_cmd)
            
            if version_check and not version_check.startswith('Error'):
                # Ollama is available, use it for title generation
                prompt = f"""You are a dream title generator. Create a short, poetic title (2-5 words) for this dream.
The title should reflect the emotional content and symbolism of the dream.
Do not include any explanations or additional text.
Only output the title itself.

Dream Content: {dream.narrative}
Primary Emotion: {dream.primary_emotion} (Intensity: {dream.emotional_intensity:.2f})
Dream Type: {dream_type.name}
Dream Symbols: {', '.join(dream.dream_symbols) if dream.dream_symbols else 'None'}

Title:"""

                # Call Ollama with strict formatting
                cmd = f'ollama run llama3 "{prompt}" --format json'
                result = self.obsidian_api.run_command(cmd)
                
                if result and not result.startswith('Error'):
                    try:
                        # Try to parse JSON response
                        import json
                        response = json.loads(result)
                        title = response.get('response', '').strip()
                    except:
                        # If JSON parsing fails, just take the raw response
                        title = result.strip()
                    
                    # Clean and validate the title
                    title = title.split('\n')[0].strip()
                    title = ' '.join(title.split()[:5])
                    title = ''.join(c for c in title if c.isalnum() or c.isspace())
                    
                    if title and len(title.strip()) >= 2:
                        # Create display and filename versions
                        display_title = title
                        filename_title = title.replace(' ', '_')
                        return display_title, filename_title
            
            # If Ollama is not available or failed, use fallback method
            return self._generate_fallback_title(dream, dream_type)
            
        except Exception as e:
            print(f"Error in title generation: {str(e)}")
            return self._generate_fallback_title(dream, dream_type)
            
    def _generate_fallback_title(self, dream: DreamContent, dream_type: DreamType) -> Tuple[str, str]:
        """Generate a title without using Ollama. Returns (display_title, filename_title)"""
        title_parts = []
        
        # Add dream type prefix for special types
        if dream_type != DreamType.PROCESSING:
            title_parts.append(dream_type.name.title())
        
        # Add emotional state if intense
        if dream.emotional_intensity > 0.7:
            emotion_phrase = {
                'joy': 'Joyful',
                'fear': 'Fearful',
                'surprise': 'Surprising',
                'anger': 'Angry',
                'sadness': 'Sorrowful',
                'disgust': 'Unsettling',
                'neutral': 'Mysterious'
            }.get(dream.primary_emotion.lower(), dream.primary_emotion.title())
            title_parts.append(emotion_phrase)
            
        # Add dream symbols if present
        if dream.dream_symbols:
            title_parts.extend([symbol.title() for symbol in dream.dream_symbols[:1]])
            
        # Add significant words from narrative
        words = dream.narrative.split()
        significant_words = [word.title() for word in words 
                           if len(word) > 3 and word.lower() not in 
                           ['with', 'and', 'the', 'was', 'were', 'that', 'this', 'then', 'during']][:2]
        title_parts.extend(significant_words)
        
        # Add "Dream" if title is too short
        if len(title_parts) < 2:
            title_parts.append('Dream')
            
        # Create display and filename versions
        display_title = ' '.join(title_parts)
        filename_title = '_'.join(title_parts)
        filename_title = ''.join(c for c in filename_title if c.isalnum() or c == '_')
        
        return display_title, filename_title
        
    def _store_dream_in_obsidian(self, dream: DreamContent, dream_type: DreamType):
        """Store dream in Obsidian with proper formatting"""
        # Generate dream title
        display_title, filename_title = self._generate_dream_title(dream, dream_type)
        
        # Create filename
        timestamp = dream.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_title}_{timestamp}.md"

        # Get brain state for metadata
        brain_state = self.brain.get_brain_state()
        stage = brain_state.get('development_stage', 'UNKNOWN')
        
        # Convert stage to string
        if isinstance(stage, int):
            stage_name = str(stage)
        elif hasattr(stage, 'name'):
            stage_name = stage.name
        else:
            stage_name = str(stage)
        
        # Format emotional states with progress bars
        def format_emotion_bar(value: float) -> str:
            bar_length = int(value * 10)
            bar_char = '█' if value > 0.8 else '▆' if value > 0.6 else '▄' if value > 0.4 else '▂' if value > 0.2 else '-'
            return f"{bar_char * bar_length}{'-' * (10 - bar_length)} {value:.2f}"
            
        # Format emotional states
        emotional_states = {}
        for emotion, value in sorted(dream.secondary_emotions.items(), key=lambda x: x[1], reverse=True):
            emotional_states[emotion] = format_emotion_bar(value)

        # Get source memory details
        source_memory_details = []
        for memory_id in dream.source_memories:
            results = self.brain.emotional_collection.get(ids=[memory_id])
            if results and 'documents' in results and len(results['documents']) > 0:
                content = results['documents'][0]
                metadata = results['metadatas'][0]
                source_memory_details.append(f"- {content} (Intensity: {metadata.get('emotional_intensity', 0.0):.2f})")

        # Get heartbeat info
        heartbeat = self.brain.heartbeat.get_current_heartbeat()
        heartbeat_state = heartbeat.get('state', 'UNKNOWN')
        heartbeat_rate = heartbeat.get('rate', 80)

        content = f"""# {display_title}

## Content
*dreaming softly* {dream.narrative}

## Type
{dream_type.name}

## Emotional Content
Primary Emotion: {dream.primary_emotion}
Emotional Intensity: {format_emotion_bar(dream.emotional_intensity)}

### Secondary Emotions
{chr(10).join([f"{emotion:<12}: {bar}" for emotion, bar in emotional_states.items()])}

## Dream Symbols
{", ".join(dream.dream_symbols) if dream.dream_symbols else "No significant symbols"}

## Source Memories
{chr(10).join(source_memory_details) if source_memory_details else "No direct memory sources"}

## Duration
{dream.duration_minutes:.1f} minutes

## Analysis
{self._analyze_dream(dream)}

## Physiological State
- Heartbeat: {heartbeat_rate} BPM ({heartbeat_state})
- Consciousness Level: {brain_state.get('consciousness_level', 0.5):.2f}
- Emotional Valence: {brain_state.get('emotion_state', {}).get('emotional_valence', 0.0):.2f}
- Arousal Level: {brain_state.get('emotion_state', {}).get('arousal', 0.0):.2f}

## Metadata
- Age: {brain_state.get('age_months', 0)} months
- Stage: {stage_name}
- Dream Type: {dream_type.name}
- Timestamp: {dream.timestamp.strftime("%Y-%m-%d %H:%M:%S")}

## Links
[[Dreams/Index]]
[[Emotional/State]]
[[Development/Current]]
[[Memories/Recent]]
[[Dreams/{dream_type.name}]]
[[Emotions/{dream.primary_emotion}]]

## Tags
#dream #{dream_type.name.lower()} #emotional-intensity-{int(dream.emotional_intensity * 10)} #{dream.primary_emotion.lower()} #stage-{stage_name.replace(' ', '-').lower()}"""

        # Save to Obsidian vault with UTF-8 encoding
        dream_path = self.dream_folder / filename
        with open(dream_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
    def _analyze_dream(self, dream: DreamContent) -> str:
        """Analyze dream content and symbolism"""
        analysis = []
        
        # Analyze emotional patterns
        if dream.emotional_intensity > 0.7:
            analysis.append("This dream shows intense emotional processing.")
            
        # Get dominant emotions
        sorted_emotions = sorted(dream.secondary_emotions.items(), key=lambda x: x[1], reverse=True)
        dominant_emotions = [emotion for emotion, value in sorted_emotions if value > 0.2]
        if dominant_emotions:
            analysis.append(f"The dream is dominated by {', '.join(dominant_emotions)} emotions.")
            
        # Analyze symbols
        for symbol in dream.dream_symbols:
            if symbol in self.dream_symbols:
                meanings = self.dream_symbols[symbol]
                analysis.append(f"The symbol '{symbol}' suggests themes of: {', '.join(meanings)}")
            elif any(symbol in color_meanings for color_meanings in self.dream_symbols['colors'].items()):
                color_meaning = self.dream_symbols['colors'][symbol]
                analysis.append(f"The color '{symbol}' represents: {', '.join(color_meaning)}")
                
        # Analyze developmental relevance
        brain_state = self.brain.get_brain_state()
        stage = brain_state.get('development_stage', 'UNKNOWN')
        stage_name = stage.name if hasattr(stage, 'name') else stage
        analysis.append(f"This dream reflects the current developmental stage ({stage_name}).")
        
        # Analyze memory integration
        if len(dream.source_memories) > 1:
            analysis.append("The dream integrates multiple memories, showing active memory consolidation.")
            
        # Analyze emotional processing
        if dream.primary_emotion != sorted_emotions[0][0]:
            analysis.append(f"While the primary emotion is {dream.primary_emotion}, there's a strong undercurrent of {sorted_emotions[0][0]} ({sorted_emotions[0][1]:.2f}).")
            
        return "\n\n".join(analysis)
        
    def _generate_processing_dream(self, recent_memories: List[Dict]) -> Tuple[str, List[str]]:
        """Generate a dream processing recent experiences"""
        return self.dream_generator.generate_processing_dream(recent_memories)
        
    def _generate_emotional_dream(self, recent_memories: List[Dict]) -> Tuple[str, List[str]]:
        """Generate a dream processing emotional experiences"""
        return self.dream_generator.generate_emotional_dream(recent_memories)
        
    def _generate_creative_dream(self, recent_memories: List[Dict]) -> Tuple[str, List[str]]:
        """Generate a creative problem-solving dream"""
        return self.dream_generator.generate_creative_dream(recent_memories)
        
    def _generate_developmental_dream(self) -> Tuple[str, List[str]]:
        """Generate a dream related to current developmental stage"""
        brain_state = self.brain.get_brain_state()
        return self.dream_generator.generate_developmental_dream(brain_state['development_stage'])
        
    def _generate_symbolic_dream(self, recent_memories: List[Dict]) -> Tuple[str, List[str]]:
        """Generate a symbolic/abstract dream"""
        return self.dream_generator.generate_symbolic_dream(recent_memories)
        
    def _generate_lucid_dream(self) -> Tuple[str, List[str]]:
        """Generate a lucid dream"""
        return self.dream_generator.generate_lucid_dream()
        
    def _determine_primary_emotion(self, narrative: str) -> str:
        """Determine the primary emotion in a dream narrative"""
        # Use the emotion classifier from dream generator
        emotions = self.dream_generator._extract_emotions([narrative])
        return emotions[0] if emotions else "neutral"
        
    def _analyze_emotional_content(self, narrative: str) -> Dict[str, float]:
        """Analyze the emotional content of a dream narrative"""
        # Get emotion predictions
        results = self.dream_generator.emotion_classifier(narrative)
        
        # The classifier returns a list with a single dictionary containing scores
        emotions = {}
        for emotion_data in results[0]:  # First element contains all emotion scores
            emotions[emotion_data['label']] = emotion_data['score']
            
        return emotions
        
    def _extract_dream_symbols(self, narrative: str) -> List[str]:
        """Extract symbolic elements from dream narrative"""
        symbols = []
        
        # Check for each symbol type in the narrative
        for symbol_type, symbol_list in self.dream_symbols.items():
            if symbol_type == 'colors':
                # Check color symbols
                for color in symbol_list:
                    if color in narrative.lower():
                        symbols.append(color)
            else:
                # Check regular symbols
                for symbol in symbol_list:
                    if symbol in narrative.lower():
                        symbols.append(symbol)
                        
        return list(set(symbols))  # Remove duplicates 