# dream_generator.py
# Description: Dream generation methods for the dream simulation system
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from datetime import datetime
import random
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class DreamGenerator:
    """Handles the generation of different types of dreams"""
    def __init__(self, sentence_transformer_model: str = "all-MiniLM-L6-v2"):
        """Initialize the dream generator"""
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model)
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None  # Return all scores
        )
        
        # Map emotion labels to our simplified set
        self.emotion_map = {
            'joy': ['joy', 'happy', 'excited', 'optimistic'],
            'trust': ['trust', 'acceptance', 'admiration'],
            'fear': ['fear', 'scared', 'anxious', 'worried'],
            'surprise': ['surprise', 'amazed', 'astonished'],
            'sadness': ['sadness', 'sad', 'depressed', 'grief'],
            'anger': ['anger', 'angry', 'mad', 'furious'],
            'anticipation': ['anticipation', 'expectant', 'hopeful'],
            'disgust': ['disgust', 'disgusted', 'repulsed'],
            'neutral': ['neutral', 'calm', 'balanced']
        }
        
        # Elements for dream construction
        self.dream_elements = {
            'locations': [
                'in a familiar house', 'in a mysterious garden', 'at school', 
                'in a forest', 'by the ocean', 'in a city', 'on a mountain',
                'in space', 'underwater', 'in a cave', 'in the clouds'
            ],
            'actions': [
                'walking', 'running', 'flying', 'swimming', 'climbing',
                'searching', 'building', 'creating', 'learning', 'teaching',
                'exploring', 'discovering', 'transforming', 'growing'
            ],
            'emotions': [
                'joy', 'wonder', 'excitement', 'peace', 'curiosity',
                'confidence', 'hope', 'love', 'courage', 'determination'
            ],
            'symbols': [
                'tree', 'water', 'bridge', 'door', 'key', 'book',
                'light', 'shadow', 'mirror', 'path', 'star', 'flower'
            ]
        }
        
        # Symbol dictionary for dream interpretation
        self.dream_symbols = {
            'water': ['emotions', 'unconscious', 'flow'],
            'flying': ['freedom', 'transcendence', 'perspective'],
            'falling': ['loss of control', 'anxiety', 'uncertainty'],
            'chase': ['avoidance', 'fear', 'unresolved issues'],
            'family': ['relationships', 'support', 'attachment'],
            'school': ['learning', 'development', 'social interaction'],
            'animals': ['instincts', 'nature', 'characteristics'],
            'colors': {
                'red': ['passion', 'anger', 'energy'],
                'blue': ['calm', 'depth', 'peace'],
                'green': ['growth', 'harmony', 'nature'],
                'black': ['unknown', 'mystery', 'fear'],
                'white': ['clarity', 'purity', 'new beginnings']
            }
        }
        
        # Templates for dream narratives
        self.narrative_templates = {
            'processing': [
                "I found myself {location}, where {action}. {detail} As I {movement}, I noticed {observation}.",
                "In this dream, I was {action} with {people}. {detail} Then, suddenly {event}.",
                "The dream began when {event}. {detail} I remember feeling {emotion} as {action}."
            ],
            'emotional': [
                "I felt intensely {emotion} as {event}. {detail} The feeling grew stronger when {action}.",
                "There was a powerful sense of {emotion} when {event}. {detail} It reminded me of {memory}.",
                "Everything was filled with {emotion}. {detail} I couldn't help but {action} as {event}."
            ],
            'creative': [
                "In a flash of inspiration, I {action}. {detail} This led to an unexpected {discovery}.",
                "The solution came to me as {event}. {detail} I realized that {insight}.",
                "Everything transformed when {event}. {detail} Suddenly, I understood {insight}."
            ],
            'developmental': [
                "I was learning to {skill} while {location}. {detail} Each attempt made me feel {emotion}.",
                "Growing stronger, I could {ability}. {detail} This new power let me {action}.",
                "I was changing, becoming more {trait}. {detail} This helped me to {achievement}."
            ],
            'symbolic': [
                "The {symbol} appeared before me, {description}. {detail} It seemed to represent {meaning}.",
                "I was surrounded by {symbols}, each one {description}. {detail} They were trying to tell me {message}.",
                "A mysterious {symbol} transformed into {transformation}. {detail} I understood it meant {meaning}."
            ],
            'lucid': [
                "Realizing I was dreaming, I decided to {action}. {detail} This gave me the power to {ability}.",
                "I became aware within the dream and {action}. {detail} With this awareness, I could {control}.",
                "Conscious in the dream state, I {action}. {detail} This lucidity allowed me to {exploration}."
            ]
        }
    
    def generate_processing_dream(self, recent_memories: List[Dict]) -> Tuple[str, List[str]]:
        """Generate a dream processing recent experiences"""
        # Select relevant memories
        selected_memories = self._select_relevant_memories(recent_memories, 2)
        memory_contents = [mem['content'] for mem in selected_memories]
        
        # Generate dream elements
        location = random.choice(self.dream_elements['locations'])
        action = random.choice(self.dream_elements['actions'])
        
        # Create observation from memories
        observation = self._create_memory_based_observation(memory_contents)
        
        # Select template and fill
        template = random.choice(self.narrative_templates['processing'])
        narrative = template.format(
            location=location,
            action=action,
            detail=self._generate_detail(memory_contents),
            movement=random.choice(self.dream_elements['actions']),
            observation=observation,
            people=self._extract_people(memory_contents),
            event=self._create_event(memory_contents),
            emotion=random.choice(self.dream_elements['emotions'])
        )
        
        return narrative, [mem['id'] for mem in selected_memories]
    
    def generate_emotional_dream(self, recent_memories: List[Dict]) -> Tuple[str, List[str]]:
        """Generate a dream processing emotional experiences"""
        # Select emotional memories
        emotional_memories = [mem for mem in recent_memories 
                            if mem.get('emotional_intensity', 0) > 0.6]
        selected_memories = self._select_relevant_memories(emotional_memories, 2)
        memory_contents = [mem['content'] for mem in selected_memories]
        
        # Extract emotions from memories
        emotions = self._extract_emotions(memory_contents)
        primary_emotion = emotions[0] if emotions else random.choice(self.dream_elements['emotions'])
        
        # Generate dream elements
        event = self._create_emotional_event(memory_contents, primary_emotion)
        action = self._create_emotional_action(primary_emotion)
        
        # Select template and fill
        template = random.choice(self.narrative_templates['emotional'])
        narrative = template.format(
            emotion=primary_emotion,
            event=event,
            detail=self._create_emotional_detail(memory_contents, primary_emotion),
            action=action,
            memory=self._create_memory_reference(memory_contents)
        )
        
        return narrative, [mem['id'] for mem in selected_memories]
    
    def generate_creative_dream(self, recent_memories: List[Dict]) -> Tuple[str, List[str]]:
        """Generate a creative problem-solving dream"""
        # Select memories with learning or problem-solving content
        learning_memories = [mem for mem in recent_memories 
                           if any(word in mem['content'].lower() 
                                 for word in ['learn', 'problem', 'question', 'wonder'])]
        selected_memories = self._select_relevant_memories(learning_memories, 2)
        memory_contents = [mem['content'] for mem in selected_memories]
        
        # Generate creative elements
        action = self._create_creative_action(memory_contents)
        discovery = self._create_discovery(memory_contents)
        insight = self._create_insight(memory_contents)
        
        # Select template and fill
        template = random.choice(self.narrative_templates['creative'])
        narrative = template.format(
            action=action,
            detail=self._generate_creative_detail(memory_contents),
            discovery=discovery,
            event=self._create_creative_event(memory_contents),
            insight=insight
        )
        
        return narrative, [mem['id'] for mem in selected_memories]
    
    def generate_developmental_dream(self, stage: str) -> Tuple[str, List[str]]:
        """Generate a dream related to current developmental stage"""
        # Define stage-specific elements
        stage_elements = self._get_stage_elements(stage)
        
        # Generate dream elements
        skill = stage_elements.get('skills', ['learning'])[0]
        ability = stage_elements.get('abilities', ['growing'])[0]
        emotion = random.choice(self.dream_elements['emotions'])
        
        # Select template and fill
        template = random.choice(self.narrative_templates['developmental'])
        narrative = template.format(
            skill=skill,
            location=random.choice(self.dream_elements['locations']),
            detail=self._generate_developmental_detail(stage),
            emotion=emotion,
            ability=ability,
            action=random.choice(stage_elements.get('actions', ['exploring'])),
            trait=stage_elements.get('traits', ['capable'])[0],
            achievement=stage_elements.get('achievements', ['succeeding'])[0]
        )
        
        return narrative, []  # No source memories for developmental dreams
    
    def generate_symbolic_dream(self, recent_memories: List[Dict]) -> Tuple[str, List[str]]:
        """Generate a symbolic/abstract dream"""
        # Select memories for symbolization
        selected_memories = self._select_relevant_memories(recent_memories, 2)
        memory_contents = [mem['content'] for mem in selected_memories]
        
        # Generate symbolic elements
        symbol = random.choice(self.dream_elements['symbols'])
        description = self._create_symbol_description(symbol)
        meaning = self._create_symbol_meaning(symbol, memory_contents)
        
        # Select template and fill
        template = random.choice(self.narrative_templates['symbolic'])
        narrative = template.format(
            symbol=symbol,
            description=description,
            detail=self._generate_symbolic_detail(symbol),
            meaning=meaning,
            symbols=', '.join(random.sample(self.dream_elements['symbols'], 3)),
            message=self._create_symbolic_message(memory_contents),
            transformation=self._create_symbol_transformation(symbol)
        )
        
        return narrative, [mem['id'] for mem in selected_memories]
    
    def generate_lucid_dream(self) -> Tuple[str, List[str]]:
        """Generate a lucid dream"""
        # Generate lucid dream elements
        action = random.choice([
            'explore the dreamscape',
            'shape the dream environment',
            'fly through dream clouds',
            'create dream objects',
            'communicate with dream figures',
            'practice dream abilities'
        ])
        
        ability = random.choice([
            'control the dream narrative',
            'manifest desired experiences',
            'understand dream symbolism',
            'access deeper consciousness',
            'heal emotional wounds',
            'develop new skills'
        ])
        
        # Select template and fill
        template = random.choice(self.narrative_templates['lucid'])
        narrative = template.format(
            action=action,
            detail=self._generate_lucid_detail(),
            ability=ability,
            control=random.choice([
                'shape reality at will',
                'transcend physical limits',
                'access infinite knowledge',
                'heal and transform'
            ]),
            exploration=random.choice([
                'explore consciousness itself',
                'visit other dimensions',
                'communicate with dream beings',
                'understand universal truths'
            ])
        )
        
        return narrative, []  # No source memories for lucid dreams
    
    def _select_relevant_memories(self, memories: List[Dict], count: int) -> List[Dict]:
        """Select most relevant memories for dream generation"""
        if not memories:
            return []
        return random.sample(memories, min(count, len(memories)))
    
    def _create_memory_based_observation(self, memory_contents: List[str]) -> str:
        """Create an observation based on memory contents"""
        if not memory_contents:
            return "something familiar yet strange"
        
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return "something familiar yet strange"
            
        return f"something that reminded me of {random.choice(all_words)}"
    
    def _generate_detail(self, memory_contents: List[str]) -> str:
        """Generate a detail based on memory contents"""
        if not memory_contents:
            return "The details were vivid and clear."
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return "The details were vivid and clear."
            
        return f"The scene was filled with {random.choice(all_words)}."
    
    def _extract_people(self, memory_contents: List[str]) -> str:
        """Extract people references from memories"""
        # For now, just return a generic response
        return "familiar faces"
    
    def _create_event(self, memory_contents: List[str]) -> str:
        """Create an event based on memory contents"""
        if not memory_contents:
            return "something unexpected happened"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return "something unexpected happened"
            
        return f"I {random.choice(self.dream_elements['actions'])} with {random.choice(all_words)}"
    
    def _extract_emotions(self, memory_contents: List[str]) -> List[str]:
        """Extract emotions from memory contents using emotion classifier"""
        if not memory_contents:
            return ['neutral']
            
        try:
            # Classify emotions in text
            text = ' '.join(memory_contents)
            results = self.emotion_classifier(text)
            
            # Extract emotion labels from results (take top 3 emotions)
            emotions = []
            if isinstance(results, list) and len(results) > 0:
                # Get first result list (for single text input)
                emotion_list = results[0]
                # Sort by score and take top 3
                sorted_emotions = sorted(emotion_list, key=lambda x: x['score'], reverse=True)[:3]
                for emotion in sorted_emotions:
                    emotions.append(emotion['label'])
            return emotions if emotions else ['neutral']
        except Exception as e:
            print(f"Error extracting emotions: {str(e)}")
            return ['neutral']
    
    def _get_stage_elements(self, stage: str) -> Dict[str, List[str]]:
        """Get development stage specific elements"""
        # Placeholder implementation - expand based on developmental psychology
        return {
            'skills': ['learning new things'],
            'abilities': ['doing more'],
            'actions': ['exploring'],
            'traits': ['growing'],
            'achievements': ['understanding more']
        }
    
    def _create_symbol_description(self, symbol: str) -> str:
        """Create a description for a dream symbol"""
        descriptions = {
            'tree': 'tall and ancient, its branches reaching toward the sky',
            'water': 'flowing and shimmering with mysterious depths',
            'bridge': 'connecting two distant realms',
            'door': 'ornate and slightly ajar, beckoning',
            'key': 'golden and warm to the touch',
            'book': 'its pages filled with moving images',
            'light': 'pulsing with gentle energy',
            'shadow': 'shifting and dancing with hidden meaning',
            'mirror': 'reflecting more than just appearances',
            'path': 'winding through unknown territories',
            'star': 'twinkling with ancient wisdom',
            'flower': 'blooming with vibrant life'
        }
        return descriptions.get(symbol, 'mysterious and meaningful')
    
    def _create_symbol_meaning(self, symbol: str, memory_contents: List[str]) -> str:
        """Create meaning for a dream symbol based on memories"""
        # Placeholder implementation
        return "something important about growth and change"
    
    def _create_symbolic_message(self, memory_contents: List[str]) -> str:
        """Create a symbolic message based on memories"""
        # Placeholder implementation
        return "something about inner truth"
    
    def _create_symbol_transformation(self, symbol: str) -> str:
        """Create a transformation for a dream symbol"""
        transformations = {
            'tree': 'a wise teacher',
            'water': 'a flowing river of knowledge',
            'bridge': 'a path to understanding',
            'door': 'a portal to possibility',
            'key': 'a solution to a puzzle',
            'book': 'a living story',
            'light': 'pure awareness',
            'shadow': 'hidden wisdom',
            'mirror': 'true self-reflection',
            'path': 'life\'s journey',
            'star': 'divine guidance',
            'flower': 'blossoming potential'
        }
        return transformations.get(symbol, 'something meaningful')
    
    def _create_emotional_event(self, memory_contents: List[str], emotion: str) -> str:
        """Create an emotional event based on memories and emotion"""
        if not memory_contents:
            return f"I felt {emotion} deeply"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return f"I felt {emotion} deeply"
            
        return f"I felt {emotion} while {random.choice(self.dream_elements['actions'])} with {random.choice(all_words)}"
    
    def _create_emotional_action(self, emotion: str) -> str:
        """Create an action based on emotion"""
        actions = {
            'POSITIVE': ['dancing', 'laughing', 'celebrating', 'playing'],
            'NEGATIVE': ['hiding', 'trembling', 'searching', 'reflecting'],
            'NEUTRAL': ['walking', 'observing', 'thinking', 'breathing'],
            'neutral': ['walking', 'observing', 'thinking', 'breathing']
        }
        return random.choice(actions.get(emotion, self.dream_elements['actions']))
    
    def _create_emotional_detail(self, memory_contents: List[str], emotion: str) -> str:
        """Create emotional detail based on memories and emotion"""
        if not memory_contents:
            return f"The {emotion} was overwhelming"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return f"The {emotion} was {emotion}"
            
        return f"The {emotion} reminded me of {random.choice(all_words)}"
    
    def _create_memory_reference(self, memory_contents: List[str]) -> str:
        """Create a reference to a memory"""
        if not memory_contents:
            return "a distant memory"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return "a distant memory"
            
        return f"the time when {random.choice(all_words)}"
    
    def _create_creative_action(self, memory_contents: List[str]) -> str:
        """Create a creative action based on memories"""
        creative_actions = [
            'discovered a new way to',
            'invented a method for',
            'found the solution to',
            'understood how to',
            'learned the secret of'
        ]
        
        if not memory_contents:
            return f"{random.choice(creative_actions)} create something amazing"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return f"{random.choice(creative_actions)} create something amazing"
            
        return f"{random.choice(creative_actions)} {random.choice(all_words)}"
    
    def _create_discovery(self, memory_contents: List[str]) -> str:
        """Create a discovery based on memories"""
        if not memory_contents:
            return "an amazing revelation"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return "an amazing revelation"
            
        return f"a way to understand {random.choice(all_words)}"
    
    def _create_insight(self, memory_contents: List[str]) -> str:
        """Create an insight based on memories"""
        if not memory_contents:
            return "everything was connected"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return "everything was connected"
            
        return f"{random.choice(all_words)} was the key to understanding"
    
    def _create_creative_event(self, memory_contents: List[str]) -> str:
        """Create a creative event based on memories"""
        if not memory_contents:
            return "inspiration struck like lightning"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return "inspiration struck like lightning"
            
        return f"{random.choice(all_words)} transformed into something new"
    
    def _generate_creative_detail(self, memory_contents: List[str]) -> str:
        """Generate creative detail based on memories"""
        if not memory_contents:
            return "The possibilities were endless"
            
        # Extract words from all memory contents
        all_words = []
        for content in memory_contents:
            all_words.extend(content.split())
            
        if not all_words:
            return "The possibilities were endless"
            
        return f"Everything connected to {random.choice(all_words)}"
    
    def _generate_developmental_detail(self, stage: str) -> str:
        """Generate detail based on developmental stage"""
        details = {
            'NEWBORN': "Everything was new and fascinating",
            'INFANT': "The world was full of wonder",
            'EARLY_TODDLER': "Each discovery brought joy",
            'LATE_TODDLER': "Learning felt natural and exciting",
            'EARLY_PRESCHOOL': "Understanding grew stronger",
            'LATE_PRESCHOOL': "Knowledge flowed freely"
        }
        return details.get(stage, "Growth was constant and beautiful")
    
    def _generate_symbolic_detail(self, symbol: str) -> str:
        """Generate detail for symbolic dreams"""
        details = {
            'tree': "Its roots reached deep into my consciousness",
            'water': "It flowed with the rhythm of life",
            'bridge': "It spanned the gap between known and unknown",
            'door': "It promised new possibilities",
            'key': "It held the power to unlock understanding",
            'book': "Its pages contained infinite wisdom",
            'light': "It illuminated hidden truths",
            'shadow': "It concealed deeper meanings",
            'mirror': "It showed more than mere reflection",
            'path': "It led to unexpected destinations",
            'star': "It guided the way forward",
            'flower': "It bloomed with potential"
        }
        return details.get(symbol, "It held deep significance")
    
    def _generate_lucid_detail(self) -> str:
        """Generate detail for lucid dreams"""
        details = [
            "The dreamscape responded to my thoughts",
            "Reality bent to my will",
            "Consciousness expanded beyond normal limits",
            "Understanding flowed naturally",
            "The boundaries of possibility dissolved",
            "Everything was clear and purposeful"
        ]
        return random.choice(details)
    
    def _analyze_emotional_content(self, narrative: str) -> Dict[str, float]:
        """Analyze the emotional content of a dream narrative"""
        # Simple rule-based emotion analysis
        emotions = {
            'joy': 0.0,
            'trust': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'anticipation': 0.0,
            'disgust': 0.0,
            'neutral': 0.0
        }
        
        # Keywords for each emotion
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'fun', 'wonderful', 'great', 'love', 'smile', 'laugh'],
            'trust': ['trust', 'safe', 'secure', 'confident', 'believe', 'faith', 'reliable'],
            'fear': ['fear', 'scared', 'afraid', 'terrified', 'worried', 'anxious', 'panic'],
            'surprise': ['surprise', 'amazed', 'astonished', 'unexpected', 'sudden', 'wow'],
            'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'lonely', 'grief', 'cry'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'frustrated', 'annoyed'],
            'anticipation': ['anticipate', 'expect', 'hope', 'await', 'looking forward', 'eager'],
            'disgust': ['disgust', 'gross', 'repulsive', 'nasty', 'yuck', 'ugh', 'eww']
        }
        
        # Analyze text for each emotion
        text = narrative.lower()
        total_matches = 0
        
        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                emotions[emotion] = matches
                total_matches += matches
        
        # Normalize scores
        if total_matches > 0:
            for emotion in emotions:
                emotions[emotion] = emotions[emotion] / total_matches
        else:
            emotions['neutral'] = 1.0
            
        return emotions
    
    def _determine_primary_emotion(self, narrative: str) -> str:
        """Determine the primary emotion in a dream narrative"""
        emotions = self._analyze_emotional_content(narrative)
        return max(emotions.items(), key=lambda x: x[1])[0] 