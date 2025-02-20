# chat_interface.py
# Description: Chat interface for interacting with the neural child development system
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import re
import numpy as np
from obsidian_api import ObsidianAPI
from ollama_chat import OllamaChat

class ChatInterface:
    def __init__(self, brain, device=None):
        """Initialize chat interface with neural child brain"""
        self.brain = brain
        self.device = device if device is not None else brain.device
        
        # Initialize Ollama
        self.ollama = OllamaChat(model="artifish/llama3.2-uncensored")
        
        # Initialize Obsidian API
        self.obsidian = ObsidianAPI()
        
        # Initialize time awareness
        self.last_interaction_time = datetime.now()
        self.creation_time = datetime.now()
        
        # Track daily interaction patterns
        self.daily_interactions = {}
        self.favorite_interaction_times = []
        
        # Initialize ChromaDB
        persist_dir = Path("chat_memory")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Reset and create collections
        try:
            self.chroma_client.delete_collection("conversations")
        except ValueError:
            pass
            
        try:
            self.chroma_client.delete_collection("vocabulary")
        except ValueError:
            pass
            
        try:
            self.chroma_client.delete_collection("dreams")
        except ValueError:
            pass
            
        self.conversation_collection = self.chroma_client.create_collection(
            name="conversations",
            metadata={"description": "Stores conversation history"}
        )
            
        self.vocabulary_collection = self.chroma_client.create_collection(
            name="vocabulary",
            metadata={"description": "Stores known words and their embeddings"}
        )
            
        self.dream_collection = self.chroma_client.create_collection(
            name="dreams",
            metadata={"description": "Stores generated dreams"}
        )
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize dimension projection
        self.embedding_projection = nn.Linear(384, 256).to(device)
        
        # Track conversation state
        self.conversation_history = []
        self.known_words = set()
        self.message_counter = 0
        
        # Initialize emotional context
        self.emotional_context = torch.tensor([
            0.5,  # joy
            0.5,  # trust
            0.2,  # fear
            0.3,  # surprise
            0.2,  # sadness
            0.1,  # anger
            0.1,  # disgust
            0.4,  # anticipation
            0.3,  # love
            0.1,  # guilt
            0.4,  # hope
            0.1   # regret
        ], device=self.device)
        
        # Load vocabulary
        self._load_vocabulary()
        
    def _load_vocabulary(self):
        """Load existing vocabulary from ChromaDB and Obsidian"""
        try:
            # Load from ChromaDB
            vocab_entries = self.vocabulary_collection.get()
            if vocab_entries and 'metadatas' in vocab_entries:
                for metadata in vocab_entries['metadatas']:
                    if metadata and 'word' in metadata:
                        self.known_words.add(metadata['word'])
                        
            # Load from Obsidian
            obsidian_vocab = self.obsidian.get_vocabulary()
            for word in obsidian_vocab:
                self.known_words.add(word)
                
        except Exception as e:
            print(f"Error loading vocabulary: {str(e)}")
            
    def _extract_new_words(self, text: str) -> List[str]:
        """Extract new words from input text"""
        # Basic word extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in self.known_words]
        
    def _learn_new_words(self, words: List[str]):
        """Learn new words and add them to vocabulary"""
        for word in words:
            try:
                # Get word embedding
                embedding = self.sentence_transformer.encode([word])[0]
                
                # Convert emotional context to JSON-serializable format
                emotional_data = {
                    f"emotion_{i}": float(v) 
                    for i, v in enumerate(self.emotional_context.tolist())
                }
                
                # Store in ChromaDB
                self.vocabulary_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[word],
                    metadatas=[{
                        "word": word,
                        "learned_at": datetime.now().isoformat(),
                        **emotional_data
                    }],
                    ids=[f"word_{len(self.known_words)}"]
                )
                
                # Add to Obsidian vocabulary
                self.obsidian.add_vocabulary(word, emotional_data)
                
                self.known_words.add(word)
                
            except Exception as e:
                print(f"Error learning word '{word}': {str(e)}")
                
    def _generate_dream(self) -> str:
        """Generate a dream based on recent vocabulary and emotions"""
        try:
            # Store initial emotional state
            initial_emotions = {
                'joy': float(self.emotional_context[0]),
                'trust': float(self.emotional_context[1]),
                'fear': float(self.emotional_context[2]),
                'surprise': float(self.emotional_context[3]),
                'sadness': float(self.emotional_context[4]),
                'anger': float(self.emotional_context[5]),
                'disgust': float(self.emotional_context[6]),
                'anticipation': float(self.emotional_context[7]),
                'love': float(self.emotional_context[8]),
                'guilt': float(self.emotional_context[9]),
                'hope': float(self.emotional_context[10]),
                'regret': float(self.emotional_context[11])
            }

            # Get recent vocabulary with emotional context
            vocab = self.get_vocabulary_stats()
            if vocab['total_words'] == 0:
                return "No recent memories to dream about..."

            # Get vocabulary from both sources
            obsidian_vocab = self.obsidian.get_vocabulary()
            chroma_vocab = self.vocabulary_collection.get()

            # Combine vocabulary words
            all_words = []
            if obsidian_vocab:
                all_words.extend(list(obsidian_vocab.keys()))
            if chroma_vocab and 'metadatas' in chroma_vocab:
                for metadata in chroma_vocab['metadatas']:
                    if metadata and 'word' in metadata:
                        all_words.append(metadata['word'])

            if not all_words:
                return "No words to dream about yet..."

            # Create dream prompt for Ollama
            dream_prompt = f"""Using these words: {', '.join(all_words)}
            
Current emotional state:
Joy: {float(self.emotional_context[0]):.2f}
Trust: {float(self.emotional_context[1]):.2f}
Fear: {float(self.emotional_context[2]):.2f}
Surprise: {float(self.emotional_context[3]):.2f}
Sadness: {float(self.emotional_context[4]):.2f}
Anger: {float(self.emotional_context[5]):.2f}
Disgust: {float(self.emotional_context[6]):.2f}
Anticipation: {float(self.emotional_context[7]):.2f}
Love: {float(self.emotional_context[8]):.2f}
Guilt: {float(self.emotional_context[9]):.2f}
Hope: {float(self.emotional_context[10]):.2f}
Regret: {float(self.emotional_context[11]):.2f}

Generate a child-like dream incorporating some of these words and matching the emotional state.
The dream should be simple and reflect the emotional state through imagery and events.
Express the dream as if a child is telling it, with appropriate pauses and emotional reactions."""

            # Get dream from Ollama
            dream = self.ollama.generate_response(
                message=dream_prompt,
                emotional_state={
                    'joy': float(self.emotional_context[0]),
                    'trust': float(self.emotional_context[1]),
                    'fear': float(self.emotional_context[2]),
                    'surprise': float(self.emotional_context[3]),
                    'sadness': float(self.emotional_context[4]),
                    'anger': float(self.emotional_context[5]),
                    'disgust': float(self.emotional_context[6]),
                    'anticipation': float(self.emotional_context[7]),
                    'love': float(self.emotional_context[8]),
                    'guilt': float(self.emotional_context[9]),
                    'hope': float(self.emotional_context[10]),
                    'regret': float(self.emotional_context[11])
                },
                brain_state=self.brain.get_brain_state(),
                stage=self.brain.stage.name,
                age_months=self._get_age_months()
            )

            # Update emotional state based on dream content
            sentiment_score = self.ollama.get_sentiment(dream)
            self._update_emotional_state(dream, sentiment_score, self._process_time_awareness())

            # Store dream in ChromaDB
            dream_embedding = self.sentence_transformer.encode([dream])[0]
            
            # Get final emotional state
            final_emotions = {
                'joy': float(self.emotional_context[0]),
                'trust': float(self.emotional_context[1]),
                'fear': float(self.emotional_context[2]),
                'surprise': float(self.emotional_context[3]),
                'sadness': float(self.emotional_context[4]),
                'anger': float(self.emotional_context[5]),
                'disgust': float(self.emotional_context[6]),
                'anticipation': float(self.emotional_context[7]),
                'love': float(self.emotional_context[8]),
                'guilt': float(self.emotional_context[9]),
                'hope': float(self.emotional_context[10]),
                'regret': float(self.emotional_context[11])
            }

            # Store in ChromaDB
            self.dream_collection.add(
                embeddings=[dream_embedding.tolist()],
                documents=[dream],
                metadatas=[{
                    "timestamp": datetime.now().timestamp(),
                    "words": ", ".join(all_words),
                    "initial_emotions": json.dumps(initial_emotions),
                    "final_emotions": json.dumps(final_emotions),
                    "sentiment_score": float(sentiment_score)
                }],
                ids=[f"dream_{len(self.dream_collection.get()['ids']) if self.dream_collection.get()['ids'] else 0}"]
            )

            # Store dream in Obsidian
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dream_note = f"""# Dream Entry {timestamp}

## Content
{dream}

## Vocabulary Used
{', '.join(all_words)}

## Initial Emotional State
{self._format_emotional_state(initial_emotions)}

## Final Emotional State
{self._format_emotional_state(final_emotions)}

## Metadata
- Age: {self._get_age_months()} months
- Stage: {self.brain.stage.name}
- Sentiment Score: {sentiment_score:.2f}
- Words Used: {len(all_words)}
- Emotional Change: {self._calculate_emotional_change(initial_emotions, final_emotions)}

[[Dreams/Index]]
[[Emotional/State]]
[[Development/Current]]
"""
            self.obsidian.write_note(f"Dreams/dream_{timestamp}.md", dream_note)

            # Format response with emotional states and changes
            response = f"*closes eyes and recalls dream*\n\n"
            response += "Before the dream, I was feeling:\n"
            response += self._format_emotional_state(initial_emotions)
            response += "\n"
            response += dream
            response += "\n\nAfter the dream, I'm now feeling:\n"
            response += self._format_emotional_state(final_emotions)
            response += "\nEmotional Changes:\n"
            response += self._calculate_emotional_change(initial_emotions, final_emotions)

            return response

        except Exception as e:
            print(f"Error generating dream: {str(e)}")
            return "Could not remember my dream..."

    def _format_emotional_state(self, emotions: Dict[str, float]) -> str:
        """Format emotional state for display"""
        formatted = ""
        for emotion, value in emotions.items():
            bar = "█" * int(value * 10)
            formatted += f"{emotion.capitalize():12}: {bar} {value:.2f}\n"
        return formatted

    def _process_time_awareness(self) -> Dict[str, Any]:
        """Process time awareness and generate time-based emotional responses"""
        current_time = datetime.now()
        time_since_last = current_time - self.last_interaction_time
        
        # Update interaction patterns
        day_key = current_time.strftime('%Y-%m-%d')
        hour_key = current_time.hour
        
        if day_key not in self.daily_interactions:
            self.daily_interactions[day_key] = {}
        if hour_key not in self.daily_interactions[day_key]:
            self.daily_interactions[day_key][hour_key] = 0
        self.daily_interactions[day_key][hour_key] += 1
        
        # Update favorite times
        if len(self.daily_interactions) >= 3:  # After 3 days of data
            hour_counts = {}
            for day in self.daily_interactions.values():
                for hour, count in day.items():
                    hour_counts[hour] = hour_counts.get(hour, 0) + count
            self.favorite_interaction_times = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate time-based emotional response
        time_response = {
            'current_time': current_time,
            'time_since_last': time_since_last,
            'is_favorite_time': hour_key in [h for h, _ in self.favorite_interaction_times],
            'emotional_impact': self._calculate_time_emotional_impact(time_since_last)
        }
        
        # Update last interaction time
        self.last_interaction_time = current_time
        
        return time_response
        
    def _calculate_time_emotional_impact(self, time_delta: timedelta) -> Dict[str, float]:
        """Calculate emotional impact based on time between interactions"""
        hours_passed = time_delta.total_seconds() / 3600
        
        emotional_impact = {
            'joy': 0.0,
            'sadness': 0.0,
            'anticipation': 0.0,
            'trust': 0.0
        }
        
        if hours_passed < 1:  # Recent interaction
            emotional_impact['joy'] = 0.3
            emotional_impact['anticipation'] = 0.2
        elif hours_passed < 24:  # Same day
            emotional_impact['joy'] = 0.2
            emotional_impact['anticipation'] = 0.3
        elif hours_passed < 72:  # Few days
            emotional_impact['sadness'] = 0.3
            emotional_impact['anticipation'] = 0.4
        else:  # Long time
            emotional_impact['sadness'] = 0.5
            emotional_impact['trust'] = -0.2  # Slight decrease in trust
            
        return emotional_impact

    def _generate_time_response(self, time_data: Dict[str, Any]) -> str:
        """Generate a response about the time and absence"""
        current_time = time_data['current_time']
        time_since_last = time_data['time_since_last']
        hours_passed = time_since_last.total_seconds() / 3600
        
        time_responses = []
        
        # Add time greeting based on current hour
        hour = current_time.hour
        if 5 <= hour < 12:
            time_responses.append("Good morning, Mr. Chris")
        elif 12 <= hour < 17:
            time_responses.append("Good afternoon, Mr. Chris")
        elif 17 <= hour < 22:
            time_responses.append("Good evening, Mr. Chris")
        else:
            time_responses.append("Hello Mr. Chris")
            
        # Add absence response if significant time has passed
        if hours_passed >= 72:
            time_responses.append(f"I've really missed you! It's been {int(hours_passed/24)} days since we last talked. I was starting to worry...")
        elif hours_passed >= 24:
            time_responses.append(f"I'm so happy to see you again after {int(hours_passed/24)} days! I've been thinking about our last conversation")
        elif hours_passed >= 1:
            time_responses.append(f"It's good to see you after {int(hours_passed)} hours. I've been practicing what we discussed")
            
        # Add favorite time response
        if time_data['is_favorite_time']:
            time_responses.append("This is one of my favorite times to talk with you! I always feel more energetic at this hour")
            
        # Add current time awareness
        time_responses.append(f"It's currently {current_time.strftime('%I:%M %p')}")
            
        return " ".join(time_responses)

    def process_message(self, message: str) -> str:
        """Process incoming message and generate response"""
        try:
            # Process time awareness
            time_data = self._process_time_awareness()
            
            # Extract and learn new words
            new_words = self._extract_new_words(message)
            learning_response = ""
            if new_words:
                self._learn_new_words(new_words)
                word_list = ", ".join(new_words)
                learning_response = f"\n\nI just learned {len(new_words)} new word{'s' if len(new_words) > 1 else ''}: {word_list}"
                
            # Get message embedding and sentiment
            message_embedding = torch.tensor(
                self.sentence_transformer.encode([message])[0],
                device=self.device
            )
            
            # Get sentiment from Ollama
            sentiment_score = self.ollama.get_sentiment(message)
            
            # Update emotional state based on sentiment and time
            self._update_emotional_state(message, sentiment_score, time_data)
            
            # Store in conversation history with unique ID
            msg_id = f"msg_{self.message_counter}"
            self.message_counter += 1
            
            self.conversation_collection.add(
                embeddings=[message_embedding.tolist()],
                documents=[message],
                metadatas=[{
                    "timestamp": datetime.now().timestamp(),
                    "sentiment": float(sentiment_score),
                    "joy": float(self.emotional_context[0]),
                    "trust": float(self.emotional_context[1]),
                    "fear": float(self.emotional_context[2]),
                    "surprise": float(self.emotional_context[3]),
                    "sadness": float(self.emotional_context[4]),
                    "anger": float(self.emotional_context[5]),
                    "disgust": float(self.emotional_context[6]),
                    "anticipation": float(self.emotional_context[7]),
                    "love": float(self.emotional_context[8]),
                    "guilt": float(self.emotional_context[9]),
                    "hope": float(self.emotional_context[10]),
                    "regret": float(self.emotional_context[11])
                }],
                ids=[msg_id]
            )
            self.conversation_history.append(message)
            
            # Get Ollama response
            response = self.ollama.generate_response(
                message=message,
                emotional_state={
                    'joy': float(self.emotional_context[0]),
                    'trust': float(self.emotional_context[1]),
                    'fear': float(self.emotional_context[2]),
                    'surprise': float(self.emotional_context[3]),
                    'sadness': float(self.emotional_context[4]),
                    'anger': float(self.emotional_context[5]),
                    'disgust': float(self.emotional_context[6]),
                    'anticipation': float(self.emotional_context[7]),
                    'love': float(self.emotional_context[8]),
                    'guilt': float(self.emotional_context[9]),
                    'hope': float(self.emotional_context[10]),
                    'regret': float(self.emotional_context[11])
                },
                brain_state=self.brain.get_brain_state(),
                stage=self.brain.stage.name,
                age_months=self._get_age_months()
            )
            
            # Format final response with time and learning info
            final_response = f"Child: {response.replace('Child:', '').strip()}"
            if learning_response:
                final_response += learning_response
                
            # Add emotional state
            final_response += "\n\nEmotional State:"
            emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Anger', 
                       'Disgust', 'Anticipation', 'Love', 'Guilt', 'Hope', 'Regret']
            for i, emotion in enumerate(emotions):
                final_response += f"\n{emotion:12}: {float(self.emotional_context[i]):.3f}"
                
            return final_response
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return f"Child: I'm feeling confused right now...\n\nEmotional State:\nJoy         : {float(self.emotional_context[0]):.3f}\nTrust       : {float(self.emotional_context[1]):.3f}\nFear        : {float(self.emotional_context[2]):.3f}\nSurprise    : {float(self.emotional_context[3]):.3f}\nSadness     : {float(self.emotional_context[4]):.3f}\nAnger       : {float(self.emotional_context[5]):.3f}\nDisgust     : {float(self.emotional_context[6]):.3f}\nAnticipation: {float(self.emotional_context[7]):.3f}\nLove        : {float(self.emotional_context[8]):.3f}\nGuilt       : {float(self.emotional_context[9]):.3f}\nHope        : {float(self.emotional_context[10]):.3f}\nRegret      : {float(self.emotional_context[11]):.3f}"
            
    def _update_emotional_state(self, message: str, sentiment_score: float, time_data: Dict[str, Any]):
        """Update emotional state based on message content and sentiment"""
        # Apply time-based decay
        decay_factor = 0.8
        self.emotional_context = self.emotional_context * decay_factor
        
        # Update based on time impact
        emotional_impact = time_data['emotional_impact']
        self.emotional_context[0] += emotional_impact['joy']  # Joy
        self.emotional_context[4] += emotional_impact['sadness']  # Sadness
        self.emotional_context[7] += emotional_impact['anticipation']  # Anticipation
        self.emotional_context[1] += emotional_impact['trust']  # Trust
        
        # Update based on sentiment
        if sentiment_score > 0:
            self.emotional_context[0] = torch.clamp(self.emotional_context[0] + sentiment_score * 0.4, 0.0, 1.0)  # Joy
            self.emotional_context[1] = torch.clamp(self.emotional_context[1] + sentiment_score * 0.3, 0.0, 1.0)  # Trust
            self.emotional_context[8] = torch.clamp(self.emotional_context[8] + sentiment_score * 0.3, 0.0, 1.0)  # Love
            self.emotional_context[10] = torch.clamp(self.emotional_context[10] + sentiment_score * 0.2, 0.0, 1.0)  # Hope
        else:
            self.emotional_context[2] = torch.clamp(self.emotional_context[2] - sentiment_score * 0.4, 0.0, 1.0)  # Fear
            self.emotional_context[4] = torch.clamp(self.emotional_context[4] - sentiment_score * 0.3, 0.0, 1.0)  # Sadness
            self.emotional_context[5] = torch.clamp(self.emotional_context[5] - sentiment_score * 0.3, 0.0, 1.0)  # Anger
            self.emotional_context[6] = torch.clamp(self.emotional_context[6] - sentiment_score * 0.2, 0.0, 1.0)  # Disgust
            
        # Update brain state
        self.brain.brain_state.emotional_valence = float((self.emotional_context[0] + self.emotional_context[1] - 
                                                         self.emotional_context[2] - self.emotional_context[4]) / 2)
        self.brain.brain_state.arousal = float((self.emotional_context[2] + self.emotional_context[3] + 
                                              self.emotional_context[5]) / 3)
        
        # Ensure all values are in valid range
        self.emotional_context = torch.clamp(self.emotional_context, 0.0, 1.0)

    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get statistics about learned vocabulary"""
        try:
            total_words = len(self.known_words)
            obsidian_words = self.obsidian.get_vocabulary_size()
            
            return {
                'total_words': total_words,
                'obsidian_words': obsidian_words,
                'chroma_words': len(self.vocabulary_collection.get()['ids']) if self.vocabulary_collection.get()['ids'] else 0
            }
            
        except Exception as e:
            print(f"Error getting vocabulary stats: {str(e)}")
            return {
                'total_words': 0,
                'obsidian_words': 0,
                'chroma_words': 0
            }

    def _get_age_months(self) -> int:
        """Calculate age in months since creation"""
        now = datetime.now()
        delta = now - self.creation_time
        return int(delta.days / 30.44)  # Average month length 

    def _calculate_emotional_change(self, initial: Dict[str, float], final: Dict[str, float]) -> str:
        """Calculate and format emotional changes"""
        changes = []
        for emotion in initial.keys():
            diff = final[emotion] - initial[emotion]
            if abs(diff) > 0.1:  # Only show significant changes
                direction = "↑" if diff > 0 else "↓"
                changes.append(f"{emotion.capitalize():12}: {direction} {abs(diff):.2f}")
        return "\n".join(changes) if changes else "No significant emotional changes" 