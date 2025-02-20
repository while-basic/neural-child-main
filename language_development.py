# language_development.py
# Description: Language development and learning system for the neural child
# Created by: Christopher Celaya

import torch
import torch.nn as nn
import json
import re
from collections import defaultdict
from enum import Enum, auto
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import spacy
from transformers import pipeline
from llm_module import chat_completion

class LanguageStage(Enum):
    PRELINGUISTIC = auto()  # 0-12 months: Crying, cooing, babbling
    HOLOPHRASTIC = auto()   # 12-18 months: Single words
    TELEGRAPHIC = auto()    # 18-24 months: Two-word combinations
    MULTIWORD = auto()      # 2-3 years: Simple sentences
    COMPLEX = auto()        # 3+ years: Complex sentences

class WordCategory(Enum):
    NOUN = auto()
    VERB = auto()
    ADJECTIVE = auto()
    PRONOUN = auto()
    PREPOSITION = auto()
    ARTICLE = auto()
    ADVERB = auto()
    INTERJECTION = auto()

class LanguageDevelopment(nn.Module):
    def __init__(self, device='cuda'):
        """Initialize the language development system"""
        super().__init__()
        self.device = device
        
        # Initialize NLP components first
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = pipeline("sentiment-analysis", device=0 if device == 'cuda' else -1)
        
        # Initialize language development state
        self.current_stage = LanguageStage.PRELINGUISTIC
        self.stage_progress = 0.0
        self.vocabulary = {category: set() for category in WordCategory}
        
        # Initialize linguistic rules
        self._initialize_linguistic_rules()
        
        # Stage-specific utterance patterns
        self.utterance_patterns = {
            LanguageStage.PRELINGUISTIC: ["OOH!", "AAH!", "WAAH!", "BAH!", "MAH!"],
            LanguageStage.HOLOPHRASTIC: ["MAMA!", "DADA!", "BALL!", "UP!", "NO!"],
            LanguageStage.TELEGRAPHIC: ["WANT MILK", "MORE PLAY", "BIG BALL", "DADDY GO"],
            LanguageStage.MULTIWORD: ["I WANT MILK", "PLAY WITH ME", "MOMMY READ BOOK"],
            LanguageStage.COMPLEX: ["Can I have milk please?", "I want to play with my toys."]
        }
        
        # Initialize word emotions
        self.word_emotions = defaultdict(lambda: torch.zeros(4))  # [joy, trust, fear, surprise]
        
        # Initialize metrics
        self.metrics = {
            'vocabulary_size': 0,
            'grammar_complexity': 0.0,
            'comprehension_level': 0.0,
            'expression_level': 0.0
        }
        
        # Initialize phoneme mastery
        self.phoneme_mastery = {
            'p': 0.0, 'b': 0.0, 'm': 0.0,  # Early developing sounds
            't': 0.0, 'd': 0.0, 'n': 0.0,
            'k': 0.0, 'g': 0.0, 'w': 0.0,
            'h': 0.0, 'y': 0.0,
            'f': 0.0, 'v': 0.0,  # Middle developing sounds
            'sh': 0.0, 'ch': 0.0,
            'j': 0.0, 'l': 0.0,
            's': 0.0, 'z': 0.0,  # Late developing sounds
            'r': 0.0, 'th': 0.0
        }
        
        # Word frequency and usage tracking
        self.word_frequency = defaultdict(int)
        self.word_contexts = defaultdict(list)
        self.recent_utterances = []
        
        # Neural components for language processing
        self.word_embedding = nn.Embedding(5000, 128).to(device)  # Start with 5000 word capacity
        self.utterance_generator = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)
        
        # Initialize stage-specific patterns
        self.initialize_stage_patterns()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.retention_rate = 0.95
        
    def initialize_stage_patterns(self):
        """Initialize language patterns for each developmental stage"""
        self.stage_patterns = {
            LanguageStage.PRELINGUISTIC: {
                'sounds': ['aa', 'oo', 'ee', 'ma', 'ba', 'da'],
                'patterns': ['CV', 'CVCV'],  # C=consonant, V=vowel
                'intonations': ['rising', 'falling', 'flat']
            },
            LanguageStage.HOLOPHRASTIC: {
                'word_types': ['mama', 'dada', 'no', 'more', 'up'],
                'contexts': ['needs', 'wants', 'observations'],
                'gestures': ['pointing', 'reaching', 'showing']
            },
            LanguageStage.TELEGRAPHIC: {
                'combinations': ['noun+verb', 'adjective+noun', 'verb+object'],
                'meanings': ['action', 'possession', 'location', 'description']
            },
            LanguageStage.MULTIWORD: {
                'sentence_patterns': ['subject-verb-object', 'subject-verb'],
                'question_forms': ['what', 'where', 'who'],
                'tenses': ['present']
            },
            LanguageStage.COMPLEX: {
                'sentence_types': ['compound', 'complex'],
                'tenses': ['present', 'past', 'future'],
                'modalities': ['can', 'should', 'might']
            }
        }
    
    def _initialize_linguistic_rules(self):
        """Initialize linguistic rules for word categorization"""
        self.linguistic_rules = {
            WordCategory.NOUN: {
                'suffixes': ['ness', 'ment', 'ship', 'dom', 'er', 'or', 'ist'],
                'prefixes': ['super', 'sub', 'inter', 'trans'],
                'patterns': [r'^[A-Z][a-z]+$']  # Proper nouns
            },
            WordCategory.VERB: {
                'suffixes': ['ate', 'ize', 'ify', 'en', 'ed', 'ing'],
                'prefixes': ['re', 'un', 'dis', 'mis'],
                'patterns': [r'[a-z]+(?:ed|ing|s)$']
            },
            WordCategory.ADJECTIVE: {
                'suffixes': ['ful', 'ous', 'able', 'ible', 'al', 'ive', 'less', 'ish'],
                'prefixes': ['un', 'in', 'im', 'il', 'ir'],
                'patterns': [r'[a-z]+(?:er|est)$']
            },
            WordCategory.ADVERB: {
                'suffixes': ['ly', 'ward', 'wise'],
                'patterns': [r'[a-z]+ly$']
            }
        }
        
        # Common words by category
        self.common_words = {
            WordCategory.PRONOUN: {'i', 'me', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'},
            WordCategory.PREPOSITION: {'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'of'},
            WordCategory.ARTICLE: {'a', 'an', 'the'},
            WordCategory.INTERJECTION: {'oh', 'wow', 'oops', 'yay', 'hey', 'hi', 'hello', 'bye'}
        }
    
    def _categorize_word(self, word: str) -> WordCategory:
        """Categorize a word using linguistic rules and NLP"""
        word = word.lower().strip()
        
        # Check common word lists first
        for category, words in self.common_words.items():
            if word in words:
                return category
        
        # Use spaCy for initial POS tagging
        doc = self.nlp(word)
        pos_tag = doc[0].pos_
        
        # Map spaCy POS tags to WordCategory
        pos_mapping = {
            'NOUN': WordCategory.NOUN,
            'VERB': WordCategory.VERB,
            'ADJ': WordCategory.ADJECTIVE,
            'ADV': WordCategory.ADVERB,
            'PRON': WordCategory.PRONOUN,
            'ADP': WordCategory.PREPOSITION,
            'DET': WordCategory.ARTICLE,
            'INTJ': WordCategory.INTERJECTION
        }
        
        if pos_tag in pos_mapping:
            return pos_mapping[pos_tag]
        
        # Apply linguistic rules if spaCy is uncertain
        for category, rules in self.linguistic_rules.items():
            # Check suffixes
            if 'suffixes' in rules:
                if any(word.endswith(suffix) for suffix in rules['suffixes']):
                    return category
            
            # Check prefixes
            if 'prefixes' in rules:
                if any(word.startswith(prefix) for prefix in rules['prefixes']):
                    return category
            
            # Check patterns
            if 'patterns' in rules:
                if any(re.match(pattern, word) for pattern in rules['patterns']):
                    return category
        
        # Default to NOUN if no other category is determined
        return WordCategory.NOUN
    
    def learn_from_interaction(self, utterance: str, emotional_state: torch.Tensor, 
                             developmental_stage: str) -> Dict[str, Any]:
        """Learn from an interaction, updating language stage and vocabulary"""
        # First, use spaCy for basic tokenization and POS tagging
        doc = self.nlp(utterance)
        new_words = set()
        
        # Use Ollama to enrich understanding
        enrichment_prompt = f"""
        Analyze this utterance for vocabulary learning: "{utterance}"
        Identify:
        1. Core words and their categories (noun, verb, etc.)
        2. Related words that could be learned
        3. Simple definitions suitable for a child
        4. Example usage in simple sentences
        
        Respond in JSON format with these fields.
        """
        
        try:
            enriched_data = chat_completion(
                system_prompt="You are a language development expert helping a child learn new words.",
                user_prompt=enrichment_prompt,
                structured_output=True
            )
            
            if enriched_data and isinstance(enriched_data, dict):
                # Process core words
                if 'core_words' in enriched_data:
                    for word_info in enriched_data['core_words']:
                        word = word_info.get('word', '').lower()
                        category = self._map_category(word_info.get('category', ''))
                        if word and category and word not in self.vocabulary[category]:
                            self.vocabulary[category].add(word)
                            new_words.add(word)
                            
                            # Store definition and examples if available
                            if 'definition' in word_info:
                                self.word_contexts[word].append({
                                    'definition': word_info['definition'],
                                    'examples': word_info.get('examples', [])
                                })
                
                # Process related words
                if 'related_words' in enriched_data:
                    for word_info in enriched_data['related_words']:
                        word = word_info.get('word', '').lower()
                        category = self._map_category(word_info.get('category', ''))
                        if word and category and word not in self.vocabulary[category]:
                            self.vocabulary[category].add(word)
                            new_words.add(word)
        
        except Exception as e:
            print(f"Error in Ollama enrichment: {str(e)}")
        
        # Continue with spaCy processing for words that might have been missed
        for token in doc:
            if not token.is_punct and not token.is_space:
                word = token.text.lower()
                category = self._categorize_word(word)
                if word not in self.vocabulary[category]:
                    self.vocabulary[category].add(word)
                    new_words.add(word)
        
        # Update stage progress based on vocabulary size and complexity
        total_vocab_size = sum(len(words) for words in self.vocabulary.values())
        stage_thresholds = {
            'NEWBORN': 10,
            'INFANT': 50,
            'TODDLER': 200,
            'EARLY_CHILDHOOD': 500
        }
        
        # Calculate stage progress
        if developmental_stage in stage_thresholds:
            threshold = stage_thresholds[developmental_stage]
            self.stage_progress = min(1.0, total_vocab_size / threshold)
        
        # Update language stage based on progress
        if self.stage_progress >= 0.8:
            if self.current_stage == LanguageStage.PRELINGUISTIC:
                self.current_stage = LanguageStage.HOLOPHRASTIC
            elif self.current_stage == LanguageStage.HOLOPHRASTIC:
                self.current_stage = LanguageStage.TELEGRAPHIC
            elif self.current_stage == LanguageStage.TELEGRAPHIC:
                self.current_stage = LanguageStage.MULTIWORD
            elif self.current_stage == LanguageStage.MULTIWORD:
                self.current_stage = LanguageStage.COMPLEX
        
        return {
            'current_stage': self.current_stage.value,
            'stage_progress': self.stage_progress,
            'new_words': list(new_words),
            'metrics': {
                'vocabulary_size': total_vocab_size,
                'nouns': len(self.vocabulary[WordCategory.NOUN]),
                'verbs': len(self.vocabulary[WordCategory.VERB]),
                'adjectives': len(self.vocabulary[WordCategory.ADJECTIVE])
            }
        }
    
    def _map_category(self, category_str: str) -> Optional[WordCategory]:
        """Map string category to WordCategory enum"""
        category_map = {
            'noun': WordCategory.NOUN,
            'verb': WordCategory.VERB,
            'adjective': WordCategory.ADJECTIVE,
            'adverb': WordCategory.ADVERB,
            'pronoun': WordCategory.PRONOUN,
            'preposition': WordCategory.PREPOSITION,
            'article': WordCategory.ARTICLE,
            'interjection': WordCategory.INTERJECTION
        }
        return category_map.get(category_str.lower())
    
    def generate_utterance(self, context: str, emotional_state: torch.Tensor) -> str:
        """Generate an age-appropriate utterance based on context and emotional state"""
        # Analyze sentiment of context
        sentiment = self.sentiment_analyzer(context)[0]
        is_positive = sentiment['label'] == 'POSITIVE'
        
        # Select appropriate patterns based on current stage
        patterns = self.utterance_patterns[self.current_stage]
        
        # Modify selection based on emotional state
        joy, trust, fear, surprise = emotional_state.cpu().numpy()
        
        if self.current_stage == LanguageStage.PRELINGUISTIC:
            if joy > 0.7 or trust > 0.7:
                return np.random.choice(['OOH!', 'AAH!'])
            elif fear > 0.7:
                return 'WAAH!'
            elif surprise > 0.7:
                return np.random.choice(['OOH!', 'BAH!'])
            else:
                return np.random.choice(patterns)
        
        elif self.current_stage == LanguageStage.HOLOPHRASTIC:
            if 'milk' in context.lower() or 'hungry' in context.lower():
                return 'MILK!'
            elif 'play' in context.lower():
                return 'PLAY!'
            elif any(word in context.lower() for word in ['sleep', 'nap', 'tired']):
                return 'SLEEP!'
            else:
                return np.random.choice(patterns)
        
        elif self.current_stage == LanguageStage.TELEGRAPHIC:
            if is_positive and (joy > 0.6 or trust > 0.6):
                return 'ME HAPPY'
            elif fear > 0.6:
                return 'NO SCARY'
            elif 'play' in context.lower():
                return 'WANT PLAY'
            else:
                return np.random.choice(patterns)
        
        elif self.current_stage == LanguageStage.MULTIWORD:
            if is_positive and (joy > 0.6 or trust > 0.6):
                return 'I VERY HAPPY'
            elif fear > 0.6:
                return 'I FEEL SCARED'
            elif 'play' in context.lower():
                return 'I WANT PLAY NOW'
            else:
                return np.random.choice(patterns)
        
        else:  # COMPLEX stage
            if is_positive and (joy > 0.6 or trust > 0.6):
                return "I'm feeling very happy!"
            elif fear > 0.6:
                return "I'm a little scared."
            elif 'play' in context.lower():
                return "Can we play together?"
            else:
                return np.random.choice(patterns)
    
    def _generate_prelinguistic(self) -> str:
        """Generate prelinguistic sounds (crying, cooing, babbling)"""
        patterns = self.stage_patterns[LanguageStage.PRELINGUISTIC]
        
        if self.stage_progress < 0.3:  # Early stage - crying and basic sounds
            sounds = ['waah', 'aah', 'ooh']
            return np.random.choice(sounds).upper() + "!"
        elif self.stage_progress < 0.6:  # Middle stage - cooing
            sounds = patterns['sounds'][:3]  # Use vowel sounds
            return (np.random.choice(sounds) * np.random.randint(1, 3)).lower()
        else:  # Late stage - babbling
            consonants = ['b', 'm', 'd']
            vowels = ['a', 'o', 'e']
            syllable = np.random.choice(consonants) + np.random.choice(vowels)
            return (syllable * np.random.randint(1, 4)).lower()
    
    def _generate_holophrastic(self, emotional_state: torch.Tensor) -> str:
        """Generate single-word utterances"""
        # Get words with similar emotional associations
        suitable_words = []
        for word, emotions in self.word_emotions.items():
            if torch.norm(emotions - emotional_state) < 0.5:
                suitable_words.append(word)
        
        if not suitable_words:  # Fallback to stage patterns
            suitable_words = self.stage_patterns[LanguageStage.HOLOPHRASTIC]['word_types']
        
        word = np.random.choice(suitable_words)
        
        # Add emphasis based on emotional state
        if emotional_state[0] > 0.7:  # High joy
            word = word.upper() + "!"
        elif emotional_state[2] > 0.7:  # High fear
            word = word.upper() + "!!"
        
        return word
    
    def _generate_telegraphic(self, context: str, emotional_state: torch.Tensor) -> str:
        """Generate two-word combinations"""
        patterns = self.stage_patterns[LanguageStage.TELEGRAPHIC]['combinations']
        pattern = np.random.choice(patterns)
        
        if pattern == 'noun+verb':
            noun = self._get_suitable_word(WordCategory.NOUN, emotional_state)
            verb = self._get_suitable_word(WordCategory.VERB, emotional_state)
            return f"{noun} {verb}"
        elif pattern == 'adjective+noun':
            adj = self._get_suitable_word(WordCategory.ADJECTIVE, emotional_state)
            noun = self._get_suitable_word(WordCategory.NOUN, emotional_state)
            return f"{adj} {noun}"
        else:  # verb+object
            verb = self._get_suitable_word(WordCategory.VERB, emotional_state)
            obj = self._get_suitable_word(WordCategory.NOUN, emotional_state)
            return f"{verb} {obj}"
    
    def _generate_multiword(self, context: str, emotional_state: torch.Tensor) -> str:
        """Generate simple sentences"""
        pattern = np.random.choice(self.stage_patterns[LanguageStage.MULTIWORD]['sentence_patterns'])
        
        if pattern == 'subject-verb-object':
            subj = self._get_suitable_word(WordCategory.NOUN, emotional_state)
            verb = self._get_suitable_word(WordCategory.VERB, emotional_state)
            obj = self._get_suitable_word(WordCategory.NOUN, emotional_state)
            return f"{subj} {verb} {obj}"
        else:  # subject-verb
            subj = self._get_suitable_word(WordCategory.NOUN, emotional_state)
            verb = self._get_suitable_word(WordCategory.VERB, emotional_state)
            return f"{subj} {verb}"
    
    def _generate_complex(self, context: str, emotional_state: torch.Tensor) -> str:
        """Generate complex sentences"""
        # This is a simplified version - can be expanded based on needs
        main_clause = self._generate_multiword(context, emotional_state)
        if np.random.random() < self.stage_progress:
            conjunction = np.random.choice(['and', 'but', 'because'])
            second_clause = self._generate_multiword(context, emotional_state)
            return f"{main_clause} {conjunction} {second_clause}"
        return main_clause
    
    def _get_suitable_word(self, category: WordCategory, emotional_state: torch.Tensor) -> str:
        """Get a word from category that matches emotional state"""
        suitable_words = list(self.vocabulary[category])
        if not suitable_words:
            # Fallback words for each category
            fallbacks = {
                WordCategory.NOUN: ['mama', 'dada', 'baby'],
                WordCategory.VERB: ['want', 'see', 'go'],
                WordCategory.ADJECTIVE: ['good', 'bad', 'big'],
                WordCategory.PRONOUN: ['me', 'you', 'it'],
                WordCategory.PREPOSITION: ['up', 'in', 'on'],
                WordCategory.ARTICLE: ['a', 'the'],
                WordCategory.ADVERB: ['now', 'more'],
                WordCategory.INTERJECTION: ['oh', 'wow', 'yay']
            }
            return np.random.choice(fallbacks[category])
        
        return np.random.choice(suitable_words)
    
    def _extract_words(self, utterance: str) -> List[str]:
        """Extract individual words from utterance"""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', utterance.lower())
        return words
    
    def _extract_context(self, utterance: str, word: str) -> Optional[str]:
        """Extract context around a word"""
        words = utterance.split()
        try:
            idx = words.index(word)
            start = max(0, idx - 2)
            end = min(len(words), idx + 3)
            return ' '.join(words[start:end])
        except ValueError:
            return None
    
    def _update_metrics(self, utterance: str):
        """Update developmental metrics"""
        words = self._extract_words(utterance)
        
        # Update vocabulary size
        self.metrics['vocabulary_size'] = sum(len(words) for words in self.vocabulary.values())
        
        # Update utterance length (moving average)
        current_length = len(words)
        self.metrics['utterance_length'] = (0.9 * self.metrics['utterance_length'] + 
                                          0.1 * current_length)
        
        # Update grammar complexity (based on word categories present)
        categories_present = set()
        for word in words:
            category = self._categorize_word(word)
            if category:
                categories_present.add(category)
        grammar_score = len(categories_present) / len(WordCategory)
        self.metrics['grammar_complexity'] = (0.9 * self.metrics['grammar_complexity'] + 
                                            0.1 * grammar_score)
        
        # Pronunciation accuracy increases with stage progress
        self.metrics['pronunciation_accuracy'] = min(0.95, 
                                                   0.3 + 0.7 * self.stage_progress)
        
        # Comprehension level based on vocabulary size and grammar
        self.metrics['comprehension_level'] = (self.metrics['vocabulary_size'] / 1000 + 
                                             self.metrics['grammar_complexity']) / 2
    
    def get_development_summary(self) -> Dict:
        """Get summary of current language development"""
        return {
            'stage': self.current_stage.name,
            'stage_progress': self.stage_progress,
            'vocabulary_size': self.metrics['vocabulary_size'],
            'most_frequent_words': sorted(
                self.word_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'metrics': self.metrics,
            'recent_utterances': self.recent_utterances[-5:],
            'phoneme_mastery': dict(self.phoneme_mastery)
        }

    def save_state(self, filepath: str):
        """Save the current language development state"""
        state = {
            'current_stage': self.current_stage.value,
            'stage_progress': self.stage_progress,
            'vocabulary': {category.value: list(words) 
                         for category, words in self.vocabulary.items()},
            'metrics': self.metrics,
            'phoneme_mastery': dict(self.phoneme_mastery)
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load a saved language development state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_stage = LanguageStage(state['current_stage'])
        self.stage_progress = state['stage_progress']
        self.vocabulary = {WordCategory(category): set(words) 
                         for category, words in state['vocabulary'].items()}
        self.metrics = state['metrics']
        self.phoneme_mastery = state['phoneme_mastery'] 

    def enrich_vocabulary(self, word_limit: int = 10) -> Dict[str, Any]:
        """Periodically enrich vocabulary with related words using Ollama"""
        # Select a sample of known words to expand upon
        known_words = []
        for category, words in self.vocabulary.items():
            known_words.extend(list(words))
        
        if not known_words:
            return {'new_words': [], 'enrichment_success': False}
        
        # Select random words to enrich, limited by word_limit
        sample_size = min(word_limit, len(known_words))
        sample_words = np.random.choice(known_words, size=sample_size, replace=False)
        
        enrichment_prompt = f"""
        For each of these words, suggest related words that a child might learn:
        {', '.join(sample_words)}
        
        For each word provide:
        1. Related words (synonyms, antonyms, associated words)
        2. Simple child-friendly definitions
        3. Example sentences using the word
        4. Word category (noun, verb, etc.)
        
        Respond in JSON format.
        """
        
        try:
            enriched_data = chat_completion(
                system_prompt="You are a language development expert helping expand a child's vocabulary.",
                user_prompt=enrichment_prompt,
                structured_output=True
            )
            
            new_words = set()
            if enriched_data and isinstance(enriched_data, dict):
                for word_data in enriched_data.get('enriched_words', []):
                    # Process related words
                    for related in word_data.get('related_words', []):
                        word = related.get('word', '').lower()
                        category = self._map_category(related.get('category', ''))
                        if word and category and word not in self.vocabulary[category]:
                            self.vocabulary[category].add(word)
                            new_words.add(word)
                            
                            # Store context information
                            if 'definition' in related:
                                self.word_contexts[word].append({
                                    'definition': related['definition'],
                                    'examples': related.get('examples', []),
                                    'learned_from': word_data.get('source_word', '')
                                })
            
            return {
                'new_words': list(new_words),
                'enrichment_success': True,
                'words_added': len(new_words)
            }
            
        except Exception as e:
            print(f"Error in vocabulary enrichment: {str(e)}")
            return {
                'new_words': [],
                'enrichment_success': False,
                'error': str(e)
            }

    def identify_new_words(self, text: str) -> List[str]:
        """Identify potential new words from the input text."""
        # Process text with spaCy
        doc = self.nlp(text)
        new_words = []
        
        # Check each token
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
                
            word = token.text.lower()
            # Check if word is in any category's vocabulary
            is_known = any(word in vocab for vocab in self.vocabulary.values())
            
            if not is_known:
                # Get word category
                category = self._categorize_word(word)
                if category:
                    new_words.append(word)
        
        return new_words

    def identify_new_concepts(self, text: str) -> List[str]:
        """Identify potential new concepts from the input text."""
        # Use Ollama to identify concepts
        concept_prompt = f"""
        Analyze this text for key concepts a child might learn: "{text}"
        Identify simple concepts that are:
        1. Age-appropriate for the current stage: {self.current_stage.name}
        2. Not too complex
        3. Related to basic understanding
        
        Format response as a JSON list of concept strings.
        """
        
        try:
            response = chat_completion(
                system_prompt="You are a child development expert identifying learning concepts.",
                user_prompt=concept_prompt,
                structured_output=True
            )
            
            if isinstance(response, dict) and "concepts" in response:
                return response["concepts"]
            
            return []
            
        except Exception as e:
            print(f"Error identifying concepts: {str(e)}")
            return []

    def get_current_stage(self) -> str:
        """Get the current language development stage name."""
        return self.current_stage.name

    def learn_word(self, word: str):
        """Learn a new word and add it to vocabulary."""
        if not word:
            return
            
        # Categorize the word
        category = self._categorize_word(word)
        
        # Add to vocabulary if it's new
        if word.lower() not in self.vocabulary[category]:
            self.vocabulary[category].add(word.lower())
            self.word_frequency[word.lower()] = 1
            
            # Update metrics
            self.metrics['vocabulary_size'] = sum(len(words) for words in self.vocabulary.values())
            
            # Progress stage if vocabulary size meets threshold
            self._check_stage_progression()

    def learn_concept(self, concept: str):
        """Learn a new concept."""
        # For now, we'll treat concepts as compound words/phrases
        words = concept.split()
        for word in words:
            self.learn_word(word)

    def _check_stage_progression(self):
        """Check and update language development stage based on metrics."""
        vocab_size = self.metrics['vocabulary_size']
        
        # Stage progression thresholds
        if self.current_stage == LanguageStage.PRELINGUISTIC and vocab_size >= 10:
            self.current_stage = LanguageStage.HOLOPHRASTIC
            self.stage_progress = 0.0
        elif self.current_stage == LanguageStage.HOLOPHRASTIC and vocab_size >= 50:
            self.current_stage = LanguageStage.TELEGRAPHIC
            self.stage_progress = 0.0
        elif self.current_stage == LanguageStage.TELEGRAPHIC and vocab_size >= 100:
            self.current_stage = LanguageStage.MULTIWORD
            self.stage_progress = 0.0
        elif self.current_stage == LanguageStage.MULTIWORD and vocab_size >= 200:
            self.current_stage = LanguageStage.COMPLEX
            self.stage_progress = 0.0
        
        # Update stage progress
        stage_ranges = {
            LanguageStage.PRELINGUISTIC: (0, 10),
            LanguageStage.HOLOPHRASTIC: (10, 50),
            LanguageStage.TELEGRAPHIC: (50, 100),
            LanguageStage.MULTIWORD: (100, 200),
            LanguageStage.COMPLEX: (200, 500)
        }
        
        current_range = stage_ranges[self.current_stage]
        self.stage_progress = (vocab_size - current_range[0]) / (current_range[1] - current_range[0])
        self.stage_progress = min(max(self.stage_progress, 0.0), 1.0) 