# language_development.py
# Description: Language development module for neural child development
# Created by: Christopher Celaya

import torch
import torch.nn as nn
import json
import re
from collections import defaultdict
from enum import Enum, auto
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import spacy
from transformers import pipeline
from llm_module import chat_completion
from developmental_stages import DevelopmentalStage

class LanguageStage(Enum):
    """Language development stages"""
    PRELINGUISTIC = auto()    # 0-12 months
    HOLOPHRASTIC = auto()     # 12-18 months
    TELEGRAPHIC = auto()      # 18-24 months
    FUNCTIONAL = auto()         # 2-3 years
    FLUENT = auto()           # 7+ years
    ADVANCED = auto()          # 5-7 years

class WordCategory(Enum):
    """Categories for vocabulary tracking"""
    NOUNS = auto()
    VERBS = auto()
    ADJECTIVES = auto()
    PRONOUNS = auto()
    PREPOSITIONS = auto()
    ARTICLES = auto()
    CONJUNCTIONS = auto()
    ADVERBS = auto()

class LanguageDevelopment(nn.Module):
    def __init__(self, device='cpu'):
        """Initialize the language development system"""
        super().__init__()
        self.device = device
        
        # Initialize NLP components first
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = pipeline("sentiment-analysis", device=-1)  # Use CPU
        
        # Initialize language development state
        self.current_stage = LanguageStage.PRELINGUISTIC
        self.stage_progress = 0.0
        
        # Initialize vocabulary
        self.vocabulary: Dict[str, Set[str]] = {
            'nouns': set(),
            'verbs': set(),
            'adjectives': set(),
            'social': set(),  # Greetings, social phrases
            'emotions': set(), # Emotion words
            'understood': set(),  # Words the child understands
            'spoken': set(),      # Words the child can speak
            'complex': set()      # Complex words/phrases
        }
        
        # Track known words
        self.known_words: Set[str] = set()
        
        # Track word usage frequency
        self.word_frequency: Dict[str, int] = {}
        
        # Initialize linguistic rules
        self._initialize_linguistic_rules()
        
        # Stage-specific utterance patterns
        self.utterance_patterns = {
            LanguageStage.HOLOPHRASTIC: ["MAMA!", "DADA!", "BALL!", "UP!", "NO!"],
            LanguageStage.TELEGRAPHIC: ["WANT MILK", "MORE PLAY", "BIG BALL", "DADDY GO"],
            LanguageStage.FUNCTIONAL: ["I WANT MILK", "PLAY WITH ME", "MOMMY READ BOOK"],
            LanguageStage.FLUENT: ["Can I have milk please?", "I want to play with my toys."],
            LanguageStage.ADVANCED: ["I would appreciate if you could help me with this task.", 
                                   "The metaphorical implications of this story are fascinating."],
            LanguageStage.FLUENT: ["The intricate interplay between syntax and semantics reveals deeper linguistic patterns.",
                                   "Let's analyze the sociolinguistic implications of this discourse."]
        }
        
        # Initialize word emotions
        self.word_emotions = defaultdict(lambda: torch.zeros(4))  # [joy, trust, fear, surprise]
        
        # Initialize metrics
        self.metrics = {
            'vocabulary_size': 0,
            'grammar_complexity': 0.0,
            'comprehension_level': 0.0,
            'expression_level': 0.0,
            'utterance_length': 0.0,  # Average length of utterances
            'pronunciation_accuracy': 0.3,  # Start with basic pronunciation accuracy
            'total_interactions': 0,
            'unique_words_used': 0,
            'complex_sentences': 0,
            'grammar_score': 0.0
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
        
        # Development tracking
        self.development_history = []
        
        # Initialize basic vocabulary
        self._initialize_basic_vocabulary()
        
        # Initialize stage mapping
        self.stage_to_language = {
            DevelopmentalStage.NEWBORN: LanguageStage.HOLOPHRASTIC,
            DevelopmentalStage.INFANT: LanguageStage.HOLOPHRASTIC,
            DevelopmentalStage.EARLY_TODDLER: LanguageStage.TELEGRAPHIC,
            DevelopmentalStage.LATE_TODDLER: LanguageStage.TELEGRAPHIC,
            DevelopmentalStage.EARLY_PRESCHOOL: LanguageStage.FUNCTIONAL,
            DevelopmentalStage.LATE_PRESCHOOL: LanguageStage.FUNCTIONAL,
            DevelopmentalStage.EARLY_CHILDHOOD: LanguageStage.FLUENT,
            DevelopmentalStage.MIDDLE_CHILDHOOD: LanguageStage.ADVANCED,
            DevelopmentalStage.LATE_CHILDHOOD: LanguageStage.FLUENT,
            DevelopmentalStage.PRE_ADOLESCENT: LanguageStage.ADVANCED,
            DevelopmentalStage.EARLY_TEEN: LanguageStage.FLUENT,
            DevelopmentalStage.MID_TEEN: LanguageStage.FLUENT,
            DevelopmentalStage.LATE_TEEN: LanguageStage.FLUENT,
            DevelopmentalStage.YOUNG_ADULT: LanguageStage.FLUENT,
            DevelopmentalStage.EARLY_TWENTIES: LanguageStage.FLUENT,
            DevelopmentalStage.LATE_TWENTIES: LanguageStage.FLUENT
        }
        
        # Initialize stage requirements
        self.stage_requirements = {
            LanguageStage.PRELINGUISTIC: {
                'vocab_size': 0,
                'interaction_count': 10
            },
            LanguageStage.HOLOPHRASTIC: {
                'vocab_size': 10,
                'interaction_count': 50
            },
            LanguageStage.TELEGRAPHIC: {
                'vocab_size': 50,
                'interaction_count': 100
            },
            LanguageStage.FUNCTIONAL: {
                'vocab_size': 200,
                'interaction_count': 500
            },
            LanguageStage.FLUENT: {
                'vocab_size': 1000,
                'interaction_count': 1000
            },
            LanguageStage.ADVANCED: {
                'vocab_size': 5000,
                'interaction_count': 2000
            }
        }
        
    def initialize_stage_patterns(self):
        """Initialize language patterns for each developmental stage"""
        self.stage_patterns = {
            LanguageStage.HOLOPHRASTIC: {
                'sounds': ['mama', 'dada', 'no', 'more', 'up'],
                'patterns': ['CV'],  # C=consonant, V=vowel
                'intonations': ['rising', 'falling', 'flat']
            },
            LanguageStage.TELEGRAPHIC: {
                'combinations': ['noun+verb', 'adjective+noun', 'verb+object'],
                'meanings': ['action', 'possession', 'location', 'description']
            },
            LanguageStage.FUNCTIONAL: {
                'sentence_patterns': ['subject-verb-object', 'subject-verb'],
                'question_forms': ['what', 'where', 'who'],
                'tenses': ['present']
            },
            LanguageStage.FLUENT: {
                'sentence_types': ['all'],
                'tenses': ['all'],
                'modalities': ['all'],
                'rhetoric': ['all'],
                'discourse': ['all'],
                'registers': ['all'],
                'metalinguistic': ['language analysis', 'linguistic theory'],
                'creativity': ['neologisms', 'wordplay', 'poetry'],
                'expertise': ['technical', 'academic', 'professional']
            },
            LanguageStage.ADVANCED: {
                'sentence_types': ['compound', 'complex', 'compound-complex'],
                'tenses': ['present', 'past', 'future', 'perfect', 'progressive'],
                'modalities': ['can', 'should', 'might', 'must', 'would', 'could'],
                'rhetoric': ['metaphor', 'simile', 'personification'],
                'discourse': ['argumentation', 'exposition', 'narrative'],
                'registers': ['formal', 'informal', 'academic', 'colloquial']
            }
        }
    
    def _initialize_linguistic_rules(self):
        """Initialize linguistic rules for word categorization"""
        self.linguistic_rules = {
            WordCategory.NOUNS: {
                'suffixes': ['ness', 'ment', 'ship', 'dom', 'er', 'or', 'ist'],
                'prefixes': ['super', 'sub', 'inter', 'trans'],
                'patterns': [r'^[A-Z][a-z]+$']  # Proper nouns
            },
            WordCategory.VERBS: {
                'suffixes': ['ate', 'ize', 'ify', 'en', 'ed', 'ing'],
                'prefixes': ['re', 'un', 'dis', 'mis'],
                'patterns': [r'[a-z]+(?:ed|ing|s)$']
            },
            WordCategory.ADJECTIVES: {
                'suffixes': ['ful', 'ous', 'able', 'ible', 'al', 'ive', 'less', 'ish'],
                'prefixes': ['un', 'in', 'im', 'il', 'ir'],
                'patterns': [r'[a-z]+(?:er|est)$']
            },
            WordCategory.ADVERBS: {
                'suffixes': ['ly', 'ward', 'wise'],
                'patterns': [r'[a-z]+ly$']
            }
        }
        
        # Common words by category
        self.common_words = {
            WordCategory.PRONOUNS: {'i', 'me', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'},
            WordCategory.PREPOSITIONS: {'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'of'},
            WordCategory.ARTICLES: {'a', 'an', 'the'},
            WordCategory.CONJUNCTIONS: {'and', 'but', 'because'},
            WordCategory.ADVERBS: {'now', 'more'}
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
            'NOUN': WordCategory.NOUNS,
            'VERB': WordCategory.VERBS,
            'ADJ': WordCategory.ADJECTIVES,
            'ADV': WordCategory.ADVERBS,
            'PRON': WordCategory.PRONOUNS,
            'ADP': WordCategory.PREPOSITIONS,
            'DET': WordCategory.ARTICLES,
            'CONJ': WordCategory.CONJUNCTIONS
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
        return WordCategory.NOUNS
    
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
            enriched_data = chat_completion(enrichment_prompt, structured_output=True)
            
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
            if self.current_stage == LanguageStage.HOLOPHRASTIC:
                self.current_stage = LanguageStage.TELEGRAPHIC
            elif self.current_stage == LanguageStage.TELEGRAPHIC:
                self.current_stage = LanguageStage.FUNCTIONAL
            elif self.current_stage == LanguageStage.FUNCTIONAL:
                self.current_stage = LanguageStage.FLUENT
        
        return {
            'current_stage': self.current_stage.value,
            'stage_progress': self.stage_progress,
            'new_words': list(new_words),
            'metrics': {
                'vocabulary_size': total_vocab_size,
                'nouns': len(self.vocabulary[WordCategory.NOUNS]),
                'verbs': len(self.vocabulary[WordCategory.VERBS]),
                'adjectives': len(self.vocabulary[WordCategory.ADJECTIVES])
            }
        }
    
    def _map_category(self, category_str: str) -> Optional[WordCategory]:
        """Map string category to WordCategory enum"""
        category_map = {
            'noun': WordCategory.NOUNS,
            'verb': WordCategory.VERBS,
            'adjective': WordCategory.ADJECTIVES,
            'adverb': WordCategory.ADVERBS,
            'pronoun': WordCategory.PRONOUNS,
            'preposition': WordCategory.PREPOSITIONS,
            'article': WordCategory.ARTICLES,
            'conjunction': WordCategory.CONJUNCTIONS
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
        
        if self.current_stage == LanguageStage.HOLOPHRASTIC:
            if joy > 0.7 or trust > 0.7:
                return np.random.choice(['MAMA!', 'DADA!'])
            elif fear > 0.7:
                return 'NO!'
            elif surprise > 0.7:
                return np.random.choice(['MAMA!', 'BALL!'])
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
        
        elif self.current_stage == LanguageStage.FUNCTIONAL:
            if is_positive and (joy > 0.6 or trust > 0.6):
                return "I'm feeling very happy!"
            elif fear > 0.6:
                return "I'm a little scared."
            elif 'play' in context.lower():
                return "Can we play together?"
            else:
                return np.random.choice(patterns)
        
        elif self.current_stage == LanguageStage.FLUENT:
            # Mastery stage responses with high sophistication
            context_responses = {
                'philosophy': [
                    "The philosophical implications of that perspective are fascinating.",
                    "Let's explore the ethical dimensions of this concept.",
                    "This raises interesting questions about consciousness and reality."
                ],
                'research': [
                    "The methodology should account for potential confounding variables.",
                    "We should consider incorporating longitudinal data analysis.",
                    "The findings suggest a significant correlation between variables."
                ],
                'presentation': [
                    "Let me articulate the key points of this analysis.",
                    "The data demonstrates a clear trend in our observations.",
                    "We can visualize these results through various analytical frameworks."
                ],
                'literature': [
                    "The author's use of metaphor creates a compelling narrative.",
                    "The thematic elements reflect deeper societal issues.",
                    "This passage exemplifies the writer's distinctive style."
                ]
            }
            
            # Match context to appropriate response category
            for key, responses in context_responses.items():
                if key in context.lower():
                    return np.random.choice(responses)
            
            return np.random.choice(self.utterance_patterns[LanguageStage.ADVANCED])
        
        else:  # ADVANCED stage
            # Mastery stage responses with high sophistication
            context_responses = {
                'philosophy': [
                    "The intersection of epistemology and cognitive science reveals fascinating insights into the nature of consciousness and human understanding.",
                    "We might consider how this philosophical framework aligns with contemporary theories of mind and artificial intelligence."
                ],
                'research': [
                    "The meta-analysis reveals significant heterogeneity in effect sizes across studies, suggesting underlying moderator variables.",
                    "By implementing a mixed-methods approach, we can triangulate our findings and strengthen the validity of our conclusions."
                ],
                'presentation': [
                    "The empirical evidence supports our theoretical model, though we should acknowledge potential limitations in ecological validity.",
                    "Let's examine the implications of these findings through both quantitative and qualitative analytical lenses."
                ],
                'literature': [
                    "The author's intricate weaving of postmodern themes with classical narrative structures creates a compelling dialectic.",
                    "The text's intertextual references and metalinguistic elements contribute to its rich interpretative possibilities."
                ]
            }
            
            # Match context to appropriate response category
            for key, responses in context_responses.items():
                if key in context.lower():
                    return np.random.choice(responses)
            
            return np.random.choice(self.utterance_patterns[LanguageStage.FLUENT])
    
    def _generate_prelinguistic(self) -> str:
        """Generate prelinguistic sounds (crying, cooing, babbling)"""
        patterns = self.stage_patterns[LanguageStage.HOLOPHRASTIC]
        
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
            suitable_words = self.stage_patterns[LanguageStage.HOLOPHRASTIC]['sounds']
        
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
            noun = self._get_suitable_word(WordCategory.NOUNS, emotional_state)
            verb = self._get_suitable_word(WordCategory.VERBS, emotional_state)
            return f"{noun} {verb}"
        elif pattern == 'adjective+noun':
            adj = self._get_suitable_word(WordCategory.ADJECTIVES, emotional_state)
            noun = self._get_suitable_word(WordCategory.NOUNS, emotional_state)
            return f"{adj} {noun}"
        else:  # verb+object
            verb = self._get_suitable_word(WordCategory.VERBS, emotional_state)
            obj = self._get_suitable_word(WordCategory.NOUNS, emotional_state)
            return f"{verb} {obj}"
    
    def _generate_multiword(self, context: str, emotional_state: torch.Tensor) -> str:
        """Generate simple sentences"""
        pattern = np.random.choice(self.stage_patterns[LanguageStage.FUNCTIONAL]['sentence_patterns'])
        
        if pattern == 'subject-verb-object':
            subj = self._get_suitable_word(WordCategory.NOUNS, emotional_state)
            verb = self._get_suitable_word(WordCategory.VERBS, emotional_state)
            obj = self._get_suitable_word(WordCategory.NOUNS, emotional_state)
            return f"{subj} {verb} {obj}"
        else:  # subject-verb
            subj = self._get_suitable_word(WordCategory.NOUNS, emotional_state)
            verb = self._get_suitable_word(WordCategory.VERBS, emotional_state)
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
                WordCategory.NOUNS: ['mama', 'dada', 'baby'],
                WordCategory.VERBS: ['want', 'see', 'go'],
                WordCategory.ADJECTIVES: ['good', 'bad', 'big'],
                WordCategory.PRONOUNS: ['me', 'you', 'it'],
                WordCategory.PREPOSITIONS: ['up', 'in', 'on'],
                WordCategory.ARTICLES: ['a', 'the'],
                WordCategory.CONJUNCTIONS: ['and', 'but', 'because'],
                WordCategory.ADVERBS: ['now', 'more']
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
            enriched_data = chat_completion(enrichment_prompt, structured_output=True)
            
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
            response = chat_completion(concept_prompt, structured_output=True)
            
            if isinstance(response, dict) and "concepts" in response:
                return response["concepts"]
            
            return []
            
        except Exception as e:
            print(f"Error identifying concepts: {str(e)}")
            return []

    def get_current_stage(self) -> str:
        """Get the current language development stage name."""
        return self.current_stage.name

    def learn_word(self, word: str, category: Optional[str] = None) -> bool:
        """Learn a new word and optionally categorize it"""
        word = word.lower().strip()
        if word in self.known_words:
            return False
            
        self.known_words.add(word)
        if category and category in self.vocabulary:
            self.vocabulary[category].add(word)
            
        self.word_frequency[word] = 0
        return True
        
    def learn_concept(self, concept: str) -> bool:
        """Learn a new concept"""
        concept = concept.lower().strip()
        words = concept.split()
        
        learned = False
        for word in words:
            if self.learn_word(word):
                learned = True
                
        return learned
        
    def get_vocabulary_size(self) -> int:
        """Get total vocabulary size"""
        return len(self.known_words)
        
    def can_understand(self, text: str) -> float:
        """Calculate how well the current stage can understand the text"""
        words = text.lower().split()
        known = sum(1 for word in words if word in self.known_words)
        return known / len(words) if words else 0.0
        
    def can_express(self, text: str) -> bool:
        """Check if the current stage can express this text"""
        if self.current_stage == LanguageStage.PRELINGUISTIC:
            return False
            
        words = text.lower().split()
        if self.current_stage == LanguageStage.HOLOPHRASTIC:
            return len(words) <= 1
            
        if self.current_stage == LanguageStage.TELEGRAPHIC:
            return len(words) <= 3
            
        if self.current_stage == LanguageStage.FUNCTIONAL:
            return len(words) <= 5
            
        return True
        
    def extract_new_words(self, text: str) -> List[str]:
        """Extract potential new words from text"""
        # Tokenize and clean text
        words = text.lower().split()
        words = [word.strip('.,!?()[]{}":;') for word in words]
        
        # Filter out known words and empty strings
        new_words = []
        for word in words:
            if (word and 
                not any(word in words for words in self.vocabulary.values()) and 
                word not in self.known_words and
                len(word) > 1):  # Avoid single characters
                new_words.append(word)
        
        return new_words
        
    def update_word_frequency(self, text: str):
        """Update word usage frequency"""
        words = text.lower().split()
        for word in words:
            if word in self.word_frequency:
                self.word_frequency[word] += 1
                
    def get_most_used_words(self, n: int = 10) -> List[tuple[str, int]]:
        """Get the most frequently used words"""
        return sorted(
            self.word_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

    def _initialize_basic_vocabulary(self):
        """Initialize basic vocabulary based on developmental stage"""
        # Basic social words
        social_words = ['mama', 'dada', 'hi', 'bye', 'no', 'yes']
        for word in social_words:
            self.learn_word(word, 'social')
            
        # Basic emotion words
        emotion_words = ['happy', 'sad', 'love', 'scared']
        for word in emotion_words:
            self.learn_word(word, 'emotions')
            
        # Basic nouns
        basic_nouns = ['milk', 'ball', 'toy', 'bed']
        for word in basic_nouns:
            self.learn_word(word, 'nouns')
            
        # Basic verbs
        basic_verbs = ['want', 'see', 'go', 'sleep']
        for word in basic_verbs:
            self.learn_word(word, 'verbs')
            
        # Basic adjectives
        basic_adjectives = ['big', 'small', 'good', 'bad']
        for word in basic_adjectives:
            self.learn_word(word, 'adjectives') 

    def process_input(self, text: str, emotional_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Process input text and generate appropriate response based on current stage"""
        try:
            # Clean and normalize input
            text = text.strip()
            
            # Extract new words and concepts
            new_words = self.identify_new_words(text)
            new_concepts = self.identify_new_concepts(text)
            
            # Update word frequencies
            self.update_word_frequency(text)
            
            # Calculate comprehension level
            comprehension = self.can_understand(text)
            
            # Generate stage-appropriate response
            if self.current_stage == LanguageStage.PRELINGUISTIC:
                response = self._generate_prelinguistic()
            elif self.current_stage == LanguageStage.HOLOPHRASTIC:
                response = self._generate_holophrastic(emotional_state if emotional_state is not None 
                                                     else torch.tensor([0.5, 0.5, 0.2, 0.3]))
            elif self.current_stage == LanguageStage.TELEGRAPHIC:
                response = self._generate_telegraphic(text, emotional_state if emotional_state is not None 
                                                    else torch.tensor([0.5, 0.5, 0.2, 0.3]))
            elif self.current_stage == LanguageStage.FUNCTIONAL:
                response = self._generate_multiword(text, emotional_state if emotional_state is not None 
                                                  else torch.tensor([0.5, 0.5, 0.2, 0.3]))
            else:  # FLUENT, ADVANCED
                response = self._generate_complex(text, emotional_state if emotional_state is not None 
                                               else torch.tensor([0.5, 0.5, 0.2, 0.3]))
            
            # Update metrics
            self._update_metrics(response)
            
            # Store response in recent utterances
            self.recent_utterances.append(response)
            if len(self.recent_utterances) > 10:
                self.recent_utterances.pop(0)
            
            return {
                'response': response,
                'new_words': new_words,
                'new_concepts': new_concepts,
                'comprehension': comprehension,
                'stage': self.current_stage.name,
                'stage_progress': self.stage_progress,
                'metrics': self.metrics
            }
            
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return {
                'response': "...",  # Default response based on stage
                'error': str(e)
            } 

    def update_stage(self, developmental_stage: DevelopmentalStage):
        """Update language stage based on developmental stage"""
        if developmental_stage in self.stage_to_language:
            self.current_stage = self.stage_to_language[developmental_stage] 

    def process_interaction(self, text: str) -> Dict:
        """Process a language interaction"""
        # Update interaction count
        self.metrics['total_interactions'] += 1
        
        # Extract words and update vocabulary
        words = set(text.lower().split())
        self.vocabulary['understood'].update(words)
        self.metrics['unique_words_used'] = len(self.vocabulary['understood'])
        
        # Calculate progress
        self._update_progress()
        
        return {
            'new_words': len(words - self.vocabulary['understood']),
            'vocab_size': len(self.vocabulary['understood']),
            'stage': self.current_stage,
            'progress': self.stage_progress
        }
        
    def _update_progress(self) -> None:
        """Update language development progress"""
        current_reqs = self.stage_requirements[self.current_stage]
        
        # Calculate progress based on vocabulary and interactions
        vocab_progress = min(1.0, len(self.vocabulary['understood']) / current_reqs['vocab_size']) if current_reqs['vocab_size'] > 0 else 1.0
        interaction_progress = min(1.0, self.metrics['total_interactions'] / current_reqs['interaction_count'])
        
        # Update stage progress
        self.stage_progress = (vocab_progress + interaction_progress) / 2.0
        
        # Check for stage advancement
        if self.stage_progress >= 1.0:
            self._advance_stage()
            
    def _advance_stage(self) -> None:
        """Advance to the next language stage if available"""
        stages = list(LanguageStage)
        current_index = stages.index(self.current_stage)
        
        if current_index < len(stages) - 1:
            self.current_stage = stages[current_index + 1]
            self.stage_progress = 0.0
            
    def get_metrics(self) -> Dict:
        """Get current language development metrics"""
        return {
            'stage': self.current_stage,
            'progress': self.stage_progress,
            'vocabulary_size': len(self.vocabulary['understood']),
            'total_interactions': self.metrics['total_interactions'],
            'grammar_score': self.metrics['grammar_score']
        } 

    def update_metrics(self, text: str):
        """Update metrics based on generated text"""
        # Update total interactions
        self.metrics['total_interactions'] += 1
        
        # Update unique words
        words = set(text.lower().split())
        self.metrics['unique_words_used'] = len(words)
        
        # Update complex sentences
        sentences = text.split('.')
        complex_count = sum(1 for s in sentences if len(s.split()) > 10)
        self.metrics['complex_sentences'] = complex_count
        
        # Calculate grammar score
        doc = self.nlp(text)
        pos_tags = [token.pos_ for token in doc]
        unique_pos = len(set(pos_tags))
        self.metrics['grammar_score'] = min(1.0, unique_pos / 10)  # Normalize by expected max of 10 POS tags 

    def learn_new_word(self, word: str) -> bool:
        """
        Learn a new word and add it to vocabulary
        
        Args:
            word (str): Word to learn
            
        Returns:
            bool: Whether word was successfully learned
        """
        try:
            prompt = f"""
            Help a child learn the following word by providing a simple definition and example:
            
            Word: {word}
            
            Response format:
            Definition: <simple definition>
            Example: <example sentence>
            """
            
            response = chat_completion(prompt)
            
            if response:
                self.vocabulary.add(word.lower())
                return True
                
            return False
            
        except Exception as e:
            print(f"Error learning word: {e}")
            return False

    def expand_vocabulary(self, text: str) -> List[str]:
        """
        Extract new words from text to expand vocabulary
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[str]: List of new words learned
        """
        try:
            prompt = f"""
            Identify 3-5 key words from the following text that would be good for a child to learn:
            
            Text: {text}
            
            Response format:
            word1, word2, word3
            """
            
            response = chat_completion(prompt)
            
            if response:
                new_words = []
                for word in response.split(','):
                    word = word.strip().lower()
                    if word not in self.vocabulary:
                        if self.learn_new_word(word):
                            new_words.append(word)
                return new_words
                
            return []
            
        except Exception as e:
            print(f"Error expanding vocabulary: {e}")
            return []

    def identify_learning_concepts(self, text: str) -> List[str]:
        """
        Identify key learning concepts from text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[str]: List of identified learning concepts
        """
        try:
            prompt = f"""
            Identify key learning concepts from the following text that would be valuable for child development:
            
            Text: {text}
            
            Response format:
            concept1, concept2, concept3
            """
            
            response = chat_completion(prompt)
            
            if response:
                return [c.strip() for c in response.split(',')]
                
            return []
            
        except Exception as e:
            print(f"Error identifying concepts: {e}")
            return [] 