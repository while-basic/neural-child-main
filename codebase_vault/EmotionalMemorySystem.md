# Emotional Memory System

The `EmotionalMemorySystem` implements a sophisticated emotional memory system using ChromaDB for storing and retrieving emotionally-contextualized memories.

## Core Components

- [[EmotionalMemoryEntry]] - Structure for emotional memories
- [[EmotionalAssociation]] - Types of emotional associations
- [[EmotionalMemoryProcessor]] - Neural processing of emotional memories

## Memory Types

The system manages three types of collections:
- [[EmotionalMemories]] - Memories with strong emotional content
- [[EpisodicMemories]] - Event-based memories
- [[SemanticMemories]] - Factual knowledge

## Architecture

```python
class EmotionalMemorySystem:
    def __init__(self, persist_dir: str = "emotional_memories"):
        # Initialize ChromaDB collections
        self.emotional_collection = self.chroma_client.get_or_create_collection(
            name="emotional_memories",
            metadata={"type": "emotional"}
        )
        self.episodic_collection = ...
        self.semantic_collection = ...
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
```

## Key Features

- Emotional state tracking
- Memory consolidation
- Temporal relevance
- Developmental relevance
- Emotional similarity matching
- Memory retrieval with context

## Connected Components

- Integrates with [[IntegratedBrain]] for brain state
- Connects to [[DecisionNetwork]] for memory-based decisions
- Links to [[LanguageDevelopment]] for language context
- Uses [[RAGMemorySystem]] for enhanced retrieval

## Implementation Details

Located in [[emotional_memory_system.py]], the system implements:
- Memory storage and retrieval
- Emotional state processing
- Memory consolidation
- Statistical tracking
- State persistence 