# RAG Memory System

The `RAGMemorySystem` implements a Retrieval-Augmented Generation memory system that enhances memory retrieval with semantic search and emotional context.

## Core Components

- [[MemoryContext]] - Context for memory operations
- [[ChromaDB]] - Vector database for memory storage
- [[SentenceTransformer]] - Semantic embedding generation

## Memory Types

The system manages different types of memories:
- [[EmotionalMemories]] - Emotional context and state
- [[EpisodicMemories]] - Event sequences
- [[SemanticMemories]] - Knowledge and facts

## Architecture

```python
class RAGMemorySystem:
    def __init__(self, persist_dir: str = "rag_memories"):
        # Initialize ChromaDB collections
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
```

## Key Features

- Semantic search
- Emotional context integration
- Temporal relevance
- Developmental relevance
- Memory consolidation
- Statistical tracking

## Memory Retrieval

The system uses multiple factors for retrieval:
- Semantic similarity
- Emotional similarity
- Temporal proximity
- Developmental stage
- Memory type relevance

## Connected Components

- Integrates with [[IntegratedBrain]] for brain state
- Links to [[DecisionNetwork]] for memory-based decisions
- Connects to [[EmotionalMemorySystem]] for emotional context
- Supports [[LanguageDevelopment]] with knowledge retrieval

## Implementation Details

Located in [[rag_memory.py]], the system implements:
- Memory storage and retrieval
- Context processing
- Relevance scoring
- Memory consolidation
- State persistence 