# Decision Network

The `DecisionNetwork` class implements a sophisticated decision-making system that considers conversation context, emotional state, memory, and developmental stage.

## Core Components

- [[ConversationEncoder]] - LSTM-based conversation processing with attention
- [[EmotionProcessor]] - Emotional state integration
- [[MemoryProcessor]] - Memory context processing
- [[StageProcessor]] - Developmental stage adaptation

## Architecture

```python
class DecisionNetwork(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_actions=4):
        # Conversation encoder with attention
        self.conversation_encoder = ConversationEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # Processing components
        self.emotion_processor = nn.Sequential(...)
        self.memory_processor = nn.Sequential(...)
        self.stage_processor = nn.Sequential(...)
        
        # Decision layers
        self.decision_layers = nn.Sequential(...)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.confidence_head = nn.Linear(hidden_dim, 1)
```

## Key Features

- Multi-head attention for conversation processing
- Emotional state integration
- Memory-based decision making
- Stage-appropriate responses
- Confidence estimation
- Learning from feedback

## Connected Components

- Integrates with [[IntegratedBrain]] for brain state
- Uses [[EmotionalMemorySystem]] for emotional context
- Connects to [[LanguageDevelopment]] for language processing
- Interfaces with [[RAGMemorySystem]] for memory retrieval

## Implementation Details

Located in [[decision_network.py]], the network implements:
- Forward pass with attention
- Feedback-based learning
- Decision metrics tracking
- State saving/loading 