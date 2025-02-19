# Neural Child Development System

A sophisticated AI system that simulates child development through neural networks, implementing psychological growth stages and emotional intelligence.

## Overview

The Neural Child Development System implements a novel approach to AI by simulating human psychological development from infancy to maturity. The system consists of two main components:

1. **Digital Child**: An AI agent that develops through various psychological stages
2. **Mother LLM**: A nurturing AI system that provides stimuli and guidance for development

## Core Components

### Digital Child Architecture

- **Dynamic Neural Child Brain**: Core neural network for processing and learning
- **Differentiable Memory**: Advanced memory system for experience storage and retrieval
- **Moral Policy Network**: System for ethical decision making
- **Metacognition System**: Self-awareness and reflection capabilities
- **Developmental System**: Manages progression through developmental stages
- **Autonomous Trainer**: Handles learning and adaptation

### Mother LLM Features

- Stage-specific prompting system
- Emotional vector generation
- Conversation history tracking
- Structured feedback generation
- Developmental guidance

## Developmental Stages

The system implements a comprehensive stage-based development model:

1. Newborn (0-3 months)
2. Early Infancy (3-6 months)
3. Late Infancy (6-12 months)
4. Early Toddler (12-18 months)
5. Late Toddler (18-24 months)
6. Early Preschool (2-3 years)
7. Late Preschool (3-4 years)
8. Early Childhood (4-5 years)
9. Middle Childhood (5-6 years)
10. Late Childhood (6-7 years)
11. Early Elementary (7-8 years)
12. Middle Elementary (8-9 years)
13. Late Elementary (9-11 years)
14. Early Adolescence (11-13 years)
15. Middle Adolescence (13-15 years)
16. Late Adolescence (15-18 years)
17. Young Adult (18-21 years)
18. Mature Adult (21+ years)

## Emotional Processing

The system implements a 4-dimensional emotional state vector:
- Joy
- Trust
- Fear
- Surprise

Each emotion is processed with:
- Dynamic updating based on mother's input
- Random variation for natural development
- Emotional state expression
- Memory integration

## Technical Requirements

### Hardware
- CUDA-capable GPU
- Minimum 16GB RAM
- SSD Storage
- Multi-core processor

### Software
- Python 3.8+
- CUDA Toolkit 11.8+
- Ollama (for running the local LLM)
- PyTorch with CUDA support
- Additional dependencies (specified in requirements.txt)

## Usage

1. Initialize the system:
```python
from main import DigitalChild

child = DigitalChild()
```

2. The system will automatically:
- Generate appropriate stimuli through the Mother LLM
- Process child responses
- Update emotional states
- Facilitate learning and development
- Save progress periodically

## Development State

The system automatically tracks:
- Learning loss
- Memory usage
- Moral scores
- Psychological metrics:
  - Attachment styles
  - Defense mechanisms
  - Theory of mind development

## Saving and Loading

The system automatically saves the child's state when interrupted:
```python
torch.save(child.brain.state_dict(), f"digital_child_{child.age()}mo.pth")
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{neural_child_development,
  title = {Neural Child Development System},
  year = {2024},
  author = {[Author Name]},
  url = {[Repository URL]},
  note = {An AI system implementing psychological development and emotional intelligence}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Build Instructions

### Prerequisites

1. **Hardware Requirements**:
   - CUDA-capable GPU
   - Minimum 16GB RAM
   - SSD Storage
   - Multi-core processor

2. **Software Requirements**:
   - Python 3.8+
   - CUDA Toolkit 11.8+
   - Ollama (for running the local LLM)
   - PyTorch with CUDA support
   - Additional dependencies (specified in requirements.txt)

### Setup Steps

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure Ollama is running with llama2 and llama3 models

## Recent Test Results
- Successfully implemented mother-child conversation system
- Emotional state tracking and regulation functioning
- Memory consolidation working with proper error handling
- Developmental stage progression tracking operational
- Telemetry and visualization systems active
- Periodic conversation and checkpoint saving implemented

## Known Issues
- Matrix multiplication errors in learning step (being addressed)
- Occasional JSON parsing errors in LLM responses
- Memory usage spikes during extended runs

## Next Steps
1. Optimize matrix dimensions in learning module
2. Enhance JSON response parsing robustness
3. Implement more sophisticated memory management
4. Add advanced conversation analysis features
5. Develop better error recovery mechanisms

## Directory Structure
```
neural-child/
├── main.py              # Main system controller
├── conversation_system.py # Mother-child interaction handler
├── logger.py            # Comprehensive logging system
├── visualization.py     # Real-time visualization
├── checkpoints/         # Model checkpoints
├── conversations/       # Saved conversations
└── logs/               # System logs
```

## Usage
Run the main script to start the system:
```bash
python main.py
```

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### Version 1.2.0
- Migrated from LM Studio to Ollama
- Updated API endpoint and payload format
- Improved JSON response parsing
- Enhanced error handling for Ollama responses
- Added structured output support for Ollama
- Updated documentation for Ollama setup
