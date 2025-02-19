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
   - LM Studio (for running the local LLM)
   - PyTorch with CUDA support
   - Additional dependencies (specified in requirements.txt)

### Setup Steps

1. **Environment Setup**:
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd neural-child

   # Install dependencies
   pip install -r requirements.txt

   # Install PyTorch with CUDA support
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```

2. **LM Studio Setup**:
   - Download and install LM Studio from https://lmstudio.ai/
   - Download the Qwen2.5-7b-instruct model in LM Studio
   - Start the local server in LM Studio
   - Ensure the server is running at `http://localhost:1234`

### Running the System

1. **Start LM Studio**:
   - Launch LM Studio
   - Load the Qwen2.5-7b-instruct model
   - Start the local server
   - Verify the server is running at http://localhost:1234

2. **Run the System**:
   ```bash
   python main.py
   ```

   The system will:
   - Test LLM connection
   - Initialize the Digital Child
   - Begin the developmental process
   - Save progress automatically

### Performance Notes

- The system performs initial LLM connection testing before starting
- Response times may vary based on:
  - GPU capabilities
  - LLM model size
  - System memory
  - Network conditions
- Memory consolidation occurs:
  - Every 24 hours of runtime
  - When memory usage exceeds 90%

### Troubleshooting

1. **LLM Connection Issues**:
   - Ensure LM Studio is running
   - Verify the server URL is correct
   - Check if the model is properly loaded
   - Default timeout is 30 seconds

2. **CUDA Issues**:
   - Verify CUDA toolkit installation
   - Ensure PyTorch is installed with CUDA support
   - Check GPU memory usage
   - Monitor system resources

3. **Memory Management**:
   - System automatically manages memory
   - Consolidation triggers at 90% usage
   - Batch size adjusts automatically

## Recent Changes

### Version 1.1.0
- Added LLM connection testing
- Improved error handling for LLM responses
- Reduced default timeout to 30 seconds
- Disabled streaming by default for better reliability
- Added structured response validation
- Improved memory management
- Enhanced error reporting
