# Neural Child Development System

A sophisticated AI system that simulates child development with emotional awareness, memory formation, and physiological responses.

## Features

### Core Systems
- Integrated brain architecture with multiple neural components
- Q-Learning based decision making
- Developmental stages progression
- Language development system
- Memory management with ChromaDB
- Emotional regulation system

### New Features
- **Heartbeat System**
  - Real-time heartbeat simulation based on emotional state
  - Five distinct heart rate states (Resting, Elevated, Anxious, Focused, Relaxed)
  - Neural network-based heart rate modulation
  - Historical heartbeat tracking and analysis

- **Emotional Chat System**
  - Message tone analysis
  - Emotional state tracking
  - Integration with heartbeat responses
  - Automatic memory recording in Obsidian
  - Special memory commands:
    - `!remember` - Store explicit memories
    - `!forget` - Mark memories as forgotten
    - `!reflect` - Analyze emotional patterns

### Memory Integration
- Automatic recording of significant emotional interactions
- Integration with Obsidian for memory storage
- Emotional pattern analysis and reflection
- Memory impact on physiological responses

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-child.git
cd neural-child
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Obsidian vault:
- Create a new Obsidian vault
- Configure the vault path in your environment variables:
```bash
export OBSIDIAN_VAULT_PATH="/path/to/your/vault"
```

## Obsidian Integration

The system now includes automatic code documentation in Obsidian. This feature:

- Tracks code changes in real-time
- Creates detailed documentation for each Python file
- Generates dependency graphs
- Maintains connections between related code components
- Stores metadata about code structure

### Setting up Obsidian Integration

1. Install Obsidian from https://obsidian.md/
2. Create or specify a vault directory using the `--vault-path` argument
3. Start the file watcher:
```bash
python file_watcher.py . /path/to/vault
```

The system will:
- Create a `Code` directory in your Obsidian vault
- Generate documentation for all Python files
- Update documentation automatically when files change
- Create a dependency graph showing relationships between files

### Documentation Structure

Each Python file gets its own note in Obsidian with:
- File structure (classes, methods, etc.)
- Dependencies and imports
- Code connections and relationships
- Metadata and tags
- Last modified timestamp

The dependency graph shows:
- File relationships
- Import connections
- Class inheritance
- Module dependencies

## Usage

### Basic Interaction
```python
from digital_child import DigitalChild
from obsidian_api import ObsidianAPI

# Initialize systems
obsidian_api = ObsidianAPI(vault_path="path/to/vault")
child = DigitalChild()
chat_system = EmotionalChatSystem(child.brain, obsidian_api)

# Process a message
response = chat_system.process_message("Hello! I'm happy to meet you!")
print(f"Heartbeat: {response['heartbeat']['current_rate']} BPM")
print(f"Emotional State: {response['brain_state']['emotional_valence']}")
```

### Memory Commands
```python
# Store a memory
chat_system.process_message("!remember I learned to count to 10 today!")

# Reflect on emotional patterns
chat_system.process_message("!reflect")
```

## Development

### Project Structure
```
neural-child/
├── main.py                     # Main application file
├── heartbeat_system.py         # Heartbeat simulation system
├── emotional_chat_system.py    # Emotional chat processing
├── integrated_brain.py         # Core brain architecture
├── memory_system/             # Memory management
│   ├── emotional_memory.py    # Emotional memory processing
│   └── obsidian_api.py       # Obsidian integration
└── utils/
    ├── logger.py             # Development logger
    └── visualizer.py        # State visualization
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- Christopher Celaya

## Acknowledgments
- Thanks to the open-source community
- Special thanks to contributors and testers
