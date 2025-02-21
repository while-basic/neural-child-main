# obsidian_heartbeat_logger.py
# Description: Obsidian integration for heartbeat logging and memory storage
# Created by: Christopher Celaya

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from obsidian_api import ObsidianAPI

class ObsidianHeartbeatLogger:
    def __init__(self, vault_path: str, api: ObsidianAPI):
        """Initialize the Obsidian heartbeat logger.
        
        Args:
            vault_path (str): Path to Obsidian vault
            api (ObsidianAPI): Instance of ObsidianAPI
        """
        self.vault_path = Path(vault_path)
        self.api = api
        
        # Ensure necessary directories exist
        self.heartbeat_dir = self.vault_path / "Heartbeat"
        self.memory_dir = self.vault_path / "Memories"
        self.stats_dir = self.vault_path / "Statistics"
        
        for directory in [self.heartbeat_dir, self.memory_dir, self.stats_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def log_heartbeat_event(self, 
                          event_data: Dict[str, Any],
                          emotional_context: Optional[Dict[str, float]] = None,
                          memory_context: Optional[Dict[str, Any]] = None):
        """Log a heartbeat event with context to Obsidian.
        
        Args:
            event_data (Dict[str, Any]): Heartbeat event data
            emotional_context (Dict[str, float], optional): Emotional state context
            memory_context (Dict[str, Any], optional): Memory context if triggered by memory
        """
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Create daily note if it doesn't exist
        daily_note_path = self.heartbeat_dir / f"Heartbeat-{date_str}.md"
        
        # Format heartbeat entry
        entry_lines = [
            f"## Heartbeat Event at {time_str}",
            "",
            f"- Current Rate: {event_data['current_rate']:.1f} BPM",
            f"- State: {event_data['state']}",
            ""
        ]
        
        if emotional_context:
            entry_lines.extend([
                "### Emotional Context",
                "",
                "```json",
                json.dumps(emotional_context, indent=2),
                "```",
                ""
            ])
            
        if memory_context:
            entry_lines.extend([
                "### Memory Context",
                "",
                f"- Content: {memory_context.get('content', 'N/A')}",
                f"- Valence: {memory_context.get('valence', 0):.2f}",
                f"- Intensity: {memory_context.get('intensity', 0):.2f}",
                ""
            ])
            
        # Add metadata for linking
        entry_lines.extend([
            "---",
            "tags: [heartbeat, vitals]",
            f"date: {date_str}",
            f"time: {time_str}",
            "---",
            ""
        ])
        
        # Write or append to daily note
        if daily_note_path.exists():
            with open(daily_note_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "\n".join(entry_lines))
        else:
            with open(daily_note_path, 'w', encoding='utf-8') as f:
                f.write("# Daily Heartbeat Log\n\n" + "\n".join(entry_lines))
                
    def log_memory_with_heartbeat(self,
                                memory_data: Dict[str, Any],
                                heartbeat_data: Dict[str, Any]):
        """Log a memory entry with associated heartbeat data.
        
        Args:
            memory_data (Dict[str, Any]): Memory data to log
            heartbeat_data (Dict[str, Any]): Associated heartbeat data
        """
        timestamp = datetime.now()
        memory_id = f"memory_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create memory note
        memory_note_path = self.memory_dir / f"{memory_id}.md"
        
        # Format memory entry
        entry_lines = [
            f"# Memory Entry: {memory_id}",
            "",
            "## Content",
            "",
            memory_data.get('content', 'No content provided'),
            "",
            "## Emotional Context",
            "",
            "```json",
            json.dumps(memory_data.get('emotional_state', {}), indent=2),
            "```",
            "",
            "## Heartbeat Response",
            "",
            f"- Initial Rate: {heartbeat_data.get('initial_rate', 'N/A')} BPM",
            f"- Peak Rate: {heartbeat_data.get('peak_rate', 'N/A')} BPM",
            f"- Final Rate: {heartbeat_data.get('current_rate', 'N/A')} BPM",
            f"- State: {heartbeat_data.get('state', 'N/A')}",
            "",
            "---",
            "tags: [memory, heartbeat-response]",
            f"created: {timestamp.isoformat()}",
            f"memory_id: {memory_id}",
            "---"
        ]
        
        # Write memory note
        with open(memory_note_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(entry_lines))
            
    def update_statistics(self, heartbeat_history: List[Dict[str, Any]]):
        """Update heartbeat statistics in Obsidian.
        
        Args:
            heartbeat_history (List[Dict[str, Any]]): List of heartbeat events
        """
        if not heartbeat_history:
            return
            
        # Calculate statistics
        rates = [event['rate'] for event in heartbeat_history]
        states = [event['state'] for event in heartbeat_history]
        
        stats = {
            'average_rate': sum(rates) / len(rates),
            'min_rate': min(rates),
            'max_rate': max(rates),
            'state_distribution': {state: states.count(state) / len(states)
                                 for state in set(states)}
        }
        
        # Update statistics file
        stats_file = self.stats_dir / "heartbeat_statistics.md"
        
        stats_lines = [
            "# Heartbeat Statistics",
            "",
            f"Last Updated: {datetime.now().isoformat()}",
            "",
            "## Overall Statistics",
            "",
            f"- Average Rate: {stats['average_rate']:.1f} BPM",
            f"- Minimum Rate: {stats['min_rate']:.1f} BPM",
            f"- Maximum Rate: {stats['max_rate']:.1f} BPM",
            "",
            "## State Distribution",
            "",
            "```json",
            json.dumps(stats['state_distribution'], indent=2),
            "```",
            "",
            "---",
            "tags: [statistics, heartbeat, analysis]",
            "---"
        ]
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(stats_lines))
            
    def create_heartbeat_graph(self, 
                             history: List[Dict[str, Any]], 
                             time_range: str = "24h"):
        """Create a graph of heartbeat data in Obsidian.
        
        Args:
            history (List[Dict[str, Any]]): Heartbeat history data
            time_range (str): Time range for the graph
        """
        # Format data for Obsidian graph
        data_points = [
            {
                'time': event['timestamp'],
                'rate': event['rate'],
                'state': event['state']
            }
            for event in history
        ]
        
        # Create graph note
        graph_file = self.heartbeat_dir / f"heartbeat_graph_{time_range}.md"
        
        graph_lines = [
            f"# Heartbeat Graph ({time_range})",
            "",
            "```chart",
            "type: line",
            "labels: [Time, Heart Rate]",
            "series:",
            "  - title: BPM",
            "    data: [" + ", ".join(str(point['rate']) for point in data_points) + "]",
            "```",
            "",
            "## State Timeline",
            "",
            "| Time | Rate | State |",
            "|------|------|--------|",
        ]
        
        # Add data rows
        for point in data_points:
            graph_lines.append(
                f"| {point['time']} | {point['rate']:.1f} | {point['state']} |"
            )
            
        graph_lines.extend([
            "",
            "---",
            "tags: [heartbeat, graph, visualization]",
            f"time_range: {time_range}",
            "---"
        ])
        
        with open(graph_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(graph_lines)) 