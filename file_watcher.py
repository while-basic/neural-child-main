# file_watcher.py
# Description: File watcher for automatic Obsidian documentation updates
# Created by: Christopher Celaya

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from typing import Set
from code_tracker import CodeTracker

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, code_tracker: CodeTracker):
        """Initialize the code change handler.
        
        Args:
            code_tracker (CodeTracker): Instance of CodeTracker
        """
        self.code_tracker = code_tracker
        self.last_modified: Set[str] = set()
        
    def on_modified(self, event):
        """Handle file modification events.
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix == '.py':
            # Avoid duplicate events
            if str(file_path) in self.last_modified:
                return
                
            self.last_modified.add(str(file_path))
            print(f"Detected change in {file_path}")
            
            # Track changes
            self.code_tracker.track_changes(str(file_path))
            
            # Remove from modified set after a delay
            time.sleep(1)
            self.last_modified.remove(str(file_path))

class FileWatcher:
    def __init__(self, path: str, vault_path: str = "/main_vault"):
        """Initialize the file watcher.
        
        Args:
            path (str): Path to watch for changes
            vault_path (str): Path to Obsidian vault
        """
        self.path = path
        self.code_tracker = CodeTracker(vault_path)
        self.event_handler = CodeChangeHandler(self.code_tracker)
        self.observer = Observer()
        
    def start(self):
        """Start watching for file changes."""
        print(f"Starting file watcher on {self.path}")
        self.observer.schedule(self.event_handler, self.path, recursive=True)
        self.observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            print("\nStopping file watcher...")
            
        self.observer.join()
        
    def initial_scan(self):
        """Perform initial scan of all Python files."""
        print("Performing initial scan of Python files...")
        for file_path in Path(self.path).rglob("*.py"):
            print(f"Scanning {file_path}")
            self.code_tracker.track_changes(str(file_path))
        print("Initial scan complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python file_watcher.py <path_to_watch> [vault_path]")
        sys.exit(1)
        
    path_to_watch = sys.argv[1]
    vault_path = sys.argv[2] if len(sys.argv) > 2 else "/main_vault"
    
    watcher = FileWatcher(path_to_watch, vault_path)
    watcher.initial_scan()
    watcher.start() 