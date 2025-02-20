# obsidian_visualizer.py
# Description: Visualize Obsidian notes and their connections
# Created by: Christopher Celaya

from obsidian_api import ObsidianAPI
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set
import re
import json
from pathlib import Path
import os

class ObsidianVisualizer:
    def __init__(self, vault_path: str):
        """Initialize the visualizer with local vault path"""
        self.api = ObsidianAPI(vault_path=vault_path)
        self.graph = nx.Graph()
        
    def extract_links(self, content: str) -> Set[str]:
        """Extract wiki-style links from content"""
        # Match both [[Link]] and [[Link|Alias]] formats
        links = re.findall(r'\[\[(.*?)(?:\|.*?)?\]\]', content)
        return set(links)
        
    def build_graph(self):
        """Build a graph of notes and their connections"""
        try:
            # Get all files in the vault
            files = self.api.list_vault_files()
            print(f"Found {len(files)} files in vault")
            
            # Process each markdown file
            for file in files:
                try:
                    # Get file content
                    content = self.api.get_vault_file(file)
                    
                    # Add node for this file
                    node_name = Path(file).stem
                    self.graph.add_node(node_name)
                    
                    # Extract and add links
                    links = self.extract_links(content)
                    for link in links:
                        self.graph.add_edge(node_name, link)
                        
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue
            
            print(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            print(f"Error building graph: {str(e)}")
    
    def visualize(self, output_file: str = "obsidian_graph.png"):
        """Create a visualization of the note graph"""
        try:
            if self.graph.number_of_nodes() == 0:
                print("No nodes in graph to visualize")
                return
                
            # Create the plot
            plt.figure(figsize=(15, 10))
            
            # Calculate node sizes based on connections
            node_sizes = [3000 * (1 + self.graph.degree(node)) for node in self.graph.nodes()]
            
            # Calculate node colors based on number of connections
            node_colors = [self.graph.degree(node) for node in self.graph.nodes()]
            
            # Create the layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Draw the graph
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                width=0.5,
                alpha=0.7,
                cmap=plt.cm.viridis
            )
            
            # Save the visualization
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_file}")
            
            # Generate statistics
            stats = {
                'total_notes': self.graph.number_of_nodes(),
                'total_connections': self.graph.number_of_edges(),
                'most_connected': sorted(
                    [(node, self.graph.degree(node)) for node in self.graph.nodes()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5] if self.graph.number_of_nodes() > 0 else []
            }
            
            # Save statistics
            stats_file = Path(output_file).with_suffix('.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics to {stats_file}")
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    def analyze_connections(self) -> Dict:
        """Analyze the connections between notes"""
        try:
            if self.graph.number_of_nodes() == 0:
                print("No nodes in graph to analyze")
                return {}
                
            analysis = {
                'most_connected': [],
                'isolated_notes': [],
                'clusters': [],
                'avg_connections': 0
            }
            
            # Find most connected notes
            analysis['most_connected'] = sorted(
                [(node, self.graph.degree(node)) for node in self.graph.nodes()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Find isolated notes
            analysis['isolated_notes'] = [
                node for node in self.graph.nodes()
                if self.graph.degree(node) == 0
            ]
            
            # Find clusters/communities
            if self.graph.number_of_nodes() > 0:
                communities = list(nx.community.greedy_modularity_communities(self.graph))
                analysis['clusters'] = [
                    [node for node in community]
                    for community in communities
                ]
            
            # Calculate average connections
            if self.graph.number_of_nodes() > 0:
                analysis['avg_connections'] = sum(
                    self.graph.degree(node) for node in self.graph.nodes()
                ) / self.graph.number_of_nodes()
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing connections: {str(e)}")
            return {}

def main():
    # Path to the codebase vault
    VAULT_PATH = "codebase_vault"
    
    # Create visualizer
    visualizer = ObsidianVisualizer(VAULT_PATH)
    
    # Build the graph
    print("Building graph of notes...")
    visualizer.build_graph()
    
    # Create visualization
    print("\nCreating visualization...")
    visualizer.visualize()
    
    # Analyze connections
    print("\nAnalyzing connections...")
    analysis = visualizer.analyze_connections()
    
    # Print analysis
    if analysis:
        print("\nAnalysis Results:")
        print(f"Most connected notes:")
        for note, connections in analysis['most_connected']:
            print(f"  - {note}: {connections} connections")
            
        print(f"\nIsolated notes: {len(analysis['isolated_notes'])}")
        if analysis['isolated_notes']:
            print("Isolated notes:")
            for note in analysis['isolated_notes']:
                print(f"  - {note}")
                
        print(f"\nNumber of clusters: {len(analysis['clusters'])}")
        if analysis['clusters']:
            print("Clusters:")
            for i, cluster in enumerate(analysis['clusters'], 1):
                print(f"\nCluster {i} ({len(cluster)} notes):")
                for note in cluster:
                    print(f"  - {note}")
                    
        print(f"\nAverage connections per note: {analysis['avg_connections']:.2f}")

if __name__ == "__main__":
    main() 