# Interactions

Stores conversation history and interaction patterns.

## Network Visualization
This section is connected to:
- [[Memories/README|Memories]] - Interactions create memories
- [[Emotional_States/README|Emotional States]] - Emotional responses
- [[Language_Learning/README|Language Learning]] - Learning from interaction

## Recent Updates
```dataview
TABLE created as "Time", tags as "Tags"
FROM "#interactions"
SORT created DESC
LIMIT 5
```

## Network Graph
```dataview
TABLE WITHOUT ID
  file.link as "Entry",
  tags as "Tags",
  created as "Time"
FROM "#interactions"
SORT created DESC
```
