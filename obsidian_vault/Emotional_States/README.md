# Emotional_States

Tracks emotional development and state changes over time.

## Network Visualization
This section is connected to:
- [[Memories/README|Memories]] - Emotions influence memory formation
- [[Development/README|Development]] - Emotional growth tracks development
- [[Interactions/README|Interactions]] - Emotional responses to interactions

## Recent Updates
```dataview
TABLE created as "Time", tags as "Tags"
FROM "#emotional_states"
SORT created DESC
LIMIT 5
```

## Network Graph
```dataview
TABLE WITHOUT ID
  file.link as "Entry",
  tags as "Tags",
  created as "Time"
FROM "#emotional_states"
SORT created DESC
```
