# EmotionalNetwork Neural Network
## Description
Processes and regulates emotional states and responses.
## Architecture Visualization
![[attachments/EmotionalNetwork_architecture_20250221_001850.png]]
## Weight Distributions
![[attachments/EmotionalNetwork_weights_20250221_001850.png]]
## Activation Heatmap
![[attachments/EmotionalNetwork_activation_20250221_001850.png]]
## Network Statistics
```json
{
  "layers": [
    {
      "name": "0",
      "type": "Linear",
      "in_features": 256,
      "out_features": 128
    },
    {
      "name": "3",
      "type": "Linear",
      "in_features": 128,
      "out_features": 64
    },
    {
      "name": "5",
      "type": "Linear",
      "in_features": 64,
      "out_features": 12
    }
  ],
  "total_parameters": 41932,
  "trainable_parameters": 41932,
  "layer_sizes": [
    128,
    64,
    12
  ]
}
```
## Mermaid Diagram
```mermaid
graph TD
    0[0<br/>256->128]
    3[3<br/>128->64]
    5[5<br/>64->12]
    0 --> 3
    3 --> 5
```
## Connections
- [[Development/README|Development]] - Network evolution
- [[Memories/README|Memories]] - Memory patterns
- [[Language_Learning/README|Language Learning]] - Language processing

## Recent Updates
```dataview
TABLE created as "Time", tags as "Tags"
FROM "#neural_network" and [[EmotionalNetwork]]
SORT created DESC
LIMIT 5
```
