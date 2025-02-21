# DecisionNetwork Neural Network
## Description
Makes decisions based on integrated information from other networks.
## Architecture Visualization
![[attachments/DecisionNetwork_architecture_20250221_001901.png]]
## Weight Distributions
![[attachments/DecisionNetwork_weights_20250221_001901.png]]
## Activation Heatmap
![[attachments/DecisionNetwork_activation_20250221_001901.png]]
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
      "out_features": 32
    }
  ],
  "total_parameters": 43232,
  "trainable_parameters": 43232,
  "layer_sizes": [
    128,
    64,
    32
  ]
}
```
## Mermaid Diagram
```mermaid
graph TD
    0[0<br/>256->128]
    3[3<br/>128->64]
    5[5<br/>64->32]
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
FROM "#neural_network" and [[DecisionNetwork]]
SORT created DESC
LIMIT 5
```
