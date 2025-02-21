# IntegratedBrain Neural Network
## Description
The main brain architecture that integrates all neural components for cognitive processing.
## Architecture Visualization
![[attachments/IntegratedBrain_architecture_20250221_001846.png]]
## Weight Distributions
![[attachments/IntegratedBrain_weights_20250221_001846.png]]
## Activation Heatmap
![[attachments/IntegratedBrain_activation_20250221_001846.png]]
## Network Statistics
```json
{
  "layers": [
    {
      "name": "0",
      "type": "Linear",
      "in_features": 384,
      "out_features": 256
    },
    {
      "name": "3",
      "type": "Linear",
      "in_features": 256,
      "out_features": 128
    },
    {
      "name": "5",
      "type": "Linear",
      "in_features": 128,
      "out_features": 64
    }
  ],
  "total_parameters": 139712,
  "trainable_parameters": 139712,
  "layer_sizes": [
    256,
    128,
    64
  ]
}
```
## Mermaid Diagram
```mermaid
graph TD
    0[0<br/>384->256]
    3[3<br/>256->128]
    5[5<br/>128->64]
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
FROM "#neural_network" and [[IntegratedBrain]]
SORT created DESC
LIMIT 5
```
