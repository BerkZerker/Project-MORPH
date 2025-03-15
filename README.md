# MORPH: Mixture Of experts with Recursive Post-processing & Hierarchy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MORPH is a novel neural network architecture implementing a **Dynamic Mixture of Experts (MoE)** model with continuous learning capabilities, adaptive expert creation, and brain-inspired post-processing mechanisms.

## Key Features

- **Dynamic Expert Creation**: Automatically generates new expert networks when existing ones underperform
- **Knowledge Graph Routing**: Routes inputs based on semantic similarity using a graph-based knowledge structure
- **Expert Consolidation**: Periodically merges similar experts to optimize memory and prevent redundancy
- **Sleep Function**: Implements a brain-inspired post-processing mechanism for knowledge consolidation
- **Continuous Learning**: Designed to learn incrementally without catastrophic forgetting

## Architecture Overview

```mermaid
graph TD
    Input[Input Data] --> Gating[Gating Network]
    Gating --> E1[Expert 1]
    Gating --> E2[Expert 2]
    Gating --> E3[Expert 3]
    Gating --> EN[Expert N]
    
    E1 --> Combine[Output Combination]
    E2 --> Combine
    E3 --> Combine
    EN --> Combine
    
    Combine --> Output[Final Output]
    
    Gating --> Create[Create New Expert]
    Create --> KG[Knowledge Graph]
    
    subgraph Sleep[Sleep Cycle]
        KG --> Merge[Merge Experts]
        KG --> Prune[Prune Experts]
        Merge --> KG
        Prune --> KG
    end
    
    KG --> Gating
    
    classDef sleepStyle fill:#f9d6ff,stroke:#333;
    class Sleep sleepStyle;
```

MORPH consists of four main components:

1. **Experts**: Specialized neural networks trained on specific subtasks or data distributions
2. **Gating Network**: Determines which experts to activate for each input
3. **Knowledge Graph**: Tracks relationships between experts and concepts
4. **Sleep Module**: Handles periodic knowledge consolidation and optimization

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- NetworkX 2.8+
- PyTorch Lightning (optional, for training utilities)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/project-morph.git
cd project-morph

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Example Usage

```python
from morph.model import MorphModel
from morph.config import MorphConfig

# Initialize MORPH model
config = MorphConfig(
    num_initial_experts=4,
    expert_hidden_size=256,
    enable_dynamic_experts=True,
    sleep_cycle_frequency=1000
)
model = MorphModel(config)

# Train model
model.train(train_dataset, epochs=10)

# Sleep cycle for consolidation
model.sleep()

# Continue training with new data
model.train(new_dataset, epochs=5)
```

## Implementation Progress

MORPH is being implemented in four phases:

```mermaid
gantt
    title MORPH Implementation Progress
    dateFormat YYYY-MM-DD
    
    section Phase 1
    Core expert implementation   :done, p1_1, 2025-01-15, 2025-02-01
    Basic gating network         :done, p1_2, 2025-02-01, 2025-02-15
    Routing mechanism            :active, p1_3, 2025-02-15, 2025-03-01
    Evaluation framework         :p1_4, 2025-03-01, 2025-03-15
    
    section Phase 2
    Uncertainty metrics          :p2_1, 2025-03-15, 2025-04-01
    Expert initialization        :p2_2, 2025-04-01, 2025-04-15
    Knowledge graph (basic)      :p2_3, 2025-04-15, 2025-05-01
    
    section Phase 3
    Expert similarity metrics    :p3_1, 2025-05-01, 2025-05-15
    Merging algorithm            :p3_2, 2025-05-15, 2025-06-01
    Pruning mechanism            :p3_3, 2025-06-01, 2025-06-15
    
    section Phase 4
    Memory replay system         :p4_1, 2025-06-15, 2025-07-01
    Expert reorganization        :p4_2, 2025-07-01, 2025-07-15
    Sleep cycle scheduler        :p4_3, 2025-07-15, 2025-08-01
```

Current status:

| Component | Status | Progress |
|-----------|--------|----------|
| Core Experts | ✅ Complete | 100% |
| Gating Network | ✅ Complete | 100% |
| Routing Mechanism | 🔄 In Progress | 60% |
| Knowledge Graph | ⏱️ Not Started | 0% |
| Sleep Module | ⏱️ Not Started | 0% |

See the [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed implementation steps.

## Documentation

- [Architecture Design](docs/architecture.md)
- [API Reference](docs/api.md)
- [Examples](examples/README.md)

## Visualization of the MORPH Approach

```mermaid
graph LR
    subgraph Train["Training Process"]
        Data[New Data] --> Gating[Gating Network]
        Gating --> E1[Expert 1]
        Gating --> E2[Expert 2]
        Gating --> E3[Expert 3]
        Gating -->|uncertainty > threshold| Create[Create New Expert]
        E1 --> Combine[Output Combination]
        E2 --> Combine
        E3 --> Combine
        Combine --> Output[Final Output]
    end
    
    subgraph KGraph["Knowledge Graph"]
        KG[Graph Structure]
    end
    
    subgraph Sleep["Sleep Cycle"]
        Merge[Merge Experts]
        Prune[Prune Experts]
    end
    
    Train --> Sleep
    Sleep --> Train
    KGraph --> Train
    Train --> KGraph
    
    classDef trainStyle fill:#d4f1f9,stroke:#333;
    classDef kgStyle fill:#e1d5e7,stroke:#333;
    classDef sleepStyle fill:#f9d6ff,stroke:#333;
    
    class Train trainStyle;
    class KGraph kgStyle;
    class Sleep sleepStyle;
```

The diagram above shows how the three key mechanisms of MORPH (training, knowledge graph management, and sleep cycles) interact to create a dynamic, adaptive system.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MORPH in your research, please cite:

```bibtex
@misc{morph2025,
  author = {Project MORPH Contributors},
  title = {MORPH: Mixture Of experts with Recursive Post-processing & Hierarchy},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/project-morph}}
}
```
