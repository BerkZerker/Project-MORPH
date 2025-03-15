# CLAUDE.md - Assistant Guidelines for Project-MORPH

## Project Overview
MORPH (Mixture Of experts with Recursive Post-processing & Hierarchy) is a dynamic neural network architecture implementing a continuously evolving Mixture of Experts (MoE) model with post-processing mechanisms.

## Build & Test Commands
```bash
# These commands will be updated as implementation progresses
# Phase 1: Basic MoE implementation
python -m pytest tests/                # Run all tests
python -m pytest tests/test_moe.py     # Test basic MoE functionality
python -m pytest tests/test_moe.py::test_gating_network  # Run specific test
```

## Code Style Guidelines
- **Naming**: Use descriptive snake_case for variables/functions, PascalCase for classes
- **Comments**: Document expert structures, knowledge graph implementation, and merging algorithms
- **Imports**: Group imports (stdlib, third-party, local) with blank line separators
- **Error Handling**: Use proper exception handling around dynamic expert creation/merging
- **Types**: Use type hints for expert definitions, knowledge graph structures
- **Architecture**: Follow modular design separating experts, gating networks, and post-processing

## Implementation Phases
1. Basic MoE with gating network
2. Adaptive expert growth with knowledge graph
3. Dynamic expert merging and pruning
4. Sleep function for memory replay and consolidation