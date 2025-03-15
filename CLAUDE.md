# CLAUDE.md - Assistant Guidelines for Project-MORPH

## Project Overview
MORPH (Mixture Of experts with Recursive Post-processing & Hierarchy) is a dynamic neural network architecture implementing a continuously evolving Mixture of Experts (MoE) model with post-processing mechanisms.

## Build & Test Commands
```bash
# Core commands
python -m pytest tests/                           # Run all tests
python -m pytest tests/test_expert.py             # Test specific file
python -m pytest tests/test_expert.py::test_expert_initialization  # Run specific test
python -m pytest -v                               # Verbose test output
python -m black morph/ tests/                     # Format code with Black
python -m isort morph/ tests/                     # Sort imports
python -m mypy morph/                             # Type checking
python -m flake8 morph/ tests/                    # Linting
```

## Code Style Guidelines
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Formatting**: Black (line length 88) and isort with Black profile
- **Imports**: Group imports (stdlib, third-party, local) with blank lines between groups
- **Types**: Use type hints for all functions/methods (disallow_untyped_defs=True)
- **Documentation**: Docstrings for classes and functions using triple quotes ("""...)
- **Error Handling**: Proper exception handling around dynamic expert creation/merging
- **Testing**: Pytest with descriptive test names (test_*) and docstrings
- **Architecture**: Follow modular design in core/ (experts, gating, knowledge graph, sleep)