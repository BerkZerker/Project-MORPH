# Project-MORPH Development Guide

## Build and Test Commands
- Install: `pip install -e .`
- Run all tests: `pytest tests/`
- Run single test: `pytest tests/path/to/test_file.py::test_function_name`
- Run tests with coverage: `pytest --cov=src tests/`
- Linting: `black src/ tests/ && isort src/ tests/ && flake8 src/ tests/`
- Type checking: `mypy src/ tests/`

## Code Style Guidelines
- Format: Black with 88 char line length
- Imports: Use isort with black profile, grouped by standard/third-party/local
- Types: Use type hints for all function parameters and return values
- Documentation: Docstrings for all classes and functions using Google style
- Error handling: Use specific exceptions with informative messages
- Naming: snake_case for functions/variables, CamelCase for classes
- Testing: Every feature needs test coverage, use fixtures for shared setup
- Device handling: Always check tensor device compatibility
- Performance: Prefer vectorized operations over loops