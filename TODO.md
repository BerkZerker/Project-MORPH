# Project-MORPH TODO List

## Current Tasks
- Build project to load all settings from some sort of config to be easily customizable
- !P1 make thesis of the exact project goal and put it in the readme as well as other project files
- Remove all emojis from tests
- Update test GUIs to show accurate info, including processor info
- Full GPU use - in progress
- Make sure the config for the tests is the smallest possible without compromising functionality
- Full CPU use
- Verify all tests pass
- Verify that all the tests do the intended functionality
- Make sure the GUI is accurate
- Make sure training uses full computer resources efficiently
- Keep debugging and optimize, make sure tests are working, and have full understanding of codebase

## Project Refocusing: Text Processing Framework

### Overview
The Project-MORPH codebase needs to be refocused to prioritize text processing applications. Currently, the example code and some testing utilities are using MNIST (image data) for demonstration purposes, but the project's goal is to build a dynamic Mixture of Experts (MoE) model for text processing.

### Required Changes

#### 1. Update Data Processing Utilities
- [ ] Replace the MNIST dataset loader with text dataset loaders
- [ ] Create text tokenization and embedding utilities
- [ ] Modify the `src/utils/data.py` module to handle text data
- [ ] Implement text-specific data augmentation techniques

#### 2. Revise Example Code
- [ ] Update `examples/mnist_example.py` to a text classification example
- [ ] Convert `examples/continual_learning_example.py` to use text datasets
- [ ] Create new examples for text-specific tasks (sentiment analysis, topic classification)
- [ ] Update `examples/benchmark_example.py` for text benchmarks

#### 3. Modify Testing Framework
- [ ] Create text-specific test datasets
- [ ] Update test cases to use text data instead of MNIST
- [ ] Add text-specific evaluation metrics
- [ ] Replace image visualization utilities with text visualization tools

#### 4. Expert Architecture Adjustments
- [ ] Optimize expert networks for text feature processing
- [ ] Implement attention mechanisms for text processing
- [ ] Add support for transformer-based expert modules
- [ ] Update expert initialization for text feature spaces

#### 5. Knowledge Graph Enhancements
- [ ] Update concept embedding for text domains
- [ ] Add text-specific semantic similarity metrics
- [ ] Implement text domain specialization tracking
- [ ] Create text-specific concept drift detection

#### 6. Update Documentation
- [x] Revise README.md to focus on text processing
- [x] Update architecture.md with text-specific components
- [ ] Create a text-specific getting started guide
- [ ] Add documentation on supported text datasets and formats

#### 7. Clean Up Dependencies
- [ ] Remove unnecessary image processing libraries
- [ ] Add text processing libraries (transformers, nltk, spacy)
- [ ] Update requirements.txt with text-specific dependencies
- [ ] Ensure GPU acceleration works with text processing models

## Implementation Priority

### Phase 1: Core Text Processing Support
1. Update data utilities to handle text
2. Revise example code for basic text tasks
3. Update requirements.txt with text-specific dependencies

### Phase 2: Text-Specific Features
1. Implement transformer-based expert modules
2. Add text-specific semantic similarity metrics
3. Create text domain specialization tracking

### Phase 3: Advanced Text Capabilities
1. Add support for complex NLP tasks
2. Implement cross-domain text knowledge transfer
3. Create text-specific visualizations for model understanding

## Original Roadmap
- Set up training data set and whatever other necessary resources
- v1 training run