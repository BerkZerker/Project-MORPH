# Live Test Visualizations for Project MORPH

This document explains how to use the live test visualization framework that has been added to Project MORPH. The framework provides real-time visual insights into what's happening during test execution, making it easier to understand the internal workings of the MORPH model.

## Overview

The live test visualization framework captures the state of MORPH models during test execution and displays visualizations in real-time that help understand what's happening under the hood. These visualizations include:

- Knowledge graph visualizations
- Expert activation patterns
- Model metrics
- Timeline of test execution

## Running Tests with Live Visualizations

All tests now have live visualizations enabled by default. To run tests with live visualizations:

```bash
# Run all tests with live visualizations
pytest

# Run a specific test with live visualizations
pytest tests/test_sleep.py::test_full_sleep_cycle
```

When you run tests, a live visualization dashboard will automatically open in your browser, showing real-time updates as the tests progress. This works for individual tests as well as for the full test suite.

The dashboard will show the current test being run, its progress, and visualizations of the model state in real-time.

## Live Visualization Dashboard

During test execution, a web-based dashboard will automatically open in your browser, showing:

- Real-time progress of the test (elapsed time, estimated time remaining)
- Live-updating visualizations of the model state
- Knowledge graph changes as they happen
- Expert activation patterns
- Model metrics
- Timeline of test steps

The live dashboard runs on a local web server (default: http://127.0.0.1:8080) and updates in real-time via WebSockets.

## Customizing Visualizations

### Disabling Visualizations

If you need to disable visualizations for a specific test:

```python
@visualize_test(enabled=False)
def test_without_visualizations():
    # Test code here
    ...
```

### Disabling Live Visualizations

If you want to disable live visualizations for a specific test:

```python
@visualize_test(live=False)
def test_without_live_visualizations():
    # Test code here
    ...
```

If you want to disable live visualizations for the entire test run, you can use the `--no-live-viz` command line option:

```bash
pytest --no-live-viz
```

### Adding Visualization Points

To add visualization points to a test:

```python
from morph.utils.testing.decorators import capture_test_state

def test_with_custom_visualizations():
    # Setup code
    
    # Capture state before an operation
    with capture_test_state(model, "Before Operation"):
        # Operation code
        result = model.some_operation()
    
    # Assertions
    assert result is True
```

### Custom Visualization Data

You can include additional data in visualizations:

```python
with capture_test_state(model, "Custom Data", 
                       additional_data={"custom_metric": 42}):
    # Operation code
    ...
```

## Visualization Components

The framework consists of several components:

1. **TestVisualizer**: Core class that captures and tracks model state
2. **Decorators**: `@visualize_test` and `capture_test_state` for easy integration
3. **LiveVisualizationServer**: Provides real-time visualizations via WebSockets
4. **ProgressTracker**: Tracks test progress and estimates remaining time
5. **LiveModelVisualizer**: Generates visualizations for the live dashboard

## Implementation Details

The live visualization framework is implemented in the following files:

- `src/utils/testing/visualizer.py`: Core state tracking logic
- `src/utils/testing/decorators.py`: Test decorators
- `src/utils/testing/live_server.py`: WebSocket server for live updates
- `src/utils/testing/progress_tracker.py`: Test progress tracking
- `src/utils/testing/live_visualizer.py`: Live visualization generation
- `src/utils/testing/web_interface/`: Web dashboard files
- `tests/conftest.py`: Pytest configuration

## Examples

### Basic Test with Live Visualizations

```python
@visualize_test
def test_example():
    # Test code
    ...
```

### Test with Multiple Visualization Points

```python
@visualize_test
def test_with_steps():
    # Setup
    model = MorphModel(config)
    
    # First operation with visualization
    with capture_test_state(model, "First Operation"):
        result1 = model.operation1()
    
    # Second operation with visualization
    with capture_test_state(model, "Second Operation"):
        result2 = model.operation2()
    
    # Assertions
    assert result1 and result2
```

## Troubleshooting

- **Missing visualizations**: Ensure the test is decorated with `@visualize_test`
- **Visualization errors**: Check that the model being visualized has the expected structure
- **Live dashboard not opening**: Check that the server is running (look for "Live visualization server running on..." in the console output)
- **WebSocket connection errors**: Make sure port 8080 is available and not blocked by a firewall
- **Slow updates**: The visualization generation might be CPU-intensive; try reducing the update frequency or disabling live visualization for very fast tests
