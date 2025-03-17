# Test Visualizations for Project MORPH

This document explains how to use the test visualization framework that has been added to Project MORPH. The framework provides visual insights into what's happening during test execution, making it easier to understand the internal workings of the MORPH model.

## Overview

The test visualization framework captures the state of MORPH models during test execution and generates visualizations that help understand what's happening under the hood. These visualizations include:

- Knowledge graph visualizations
- Expert activation patterns
- Model metrics
- Timeline of test execution
- HTML reports with interactive elements
- **NEW**: Live visualizations during test execution

## Running Tests with Visualizations

All tests now have visualizations enabled by default. To run tests and generate visualizations:

```bash
# Run all tests with visualizations
pytest

# Run a specific test with visualizations
pytest tests/test_sleep.py::test_full_sleep_cycle
```

When you run tests, a live visualization dashboard will automatically open in your browser, showing real-time updates as the tests progress. This works for individual tests as well as for the full test suite.

```bash
# Run all tests with live visualizations
pytest
```

The dashboard will show the current test being run, its progress, and visualizations of the model state in real-time.

## Viewing Visualizations

### Live Visualizations

During test execution, a web-based dashboard will automatically open in your browser, showing:

- Real-time progress of the test (elapsed time, estimated time remaining)
- Live-updating visualizations of the model state
- Knowledge graph changes as they happen
- Expert activation patterns
- Model metrics
- Timeline of test steps

The live dashboard runs on a local web server (default: http://127.0.0.1:5000) and updates in real-time via WebSockets.

### Static Reports

After running tests, visualizations are also saved to the `test_visualizations` directory in the project root. Each test gets its own subdirectory with timestamped visualizations.

A summary HTML report is also generated in the `test_reports` directory. This report provides links to all test visualizations and summary metrics.

To view the static report:

1. Open the HTML file indicated in the test output (look for "Test visualization report generated at: ...")
2. Navigate through the report to view different tests and visualizations

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

If you want to keep static visualizations but disable the live dashboard for a specific test:

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

By default, these visualization points will also be shown in the live dashboard. To disable live updates for a specific capture point:

```python
with capture_test_state(model, "Before Operation", live=False):
    # Operation code
    result = model.some_operation()
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

1. **TestVisualizer**: Core class that captures and visualizes model state
2. **Decorators**: `@visualize_test` and `capture_test_state` for easy integration
3. **TestReporter**: Generates HTML reports from visualizations
4. **LiveVisualizationServer**: Provides real-time visualizations via WebSockets
5. **ProgressTracker**: Tracks test progress and estimates remaining time
6. **LiveModelVisualizer**: Generates visualizations for the live dashboard

## Directory Structure

Visualizations are organized as follows:

```
test_visualizations/
  ├── test_name_timestamp/
  │   ├── index.html           # Test summary
  │   ├── timeline.png         # Test timeline
  │   ├── 00_Initial_State/    # First step
  │   │   ├── index.html       # Step summary
  │   │   ├── knowledge_graph.png
  │   │   ├── expert_activations.png
  │   │   └── model_metrics.png
  │   ├── 01_Step_Name/        # Second step
  │   │   └── ...
  │   └── ...
  └── ...

test_reports/
  └── report_timestamp/
      ├── index.html           # Main report
      └── ...

morph/utils/testing/
  ├── visualizer.py            # Core visualization logic
  ├── decorators.py            # Test decorators
  ├── reporters.py             # Report generation
  ├── live_server.py           # Live visualization server
  ├── progress_tracker.py      # Test progress tracking
  ├── live_visualizer.py       # Live visualization generator
  └── web_interface/           # Web dashboard
      ├── templates/           # HTML templates
      └── static/              # CSS, JS, and other static files
```

## Implementation Details

The visualization framework is implemented in the following files:

- `morph/utils/testing/visualizer.py`: Core visualization logic
- `morph/utils/testing/decorators.py`: Test decorators
- `morph/utils/testing/reporters.py`: Report generation
- `morph/utils/testing/live_server.py`: WebSocket server for live updates
- `morph/utils/testing/progress_tracker.py`: Test progress tracking
- `morph/utils/testing/live_visualizer.py`: Live visualization generation
- `morph/utils/testing/web_interface/`: Web dashboard files
- `tests/conftest.py`: Pytest configuration

## Examples

### Basic Test with Visualizations

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
- **Empty reports**: Make sure tests are actually running and passing
- **Live dashboard not opening**: Check that the server is running (look for "Live visualization server running on..." in the console output)
- **WebSocket connection errors**: Make sure port 5000 is available and not blocked by a firewall
- **Slow updates**: The visualization generation might be CPU-intensive; try reducing the update frequency or disabling live visualization for very fast tests
