"""
Pytest configuration for MORPH tests.

This file configures pytest to use the test visualization framework,
ensuring that visualizations are properly set up and displayed.
"""

import os
import pytest
import shutil
from pathlib import Path

from src.utils.testing.visualizer import TestVisualizer, get_default_visualizer
from src.utils.testing.reporters import TestReporter
from src.utils.testing import live_server, progress_tracker


# Add command line option to disable live visualizations
def pytest_addoption(parser):
    """Add command line options for test visualization."""
    parser.addoption(
        "--no-live-viz",
        action="store_true",
        default=False,
        help="Disable live visualizations during test runs"
    )


# Create visualization directory if it doesn't exist
@pytest.fixture(scope="session", autouse=True)
def setup_visualization_dir():
    """Set up the visualization directory for test results."""
    vis_dir = os.path.join(os.getcwd(), 'test_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Clean up old visualizations if needed
    # Uncomment to enable cleanup of old visualizations
    # for item in os.listdir(vis_dir):
    #     item_path = os.path.join(vis_dir, item)
    #     if os.path.isdir(item_path):
    #         shutil.rmtree(item_path)
    
    yield vis_dir


# Configure the default visualizer
@pytest.fixture(scope="session", autouse=True)
def configure_visualizer(setup_visualization_dir):
    """Configure the default test visualizer."""
    visualizer = get_default_visualizer()
    visualizer.output_dir = setup_visualization_dir
    
    yield visualizer


# Start the live visualization server for the test session
@pytest.fixture(scope="session", autouse=True)
def start_live_visualization_server(request):
    """Start the live visualization server for the test session."""
    # Check if live visualizations are disabled
    if request.config.getoption("--no-live-viz"):
        yield
        return
    
    # Start the server
    live_server.start_server(auto_open=True)
    
    yield
    
    # Stop the server after all tests have run
    live_server.stop_server()


# Generate a test report after all tests have run
@pytest.fixture(scope="session", autouse=True)
def generate_test_report(request):
    """Generate a test report after all tests have run."""
    yield
    
    # After all tests have run, generate a report
    reporter = TestReporter()
    report_path = reporter.generate_html_report()
    
    print(f"\n\n📊 Test visualization report generated at: {report_path}")
    print("Open this file in a browser to view the test visualizations.")


# Fixture for creating a test-specific visualizer
@pytest.fixture
def test_visualizer(request):
    """Create a test-specific visualizer."""
    test_name = request.node.name
    visualizer = TestVisualizer()
    visualizer.start_test(test_name)
    
    yield visualizer
    
    visualizer.end_test()


# Add a hook to automatically capture test results
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test results for visualization."""
    outcome = yield
    report = outcome.get_result()
    
    # Store the test result on the test item for later use
    if report.when == "call":
        item.test_outcome = report.outcome


# Add hooks for test start and finish to track progress
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Called before each test is run."""
    # Check if live visualizations are disabled
    if item.config.getoption("--no-live-viz"):
        return
    
    # Start tracking progress for this test
    test_name = item.name
    progress_tracker.start_test(test_name)
    
    # Start test in live server
    live_server.start_test(test_name)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Called after each test is run."""
    # Check if live visualizations are disabled
    if item.config.getoption("--no-live-viz"):
        return
    
    # End tracking progress for this test
    progress_tracker.end_test()
    
    # End test in live server
    live_server.end_test(item.name)
