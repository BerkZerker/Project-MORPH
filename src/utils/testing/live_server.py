"""
Live visualization server for MORPH tests.

This module provides a Flask-SocketIO server that serves real-time
visualizations of MORPH model tests.
"""

import os
import time
import threading
import webbrowser
from typing import Dict, List, Any, Optional, Union, Callable

import flask
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

# Global variables to track server state
_server_instance = None
_server_thread = None
_test_data = {
    'current_test': None,
    'start_time': None,
    'steps_completed': 0,
    'total_steps': 0,
    'snapshots': [],
    'current_snapshot': None,
    'test_history': {}  # For time estimation
}


class LiveVisualizationServer:
    """
    Flask-SocketIO server for live test visualizations.
    
    This class provides a web server that serves real-time visualizations
    of MORPH model tests via WebSockets.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8080, 
                debug: bool = False, auto_open: bool = True):
        """
        Initialize the live visualization server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            debug: Whether to run the server in debug mode
            auto_open: Whether to automatically open a browser window
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.auto_open = auto_open
        
        # Create Flask app
        self.app = Flask(
            __name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'web_interface/templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'web_interface/static')
        )
        
        # Create SocketIO instance
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Register routes
        self._register_routes()
        
        # Register SocketIO events
        self._register_socketio_events()
        
        # Server state
        self.running = False
        self.thread = None
    
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/static/<path:path>')
        def serve_static(path):
            """Serve static files."""
            return send_from_directory(self.app.static_folder, path)
    
    def _register_socketio_events(self):
        """Register SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            # Send current state to the new client
            self.socketio.emit('test_update', _test_data)
            
            if _test_data['current_snapshot']:
                self.socketio.emit('snapshot_update', _test_data['current_snapshot'])
        
        @self.socketio.on('request_history')
        def handle_request_history():
            """Handle request for test history."""
            self.socketio.emit('test_history', _test_data['test_history'])
    
    def start(self):
        """Start the server in a background thread."""
        if self.running:
            return
        
        self.running = True
        
        # Start the server in a background thread
        self.thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.thread.start()
        
        # Wait for server to start
        time.sleep(1)
        
        # Open browser if requested
        if self.auto_open:
            webbrowser.open(f'http://{self.host}:{self.port}')
    
    def _run_server(self):
        """Run the Flask-SocketIO server."""
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug, allow_unsafe_werkzeug=True)
    
    def stop(self):
        """Stop the server."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop the server
        self.socketio.stop()
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
    
    def update_test_state(self, test_name: str, step_name: str, 
                         step_number: int, total_steps: int, 
                         snapshot: Dict[str, Any]):
        """
        Update the test state and broadcast to clients.
        
        Args:
            test_name: Name of the current test
            step_name: Name of the current step
            step_number: Current step number
            total_steps: Total number of steps
            snapshot: Current state snapshot
        """
        global _test_data
        
        # Update test data
        _test_data['current_test'] = test_name
        _test_data['steps_completed'] = step_number
        _test_data['total_steps'] = total_steps
        _test_data['current_snapshot'] = snapshot
        
        # Add snapshot to history
        _test_data['snapshots'].append({
            'step_name': step_name,
            'step_number': step_number,
            'timestamp': time.time(),
            'snapshot': snapshot
        })
        
        # Broadcast update to clients
        self.socketio.emit('test_update', {
            'current_test': test_name,
            'steps_completed': step_number,
            'total_steps': total_steps,
            'elapsed_time': time.time() - _test_data['start_time'] if _test_data['start_time'] else 0,
            'estimated_remaining': self._estimate_remaining_time(test_name, step_number, total_steps)
        })
        
        # Broadcast snapshot update
        self.socketio.emit('snapshot_update', snapshot)
    
    def _estimate_remaining_time(self, test_name: str, step_number: int, total_steps: int) -> float:
        """
        Estimate remaining time based on current progress and historical data.
        
        Args:
            test_name: Name of the current test
            step_number: Current step number
            total_steps: Total number of steps
            
        Returns:
            Estimated remaining time in seconds
        """
        if step_number >= total_steps:
            return 0
        
        # If we have historical data for this test, use it
        if test_name in _test_data['test_history']:
            history = _test_data['test_history'][test_name]
            avg_step_time = history['total_time'] / history['total_steps']
            return avg_step_time * (total_steps - step_number)
        
        # Otherwise, use current test data
        if not _test_data['start_time'] or step_number == 0:
            return 0
        
        elapsed = time.time() - _test_data['start_time']
        avg_step_time = elapsed / step_number
        return avg_step_time * (total_steps - step_number)
    
    def start_test(self, test_name: str):
        """
        Start tracking a new test.
        
        Args:
            test_name: Name of the test being executed
        """
        global _test_data
        
        # Reset test data
        _test_data['current_test'] = test_name
        _test_data['start_time'] = time.time()
        _test_data['steps_completed'] = 0
        _test_data['total_steps'] = 0
        _test_data['snapshots'] = []
        _test_data['current_snapshot'] = None
        
        # Broadcast test start
        self.socketio.emit('test_start', {
            'test_name': test_name,
            'start_time': _test_data['start_time']
        })
    
    def end_test(self, test_name: str):
        """
        End tracking for the current test.
        
        Args:
            test_name: Name of the test being executed
        """
        global _test_data
        
        # Calculate test duration
        end_time = time.time()
        duration = end_time - _test_data['start_time'] if _test_data['start_time'] else 0
        
        # Update test history
        _test_data['test_history'][test_name] = {
            'total_time': duration,
            'total_steps': _test_data['total_steps'],
            'last_run': end_time
        }
        
        # Broadcast test end
        self.socketio.emit('test_end', {
            'test_name': test_name,
            'duration': duration,
            'steps_completed': _test_data['steps_completed']
        })


def get_server(host: str = '127.0.0.1', port: int = 8080, 
              debug: bool = False, auto_open: bool = True) -> LiveVisualizationServer:
    """
    Get or create the live visualization server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        debug: Whether to run the server in debug mode
        auto_open: Whether to automatically open a browser window
        
    Returns:
        LiveVisualizationServer instance
    """
    global _server_instance
    
    if _server_instance is None:
        _server_instance = LiveVisualizationServer(host, port, debug, auto_open)
    
    return _server_instance


def start_server(host: str = '127.0.0.1', port: int = 8080, 
                debug: bool = False, auto_open: bool = True):
    """
    Start the live visualization server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        debug: Whether to run the server in debug mode
        auto_open: Whether to automatically open a browser window
    """
    server = get_server(host, port, debug, auto_open)
    server.start()


def stop_server():
    """Stop the live visualization server."""
    global _server_instance
    
    if _server_instance and _server_instance.running:
        try:
            _server_instance.stop()
        except RuntimeError as e:
            # Handle the "Working outside of request context" error
            print(f"Note: Could not cleanly stop the server: {e}")
            # Force the server to stop by setting running to False
            _server_instance.running = False
            if _server_instance.thread:
                _server_instance.thread.join(timeout=1)
        
        _server_instance = None


def update_test_state(test_name: str, step_name: str, 
                     step_number: int, total_steps: int, 
                     snapshot: Dict[str, Any]):
    """
    Update the test state and broadcast to clients.
    
    Args:
        test_name: Name of the current test
        step_name: Name of the current step
        step_number: Current step number
        total_steps: Total number of steps
        snapshot: Current state snapshot
    """
    global _server_instance
    
    if _server_instance:
        _server_instance.update_test_state(
            test_name, step_name, step_number, total_steps, snapshot
        )


def start_test(test_name: str):
    """
    Start tracking a new test.
    
    Args:
        test_name: Name of the test being executed
    """
    global _server_instance, _test_data
    
    _test_data['start_time'] = time.time()
    
    if _server_instance:
        _server_instance.start_test(test_name)


def end_test(test_name: str):
    """
    End tracking for the current test.
    
    Args:
        test_name: Name of the test being executed
    """
    global _server_instance
    
    if _server_instance:
        _server_instance.end_test(test_name)
