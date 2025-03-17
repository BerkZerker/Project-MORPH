"""
Test reporting utilities for MORPH models.

This module provides tools for generating reports from test visualizations,
allowing users to view and share test results.
"""

import os
import time
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


class TestReporter:
    """
    Generates reports from test visualizations.
    
    This class provides methods to create HTML reports, summary visualizations,
    and other reporting formats from test visualization data.
    """
    
    def __init__(self, visualization_dir: Optional[str] = None, 
                output_dir: Optional[str] = None):
        """
        Initialize the test reporter.
        
        Args:
            visualization_dir: Directory containing test visualizations
                              (default: None, uses 'test_visualizations' in current directory)
            output_dir: Directory to save reports
                       (default: None, uses 'test_reports' in current directory)
        """
        self.visualization_dir = visualization_dir or os.path.join(os.getcwd(), 'test_visualizations')
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'test_reports')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_html_report(self, test_dirs: Optional[List[str]] = None, 
                           include_images: bool = True):
        """
        Generate an HTML report from test visualizations.
        
        Args:
            test_dirs: List of test directories to include in the report
                      (default: None, uses all directories in visualization_dir)
            include_images: Whether to include images in the report (default: True)
                          If False, links to images will be used instead
        
        Returns:
            Path to the generated HTML report
        """
        # Find test directories if not provided
        if test_dirs is None:
            test_dirs = [d for d in os.listdir(self.visualization_dir) 
                        if os.path.isdir(os.path.join(self.visualization_dir, d))]
        
        # Create report directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_dir = os.path.join(self.output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create HTML report
        html_path = os.path.join(report_dir, "index.html")
        
        with open(html_path, 'w') as f:
            f.write(self._generate_html_header())
            
            # Add summary section
            f.write("""
            <h1>MORPH Test Visualization Report</h1>
            <div class="summary">
                <h2>Test Summary</h2>
                <p>Generated: {}</p>
                <p>Tests: {}</p>
            </div>
            """.format(time.strftime('%Y-%m-%d %H:%M:%S'), len(test_dirs)))
            
            # Add test sections
            for test_dir in test_dirs:
                test_path = os.path.join(self.visualization_dir, test_dir)
                test_name = test_dir.split('_')[0]  # Extract test name from directory name
                
                # Create test directory in report
                test_report_dir = os.path.join(report_dir, test_dir)
                os.makedirs(test_report_dir, exist_ok=True)
                
                # Copy or link test visualizations
                if include_images:
                    self._copy_test_visualizations(test_path, test_report_dir)
                
                # Add test section to HTML
                f.write(self._generate_test_section(test_dir, test_name, test_path, include_images))
            
            f.write(self._generate_html_footer())
        
        print(f"HTML report generated at: {html_path}")
        return html_path
    
    def generate_summary_report(self):
        """
        Generate a summary report of all test visualizations.
        
        This report includes metrics like test duration, number of steps,
        and other summary statistics.
        
        Returns:
            Path to the generated summary report
        """
        # Find test directories
        test_dirs = [d for d in os.listdir(self.visualization_dir) 
                    if os.path.isdir(os.path.join(self.visualization_dir, d))]
        
        # Create report directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_dir = os.path.join(self.output_dir, f"summary_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Collect test metrics
        test_metrics = []
        
        for test_dir in test_dirs:
            test_path = os.path.join(self.visualization_dir, test_dir)
            test_name = test_dir.split('_')[0]  # Extract test name from directory name
            
            # Find step directories
            step_dirs = [d for d in os.listdir(test_path) 
                        if os.path.isdir(os.path.join(test_path, d))]
            
            # Calculate metrics
            metrics = {
                'test_name': test_name,
                'steps': len(step_dirs),
                'directory': test_dir
            }
            
            test_metrics.append(metrics)
        
        # Generate summary visualizations
        self._generate_summary_visualizations(test_metrics, report_dir)
        
        # Create HTML summary
        html_path = os.path.join(report_dir, "summary.html")
        
        with open(html_path, 'w') as f:
            f.write(self._generate_html_header())
            
            # Add summary section
            f.write("""
            <h1>MORPH Test Summary Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Generated: {}</p>
                <p>Tests: {}</p>
            </div>
            """.format(time.strftime('%Y-%m-%d %H:%M:%S'), len(test_metrics)))
            
            # Add summary visualizations
            f.write("""
            <h2>Test Metrics</h2>
            <div class="images">
                <div class="image-container">
                    <img src="test_steps.png" alt="Test Steps">
                </div>
            </div>
            """)
            
            # Add test table
            f.write("""
            <h2>Test Details</h2>
            <table class="test-table">
                <tr>
                    <th>Test Name</th>
                    <th>Steps</th>
                    <th>Link</th>
                </tr>
            """)
            
            for metrics in test_metrics:
                f.write(f"""
                <tr>
                    <td>{metrics['test_name']}</td>
                    <td>{metrics['steps']}</td>
                    <td><a href="../{metrics['directory']}/index.html">View Test</a></td>
                </tr>
                """)
            
            f.write("</table>")
            
            f.write(self._generate_html_footer())
        
        print(f"Summary report generated at: {html_path}")
        return html_path
    
    def _generate_html_header(self):
        """Generate HTML header with styles."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MORPH Test Visualization Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .summary {
                    background: #f5f5f5;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                .test-section {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                .images {
                    display: flex;
                    flex-wrap: wrap;
                }
                .image-container {
                    margin: 10px;
                    max-width: 45%;
                }
                img {
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                }
                .test-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .test-table th, .test-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .test-table th {
                    background-color: #f2f2f2;
                }
                .test-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .test-table tr:hover {
                    background-color: #f5f5f5;
                }
            </style>
        </head>
        <body>
        """
    
    def _generate_html_footer(self):
        """Generate HTML footer."""
        return """
        </body>
        </html>
        """
    
    def _generate_test_section(self, test_dir, test_name, test_path, include_images):
        """Generate HTML for a test section."""
        # Find step directories
        step_dirs = sorted([d for d in os.listdir(test_path) 
                          if os.path.isdir(os.path.join(test_path, d))])
        
        html = f"""
        <div class="test-section">
            <h2>Test: {test_name}</h2>
            <p>Steps: {len(step_dirs)}</p>
        """
        
        # Add timeline image if it exists
        timeline_path = os.path.join(test_path, "timeline.png")
        if os.path.exists(timeline_path):
            if include_images:
                html += f"""
                <h3>Timeline</h3>
                <div class="images">
                    <div class="image-container">
                        <img src="{test_dir}/timeline.png" alt="Timeline">
                    </div>
                </div>
                """
            else:
                html += f"""
                <h3>Timeline</h3>
                <p><a href="{test_path}/timeline.png" target="_blank">View Timeline</a></p>
                """
        
        # Add step links
        html += "<h3>Steps</h3><ul>"
        
        for step_dir in step_dirs:
            step_name = step_dir.split('_', 1)[1] if '_' in step_dir else step_dir
            step_path = os.path.join(test_path, step_dir)
            
            if include_images:
                html += f"""
                <li><a href="{test_dir}/{step_dir}/index.html">{step_name}</a></li>
                """
            else:
                html += f"""
                <li><a href="{step_path}/index.html" target="_blank">{step_name}</a></li>
                """
        
        html += "</ul>"
        
        # Add link to full test
        html += f"""
        <p><a href="{test_dir}/index.html">View Full Test Report</a></p>
        </div>
        """
        
        return html
    
    def _copy_test_visualizations(self, src_dir, dst_dir):
        """Copy test visualizations to the report directory."""
        # Copy index.html and timeline.png
        for file in ['index.html', 'timeline.png']:
            src_file = os.path.join(src_dir, file)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_dir)
        
        # Copy step directories
        step_dirs = [d for d in os.listdir(src_dir) 
                    if os.path.isdir(os.path.join(src_dir, d))]
        
        for step_dir in step_dirs:
            src_step_dir = os.path.join(src_dir, step_dir)
            dst_step_dir = os.path.join(dst_dir, step_dir)
            
            os.makedirs(dst_step_dir, exist_ok=True)
            
            # Copy all files in step directory
            for file in os.listdir(src_step_dir):
                src_file = os.path.join(src_step_dir, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_step_dir)
    
    def _generate_summary_visualizations(self, test_metrics, output_dir):
        """Generate summary visualizations."""
        # Create bar chart of test steps
        plt.figure(figsize=(10, 6))
        
        test_names = [m['test_name'] for m in test_metrics]
        steps = [m['steps'] for m in test_metrics]
        
        plt.bar(test_names, steps)
        plt.xlabel('Test Name')
        plt.ylabel('Number of Steps')
        plt.title('Test Steps')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "test_steps.png"))
        plt.close()


def find_test_visualizations(base_dir: Optional[str] = None) -> List[str]:
    """
    Find all test visualization directories.
    
    Args:
        base_dir: Base directory to search (default: None, uses 'test_visualizations' in current directory)
        
    Returns:
        List of test visualization directory paths
    """
    base_dir = base_dir or os.path.join(os.getcwd(), 'test_visualizations')
    
    if not os.path.exists(base_dir):
        return []
    
    return [d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]


def open_latest_report(report_type: str = 'html'):
    """
    Open the latest test report.
    
    Args:
        report_type: Type of report to open ('html' or 'summary')
        
    Returns:
        Path to the opened report
    """
    report_dir = os.path.join(os.getcwd(), 'test_reports')
    
    if not os.path.exists(report_dir):
        raise FileNotFoundError(f"Report directory not found: {report_dir}")
    
    # Find latest report directory
    report_dirs = [d for d in os.listdir(report_dir) 
                  if os.path.isdir(os.path.join(report_dir, d))]
    
    if not report_dirs:
        raise FileNotFoundError(f"No reports found in {report_dir}")
    
    # Sort by timestamp (assuming directory names contain timestamps)
    report_dirs.sort(reverse=True)
    latest_dir = os.path.join(report_dir, report_dirs[0])
    
    # Find report file
    if report_type == 'html':
        report_file = os.path.join(latest_dir, "index.html")
    elif report_type == 'summary':
        report_file = os.path.join(latest_dir, "summary.html")
    else:
        raise ValueError(f"Invalid report type: {report_type}")
    
    if not os.path.exists(report_file):
        raise FileNotFoundError(f"Report file not found: {report_file}")
    
    # Open report file
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(report_file)}")
    
    return report_file
