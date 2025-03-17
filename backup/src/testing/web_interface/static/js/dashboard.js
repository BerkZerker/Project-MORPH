// Dashboard JavaScript for MORPH test visualization

// Connect to the WebSocket server
const socket = io();

// DOM elements
const currentTestElement = document.getElementById('current-test');
const testStatusElement = document.getElementById('test-status');
const elapsedTimeElement = document.getElementById('elapsed-time');
const remainingTimeElement = document.getElementById('remaining-time');
const progressBarElement = document.getElementById('progress-bar');
const stepInfoElement = document.getElementById('step-info');
const progressPercentageElement = document.getElementById('progress-percentage');
const knowledgeGraphElement = document.getElementById('knowledge-graph');
const expertActivationsElement = document.getElementById('expert-activations');
const modelMetricsElement = document.getElementById('model-metrics');
const expertSpecializationElement = document.getElementById('expert-specialization');
const timelineElement = document.getElementById('timeline');

// Metric elements
const kgNodesElement = document.getElementById('kg-nodes');
const kgEdgesElement = document.getElementById('kg-edges');
const numExpertsElement = document.getElementById('num-experts');
const maxActivationElement = document.getElementById('max-activation');
const sleepCyclesElement = document.getElementById('sleep-cycles');
const stepCountElement = document.getElementById('step-count');
const specExpertsElement = document.getElementById('spec-experts');
const maxScoreElement = document.getElementById('max-score');

// Test state
let testStartTime = null;
let testSteps = [];

// Format time in seconds to a human-readable string
function formatTime(seconds) {
    if (seconds === undefined || seconds === null) return '0s';
    
    if (seconds < 60) {
        return `${Math.round(seconds)}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        return `${minutes}m ${remainingSeconds}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

// Update the progress bar and related elements
function updateProgress(stepsCompleted, totalSteps, elapsedTime, remainingTime) {
    // Calculate progress percentage
    const progress = totalSteps > 0 ? (stepsCompleted / totalSteps) * 100 : 0;
    
    // Update progress bar
    progressBarElement.style.width = `${progress}%`;
    
    // Update step info
    stepInfoElement.textContent = `Step: ${stepsCompleted}/${totalSteps}`;
    
    // Update progress percentage
    progressPercentageElement.textContent = `${Math.round(progress)}%`;
    
    // Update time info
    elapsedTimeElement.textContent = `Elapsed: ${formatTime(elapsedTime)}`;
    remainingTimeElement.textContent = `Remaining: ${formatTime(remainingTime)}`;
}

// Update the timeline with a new step
function addTimelineStep(stepName, stepNumber, timestamp) {
    // Remove placeholder if present
    const placeholder = timelineElement.querySelector('.timeline-placeholder');
    if (placeholder) {
        timelineElement.removeChild(placeholder);
    }
    
    // Create step element
    const stepElement = document.createElement('div');
    stepElement.className = 'timeline-step';
    
    // Calculate time since test start
    const timeSinceStart = testStartTime ? ((timestamp - testStartTime) / 1000).toFixed(1) : 0;
    
    // Add step content
    stepElement.innerHTML = `
        <div class="timeline-step-name">${stepNumber}. ${stepName}</div>
        <div class="timeline-step-time">+${timeSinceStart}s</div>
    `;
    
    // Add to timeline
    timelineElement.appendChild(stepElement);
    
    // Scroll to bottom
    timelineElement.scrollTop = timelineElement.scrollHeight;
}

// Update visualization image
function updateVisualization(element, imageData, altText) {
    if (!imageData) {
        element.innerHTML = `<div class="placeholder">No data available</div>`;
        return;
    }
    
    // Remove placeholder if present
    const placeholder = element.querySelector('.placeholder');
    if (placeholder) {
        element.removeChild(placeholder);
    }
    
    // Check if image already exists
    let imgElement = element.querySelector('img');
    
    if (!imgElement) {
        // Create new image element
        imgElement = document.createElement('img');
        imgElement.alt = altText;
        element.appendChild(imgElement);
    }
    
    // Update image source
    imgElement.src = imageData;
}

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('test_start', (data) => {
    console.log('Test started:', data);
    
    // Update test info
    currentTestElement.textContent = data.test_name;
    testStatusElement.textContent = 'Running';
    testStatusElement.className = 'running';
    
    // Reset test state
    testStartTime = data.start_time;
    testSteps = [];
    
    // Reset timeline
    timelineElement.innerHTML = '<div class="timeline-placeholder">Test steps will appear here...</div>';
    
    // Reset visualizations
    knowledgeGraphElement.innerHTML = '<div class="placeholder">Waiting for data...</div>';
    expertActivationsElement.innerHTML = '<div class="placeholder">Waiting for data...</div>';
    modelMetricsElement.innerHTML = '<div class="placeholder">Waiting for data...</div>';
    expertSpecializationElement.innerHTML = '<div class="placeholder">Waiting for data...</div>';
    
    // Reset metrics
    kgNodesElement.textContent = '0';
    kgEdgesElement.textContent = '0';
    numExpertsElement.textContent = '0';
    maxActivationElement.textContent = '0';
    sleepCyclesElement.textContent = '0';
    stepCountElement.textContent = '0';
    specExpertsElement.textContent = '0';
    maxScoreElement.textContent = '0';
    
    // Reset progress
    updateProgress(0, 0, 0, 0);
});

socket.on('test_update', (data) => {
    console.log('Test update:', data);
    
    // Update progress
    updateProgress(
        data.steps_completed,
        data.total_steps,
        data.elapsed_time,
        data.estimated_remaining
    );
});

socket.on('snapshot_update', (data) => {
    console.log('Snapshot update:', data);
    
    // Update metrics
    kgNodesElement.textContent = data.knowledge_graph_nodes;
    kgEdgesElement.textContent = data.knowledge_graph_edges;
    numExpertsElement.textContent = data.num_experts;
    sleepCyclesElement.textContent = data.sleep_cycles_completed;
    stepCountElement.textContent = data.step_count;
    
    // Update visualizations if available
    if (data.visualizations) {
        const vis = data.visualizations;
        
        // Knowledge graph
        if (vis.knowledge_graph) {
            updateVisualization(
                knowledgeGraphElement,
                vis.knowledge_graph.image,
                'Knowledge Graph Visualization'
            );
        }
        
        // Expert activations
        if (vis.expert_activations) {
            updateVisualization(
                expertActivationsElement,
                vis.expert_activations.image,
                'Expert Activations Visualization'
            );
            
            maxActivationElement.textContent = vis.expert_activations.max_activation;
        }
        
        // Model metrics
        if (vis.model_metrics) {
            updateVisualization(
                modelMetricsElement,
                vis.model_metrics.image,
                'Model Metrics Visualization'
            );
        }
        
        // Expert specialization
        if (vis.expert_specialization) {
            updateVisualization(
                expertSpecializationElement,
                vis.expert_specialization.image,
                'Expert Specialization Visualization'
            );
            
            specExpertsElement.textContent = vis.expert_specialization.experts;
            maxScoreElement.textContent = vis.expert_specialization.max_score.toFixed(2);
        }
    }
});

socket.on('test_end', (data) => {
    console.log('Test ended:', data);
    
    // Update test status
    testStatusElement.textContent = 'Completed';
    testStatusElement.className = 'completed';
    
    // Update progress to 100%
    updateProgress(data.steps_completed, data.steps_completed, data.duration, 0);
});

// Add step to timeline when received
socket.on('step_update', (data) => {
    console.log('Step update:', data);
    
    // Add to timeline
    addTimelineStep(data.step_name, data.step_number, data.timestamp);
    
    // Store step
    testSteps.push(data);
});

// Handle connection errors
socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
    testStatusElement.textContent = 'Connection Error';
    testStatusElement.className = 'error';
});

// Handle disconnection
socket.on('disconnect', () => {
    console.log('Disconnected from server');
    testStatusElement.textContent = 'Disconnected';
    testStatusElement.className = 'error';
});

// Request test history on page load
socket.emit('request_history');
