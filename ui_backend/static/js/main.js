
// Global variables
let selectedFile = null;
let fileDataUrl = null; // Store the file data URL for persistence
let lastDetectionResult = null; // Store last result

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const previewSection = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const uploadForm = document.getElementById('uploadForm');
const loadingSection = document.getElementById('loading');
const resultsSection = document.getElementById('results');
const resultImg = document.getElementById('resultImg');
const confInput = document.getElementById('confInput');
const rangeValue = document.querySelector('.range-value');

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    updateConfidenceValue();
    restoreState(); // Restore state on page load
    handlePageShow(); // Handle browser back/forward
});

// Handle browser back/forward navigation
function handlePageShow() {
    window.addEventListener('pageshow', function(event) {
        // Restore state when user navigates back
        if (event.persisted) {
            restoreState();
            console.log('Page restored from cache, restoring state');
        }
    });
}

// Store state in sessionStorage for persistence across navigation
function saveState() {
    const state = {
        hasFile: selectedFile !== null,
        fileDataUrl: fileDataUrl,
        fileName: selectedFile ? selectedFile.name : null,
        modelValue: document.getElementById('modelSelect').value,
        confValue: confInput.value,
        lastResult: lastDetectionResult
    };
    
    try {
        sessionStorage.setItem('detectionAppState', JSON.stringify(state));
        console.log('State saved:', state.fileName);
    } catch (e) {
        console.warn('Failed to save state:', e);
    }
}

// Restore state from sessionStorage
function restoreState() {
    try {
        const stateStr = sessionStorage.getItem('detectionAppState');
        if (!stateStr) return;
        
        const state = JSON.parse(stateStr);
        console.log('Restoring state:', state.fileName);
        
        // Restore model selection
        if (state.modelValue) {
            document.getElementById('modelSelect').value = state.modelValue;
        }
        
        // Restore confidence value
        if (state.confValue) {
            confInput.value = state.confValue;
            updateConfidenceValue();
        }
        
        // Restore file if available
        if (state.hasFile && state.fileDataUrl && state.fileName) {
            // Convert data URL back to file
            fetch(state.fileDataUrl)
                .then(res => res.blob())
                .then(blob => {
                    const file = new File([blob], state.fileName, { type: blob.type });
                    selectedFile = file;
                    fileDataUrl = state.fileDataUrl;
                    
                    // Show preview
                    previewImg.src = state.fileDataUrl;
                    previewSection.classList.remove('hide');
                    
                    // Show a restoration message
                    showNotification('File restored: ' + state.fileName, 'success');
                    
                    console.log('File restored:', state.fileName);
                })
                .catch(err => {
                    console.warn('Failed to restore file:', err);
                    clearState(); // Clear corrupted state
                });
        }
        
        // Restore last detection result if available
        if (state.lastResult) {
            lastDetectionResult = state.lastResult;
            displayResults(state.lastResult, false); // Don't save state again
        }
        
    } catch (e) {
        console.warn('Failed to restore state:', e);
        clearState(); // Clear corrupted state
    }
}

// Clear stored state
function clearState() {
    try {
        sessionStorage.removeItem('detectionAppState');
        console.log('State cleared');
    } catch (e) {
        console.warn('Failed to clear state:', e);
    }
}

// Show notification to user
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : '#2196F3'};
        color: white;
        padding: 12px 24px;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

function setupEventListeners() {
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => imageInput.click());
    
    // File selection
    imageInput.addEventListener('change', handleFileSelect);
    
    // Form submission
    uploadForm.addEventListener('submit', handleFormSubmit);
    
    // Confidence slider
    confInput.addEventListener('input', updateConfidenceValue);
    
    // Model selection change
    document.getElementById('modelSelect').addEventListener('change', saveState);
    
    // Save state before user navigates away
    window.addEventListener('beforeunload', saveState);
    
    // Clear file button (add this if you want)
    const clearButton = document.createElement('button');
    clearButton.type = 'button';
    clearButton.className = 'btn-secondary btn-sm';
    clearButton.textContent = '✕ Clear';
    clearButton.style.marginLeft = '10px';
    clearButton.onclick = clearSelectedFile;
    
    // Add clear button to preview section
    const previewActions = previewSection.querySelector('button');
    if (previewActions && previewActions.parentNode) {
        previewActions.parentNode.appendChild(clearButton);
    }
}

function clearSelectedFile() {
    selectedFile = null;
    fileDataUrl = null;
    imageInput.value = ''; // Clear file input
    previewSection.classList.add('hide');
    resultsSection.classList.add('hide');
    clearState();
    showNotification('File cleared', 'info');
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file.', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Convert to data URL for storage
    const reader = new FileReader();
    reader.onload = function(e) {
        fileDataUrl = e.target.result;
        previewImg.src = fileDataUrl;
        previewSection.classList.remove('hide');
        
        // Save state immediately after file selection
        saveState();
        
        showNotification(`File selected: ${file.name}`, 'success');
    };
    reader.readAsDataURL(file);
}

function updateConfidenceValue() {
    rangeValue.textContent = confInput.value;
    // Save state when confidence changes
    if (selectedFile) {
        saveState();
    }
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    if (!selectedFile) {
        showNotification('Please select an image first.', 'error');
        return;
    }
    
    // Show loading, hide results
    loadingSection.classList.remove('hide');
    resultsSection.classList.add('hide');
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model', document.getElementById('modelSelect').value);
        formData.append('conf', confInput.value);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Upload failed');
        }
        
        const result = await response.json();
        lastDetectionResult = result;
        displayResults(result, true); // Save state after successful detection
        
        showNotification('Detection completed!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        loadingSection.classList.add('hide');
    }
}

function displayResults(data, shouldSaveState = true) {
    console.log('Display results:', data);
    
    // Display the result image
    if (data.ui_image_url) {
        resultImg.src = data.ui_image_url;
    } else if (data.public_image_url) {
        resultImg.src = data.public_image_url;
    } else if (data.image_filename) {
        resultImg.src = `http://localhost:8001/outputs/images/${data.image_filename}`;
    }
    
    // Add error handler for image loading
    resultImg.onerror = function() {
        console.error('Failed to load image:', this.src);
        this.alt = 'Image failed to load';
        this.style.display = 'block';
        this.style.background = '#f0f0f0';
        this.style.minHeight = '200px';
        this.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">Image could not be loaded</div>';
    };
    
    resultImg.onload = function() {
        console.log('Image loaded successfully:', this.src);
    };
    
    // Update summary
    const summary = document.getElementById('summary');
    summary.innerHTML = `
        <div class="summary-badge">
            Model: ${data.model.toUpperCase()} | Threshold: ${data.confidence_threshold}
        </div>
        <div class="chip-row">
            <span class="chip">${data.total_objects} objects detected</span>
        </div>
    `;
    
    // Update table
    const tableBody = document.querySelector('#detectionsTable tbody');
    tableBody.innerHTML = '';
    
    data.detections.forEach((detection, index) => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td><span class="index-badge">${index + 1}</span></td>
            <td><strong>${detection.class}</strong></td>
            <td>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${detection.confidence * 100}%"></div>
                    <div class="confidence-text">${Math.round(detection.confidence * 100)}%</div>
                </div>
            </td>
            <td>
                <code>(${detection.bbox.x1}, ${detection.bbox.y1}) -> (${detection.bbox.x2}, ${detection.bbox.y2})</code>
            </td>
        `;
    });
    
    // Setup download buttons
    setupDownloadButtons(data);
    
    // Show results
    resultsSection.classList.remove('hide');
    
    // Save state after displaying results
    if (shouldSaveState) {
        saveState();
    }
}

function setupDownloadButtons(data) {
    const downloadJson = document.getElementById('downloadJson');
    const downloadImage = document.getElementById('downloadImage');
    
    downloadJson.onclick = () => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `detection_${data.image_id}_${data.model}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        // Don't save state here, just show notification
        showNotification('JSON downloaded', 'success');
    };
    
    downloadImage.onclick = () => {
        const a = document.createElement('a');
        a.href = data.ui_image_url || data.public_image_url || `http://localhost:8001/outputs/images/${data.image_filename}`;
        a.download = data.image_filename || `detection_${data.image_id}_${data.model}.jpg`;
        a.click();
        
        // Don't save state here, just show notification
        showNotification('Image downloaded', 'success');
    };
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
