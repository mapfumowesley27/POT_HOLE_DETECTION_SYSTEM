
const API_BASE_URL = window.APP_CONFIG ? window.APP_CONFIG.API_BASE_URL : 'http://localhost:5000';
// Make it globally accessible
window.API_BASE_URL = API_BASE_URL;

// Initialize maps and global variables
let mainMap, reportMap, heatMap, heatLayer;
let markers = [];

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initMaps();
    setupEventListeners();
    loadPotholeData();
    loadDashboardStats();
    setupRealTimeUpdates();
});

function showNotification(message, type) {
    // Simple notification
    alert(message);
}

function initMaps() {
    console.log("Initializing maps...");
    try {
        // Initialize main map (Zimbabwe centered)
        const mainMapEl = document.getElementById('mainMap');
        if (mainMapEl) {
            mainMap = L.map('mainMap').setView([-17.8252, 31.0335], 12); // Harare coordinates
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(mainMap);
        }

        // Initialize report map
        const reportMapEl = document.getElementById('reportMap');
        if (reportMapEl) {
            reportMap = L.map('reportMap').setView([-17.8252, 31.0335], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(reportMap);

            reportMap.on('click', function(e) {
                document.getElementById('location').value =
                    `${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`;
                window.selectedLocation = e.latlng;
            });
        }

        // Initialize heatmap map
        const heatMapEl = document.getElementById('heatmapContainer');
        if (heatMapEl) {
            heatMap = L.map('heatmapContainer').setView([-17.8252, 31.0335], 12);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(heatMap);
        }
    } catch (e) {
        console.error("Error initializing maps:", e);
    }
}

function setupEventListeners() {
    console.log("Setting up event listeners...");
    // Report form submission
    const reportForm = document.getElementById('reportForm');
    if (reportForm) {
        reportForm.addEventListener('submit', handleReportSubmit);
    }

    // Get current location button
    const getCurrentLocBtn = document.getElementById('getCurrentLocation');
    if (getCurrentLocBtn) {
        getCurrentLocBtn.addEventListener('click', getCurrentLocation);
    }

    // Camera capture listeners
    const startCaptureBtn = document.getElementById('startCapture');
    if (startCaptureBtn) {
        console.log("Start capture button found, adding listener");
        // Check for secure context
        if (!window.isSecureContext && window.location.hostname !== 'localhost') {
            console.warn("Camera access requires a secure context (HTTPS or localhost).");
            startCaptureBtn.classList.add('btn-disabled-secure');
            startCaptureBtn.title = "Camera access requires HTTPS";
        }
        
        startCaptureBtn.addEventListener('click', function() {
            console.log("Start capture button clicked");
            if (!window.isSecureContext && window.location.hostname !== 'localhost') {
                alert("Camera access is blocked because this page is not served over HTTPS. Please use HTTPS or localhost to enable camera features.");
                return;
            }
            startCameraFeed();
        });
    } else {
        console.error("Start capture button NOT found");
    }

    const captureBtn = document.getElementById('captureBtn');
    if (captureBtn) {
        captureBtn.addEventListener('click', function() {
            console.log("Capture button clicked");
            captureLivePhoto();
        });
    }

    const stopCameraBtn = document.getElementById('stopCamera');
    if (stopCameraBtn) {
        stopCameraBtn.addEventListener('click', function() {
            console.log("Stop camera button clicked");
            stopCameraFeed();
        });
    }
}

let videoStream = null;

async function startCameraFeed() {
    const video = document.getElementById('video');
    const container = document.getElementById('cameraContainer');

    if (!window.isSecureContext && window.location.hostname !== 'localhost') {
        alert("SECURITY ERROR: Camera access (getUserMedia) is only allowed in Secure Contexts (HTTPS or localhost). \n\nYour current environment: " + window.location.protocol + "//" + window.location.hostname);
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Your browser does not support camera access. Please use a modern browser (Chrome, Firefox, Safari) and ensure you're using HTTPS.");
        return;
    }

    // Show container first so video is visible when stream starts
    container.style.display = 'block';
    document.getElementById('startCapture').disabled = true;

    try {
        console.log("Requesting camera access with constraints:", { video: { facingMode: "environment" } });
        const constraints = { 
            video: { facingMode: "environment" }, // Prefer back camera
            audio: false 
        };
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        console.log("Camera access granted. Stream ID:", videoStream.id);
        const videoTracks = videoStream.getVideoTracks();
        if (videoTracks.length > 0) {
            console.log("Video track label:", videoTracks[0].label);
            console.log("Video track settings:", videoTracks[0].getSettings());
        }

        console.log("Setting video.srcObject...");
        video.srcObject = videoStream;
        
        // Wait for metadata to be loaded then play
        video.onloadedmetadata = () => {
            console.log("Video metadata loaded. Dimensions:", video.videoWidth, "x", video.videoHeight);
            video.play()
                .then(() => {
                    console.log("Camera started successfully and playing (onloadedmetadata)");
                })
                .catch(e => {
                    console.error("Video play failed (onloadedmetadata):", e);
                });
        };
        
        // Also try playing immediately in case metadata is already loaded
        video.play().then(() => {
            console.log("Camera playing immediately after setting srcObject");
        }).catch(e => {
            console.warn("Immediate play failed (expected if metadata not yet loaded):", e.message);
        });

    } catch (err) {
        console.error("Camera error:", err);
        container.style.display = 'none';
        document.getElementById('startCapture').disabled = false;
        
        let errorMsg = "Could not access camera: ";
        if (err.name === 'NotAllowedError') {
            errorMsg += "Permission denied. Please allow camera access.";
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            errorMsg += "No camera found on this device.";
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            errorMsg += "Camera is already in use by another application.";
        } else {
            errorMsg += err.name + ": " + err.message;
        }
        alert(errorMsg);
    }
}

function stopCameraFeed() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    document.getElementById('cameraContainer').style.display = 'none';
    document.getElementById('startCapture').disabled = false;
}

function retakePhoto() {
    const previewContainer = document.getElementById('imagePreviewContainer');
    if (previewContainer) previewContainer.style.display = 'none';
    startCameraFeed();
}

function captureLivePhoto() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const preview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('imagePreviewContainer');

    if (!video || !canvas || !preview || !previewContainer) {
        console.error("Missing camera elements");
        return;
    }

    // Draw frame to canvas
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to image data
    const dataUrl = canvas.toDataURL('image/jpeg');
    preview.src = dataUrl;
    previewContainer.style.display = 'block';
    
    // Update preview buttons if they don't exist
    let actionBtns = previewContainer.querySelector('.preview-actions');
    if (!actionBtns) {
        actionBtns = document.createElement('div');
        actionBtns.className = 'preview-actions mt-2';
        actionBtns.innerHTML = `
            <button type="button" class="btn btn-glass-danger btn-sm" onclick="retakePhoto()">
                <i class="fas fa-redo me-1"></i> Retake
            </button>
        `;
        previewContainer.appendChild(actionBtns);
    }
    
    // Store dataUrl for submission
    window.capturedImageData = dataUrl;
    
    // Clear file input if used
    const fileInput = document.getElementById('image');
    if (fileInput) fileInput.value = '';

    // Take coordinates as well upon capture (requirement)
    getCurrentLocation();

    // Stop the camera
    stopCameraFeed();
    
    showNotification("Photo captured and location updated!", 'success');
}

async function getCurrentLocation() {
    console.log("Get current location triggered");
    if (!navigator.geolocation) {
        alert("Geolocation is not supported by your browser");
        return;
    }

    const btn = document.getElementById('getCurrentLocation');
    const originalContent = btn ? btn.innerHTML : '';
    if (btn) {
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Locating...';
        btn.disabled = true;
    }

    return new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                console.log("Position acquired:", position.coords.latitude, position.coords.longitude);
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                window.selectedLocation = { lat, lng };
                
                const locInput = document.getElementById('location');
                if (locInput) locInput.value = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                
                if (reportMap) {
                    reportMap.setView([lat, lng], 15);
                    L.marker([lat, lng]).addTo(reportMap);
                }
                
                if (btn) {
                    btn.innerHTML = originalContent;
                    btn.disabled = false;
                }
                resolve(position);
            },
            (error) => {
                console.error("Geolocation error:", error);
                // Don't alert if it's an automatic call from capture
                if (btn) {
                    alert("Error getting location: " + error.message);
                    btn.innerHTML = originalContent;
                    btn.disabled = false;
                }
                resolve(null); // Resolve with null so caller can continue
            },
            { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
        );
    });
}

async function loadPotholeData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes`);
        const potholes = await response.json();

        // Clear existing markers
        markers.forEach(marker => mainMap.removeLayer(marker));
        markers = [];

        // Add new markers with size-based styling (Objective 1)
        potholes.forEach(pothole => {
            const marker = L.circleMarker([pothole.latitude, pothole.longitude], {
                radius: getMarkerSize(pothole.size_classification),
                color: getMarkerColor(pothole.size_classification),
                fillColor: getMarkerColor(pothole.size_classification),
                fillOpacity: 0.8
            }).addTo(mainMap);

            marker.bindPopup(`
                <b>Pothole Report</b><br>
                Size: ${pothole.size_classification}<br>
                Diameter: ${pothole.diameter?.toFixed(2)}m<br>
                Status: ${pothole.status}<br>
                Reported: ${new Date(pothole.reported_at).toLocaleDateString()}
            `);

            markers.push(marker);
        });

        // Update heatmap with density data
        await updateHeatmap(potholes);

    } catch (error) {
        console.error('Error loading pothole data:', error);
    }
}

function getMarkerSize(size) {
    switch(size) {
        case 'small': return 6;
        case 'medium': return 8;
        case 'large': return 10;
        default: return 6;
    }
}

function getMarkerColor(size) {
    switch(size) {
        case 'small': return '#ffc107';
        case 'medium': return '#fd7e14';
        case 'large': return '#dc3545';
        default: return '#6c757d';
    }
}

async function updateHeatmap(potholes) {
    try {
        // Fetch density data from API (calculates potholes per 100m²)
        const response = await fetch(`${API_BASE_URL}/api/potholes/density`);
        const densityData = await response.json();

        let heatData;
        
        // Use density API data if available, otherwise fallback to simple pothole data
        if (densityData.heatmap_data && densityData.heatmap_data.length > 0) {
            // Use the calculated density heatmap data
            heatData = densityData.heatmap_data;
            
            // Log high density areas for debugging
            if (densityData.high_density_areas && densityData.high_density_areas.length > 0) {
                console.log('High density areas detected:', densityData.high_density_areas);
            }
        } else {
            // Fallback to simple pothole-based heatmap
            heatData = potholes
                .filter(p => p.status !== 'repaired')
                .map(p => [p.latitude, p.longitude, 1]);
        }

        // Remove existing heat layer from heatMap
        if (heatLayer) {
            heatMap.removeLayer(heatLayer);
        }

        // Invalidate map size to ensure proper rendering
        setTimeout(() => {
            heatMap.invalidateSize();
        }, 100);

        // Create heatmap with intensity based on density - add to heatMap
        heatLayer = L.heatLayer(heatData, {
            radius: 30,
            blur: 20,
            maxZoom: 17,
            // Red gradient for high density areas (5+ per 100m²)
            gradient: {0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        }).addTo(heatMap);
        
        return densityData;
    } catch (error) {
        console.error('Error loading density data:', error);
        // Fallback to simple heatmap
        const heatData = potholes
            .filter(p => p.status !== 'repaired')
            .map(p => [p.latitude, p.longitude, 1]);

        if (heatLayer) {
            heatMap.removeLayer(heatLayer);
        }

        setTimeout(() => {
            heatMap.invalidateSize();
        }, 100);

        heatLayer = L.heatLayer(heatData, {
            radius: 30,
            blur: 20,
            maxZoom: 17,
            gradient: {0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        }).addTo(heatMap);
    }
}

async function loadDashboardStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes`);
        const potholes = await response.json();

        // Calculate stats
        const critical = potholes.filter(p => p.size_classification === 'large' && p.status !== 'repaired').length;
        const pending = potholes.filter(p => p.status === 'pending').length;
        const repaired = potholes.filter(p => p.status === 'repaired').length;

        // Update UI
        document.getElementById('criticalCount').textContent = critical;
        document.getElementById('pendingCount').textContent = pending;
        document.getElementById('repairedCount').textContent = repaired;

        // Load alerts
        loadAlerts();

    } catch (error) {
        console.error('Error loading dashboard stats:', error);
    }
}

async function loadAlerts() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/alerts`);
        const alerts = await response.json();

        document.getElementById('alertCount').textContent = alerts.length;

        const alertsList = document.getElementById('alertsList');
        alertsList.innerHTML = alerts.map(alert => `
            <div class="alert alert-${alert.type === 'large_pothole' ? 'danger' : 'warning'}">
                ${alert.message}
                <br>
                <small>${new Date(alert.sent_at).toLocaleString()}</small>
                ${!alert.acknowledged ?
                    `<button class="btn btn-sm btn-outline-primary mt-2" onclick="acknowledgeAlert(${alert.id})">Acknowledge</button>`
                    : ''}
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading alerts:', error);
    }
}

async function acknowledgeAlert(alertId) {
    try {
        await fetch(`${API_BASE_URL}/api/alerts/${alertId}/acknowledge`, {
            method: 'POST'
        });
        loadAlerts();
    } catch (error) {
        console.error('Error acknowledging alert:', error);
    }
}

function setupRealTimeUpdates() {
    // Refresh data every 30 seconds
    setInterval(() => {
        loadPotholeData();
        loadDashboardStats();
    }, 30000);
}

//  function to handle image to base64 conversion
function imageToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

// Update the handleReportSubmit function
function previewImage(event) {
    console.log("Image preview triggered");
    const input = event.target;
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('imagePreview');
            const previewContainer = document.getElementById('imagePreviewContainer');
            if (preview && previewContainer) {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
                console.log("Image preview displayed");
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

async function handleReportSubmit(e) {
    e.preventDefault();

    if (!window.selectedLocation) {
        alert('Please select a location on the map');
        return;
    }

    // Show loading
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.innerHTML = '<span class="loading-spinner"></span> Analyzing image...';
    submitBtn.disabled = true;

    try {
        const formData = {
            latitude: window.selectedLocation.lat,
            longitude: window.selectedLocation.lng,
            reporter: 'anonymous'
        };

        // Handle image upload
        const imageFile = document.getElementById('image').files[0];
        if (window.capturedImageData) {
            formData.image_base64 = window.capturedImageData;
        } else if (imageFile) {
            // Convert image to base64
            const base64Image = await imageToBase64(imageFile);
            formData.image_base64 = base64Image;
        }

        const response = await fetch(`${API_BASE_URL}/api/report`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (data.success) {
            let message = 'Pothole reported successfully!';
            if (data.detection_result) {
                if (data.detection_result.pothole_detected) {
                    message = `Pothole detected! Size: ${data.detection_result.size_classification}, Diameter: ${data.detection_result.diameter.toFixed(2)}m`;
                } else {
                    message = 'No pothole detected in the image. Report submitted for review.';
                }
            }

            if (data.alert_generated) {
                message += ' ⚠️ Alert sent to authorities!';
            }

            alert(message);

            // Reset form
            document.getElementById('reportForm').reset();
            document.getElementById('location').value = '';
            window.selectedLocation = null;
            window.capturedImageData = null;

            // Clear image preview
            const previewContainer = document.getElementById('imagePreviewContainer');
            if (previewContainer) {
                previewContainer.style.display = 'none';
            }

            // Reload data
            loadPotholeData();
            loadDashboardStats();
            loadAlerts();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error submitting report: ' + error.message);
    } finally {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
}

async function getCurrentLocation() {
    console.log("Get current location triggered");
    if (!navigator.geolocation) {
        alert("Geolocation is not supported by your browser");
        return;
    }

    const btn = document.getElementById('getCurrentLocation');
    const originalContent = btn ? btn.innerHTML : '';
    if (btn) {
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Locating...';
        btn.disabled = true;
    }

    return new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                console.log("Position acquired:", position.coords.latitude, position.coords.longitude);
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                window.selectedLocation = { lat, lng };
                
                const locInput = document.getElementById('location');
                if (locInput) locInput.value = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                
                if (reportMap) {
                    reportMap.setView([lat, lng], 15);
                    L.marker([lat, lng]).addTo(reportMap);
                }
                
                if (btn) {
                    btn.innerHTML = originalContent;
                    btn.disabled = false;
                }
                resolve(position);
            },
            (error) => {
                console.error("Geolocation error:", error);
                if (btn) {
                    alert("Error getting location: " + error.message);
                    btn.innerHTML = originalContent;
                    btn.disabled = false;
                }
                resolve(null);
            },
            { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
        );
    });
}

// Image preview function
function previewImage(event) {
    const previewContainer = document.getElementById('imagePreviewContainer');
    const preview = document.getElementById('imagePreview');
    const analysis = document.getElementById('imageAnalysis');

    // Clear any previous captured live image when a file is selected
    window.capturedImageData = null;

    if (event.target.files && event.target.files[0]) {
        const reader = new FileReader();

        reader.onload = async function(e) {
            preview.src = e.target.result;
            previewContainer.style.display = 'block';

            // Optional: Quick preview analysis
            analysis.innerHTML = '<small class="text-muted">Image ready for analysis</small>';
        }

        reader.readAsDataURL(event.target.files[0]);
    } else {
        previewContainer.style.display = 'none';
    }
}