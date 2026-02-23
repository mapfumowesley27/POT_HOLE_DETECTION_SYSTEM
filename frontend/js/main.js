
const API_BASE_URL = window.APP_CONFIG ? window.APP_CONFIG.API_BASE_URL : 'http://localhost:5000';
// Initialize maps and global variables
let mainMap, reportMap, heatLayer;
let markers = [];

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initMaps();
    setupEventListeners();
    loadPotholeData();
    loadDashboardStats();
    setupRealTimeUpdates();
});

function initMaps() {
    // Initialize main map (Zimbabwe centered)
    mainMap = L.map('mainMap').setView([-17.8252, 31.0335], 12); // Harare coordinates

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(mainMap);

    // Initialize report map
    reportMap = L.map('reportMap').setView([-17.8252, 31.0335], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(reportMap);

    // Add click handler to report map
    reportMap.on('click', function(e) {
        document.getElementById('location').value =
            `${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`;
        window.selectedLocation = e.latlng;
    });
}

function setupEventListeners() {
    // Report form submission
    document.getElementById('reportForm').addEventListener('submit', handleReportSubmit);

    // Get current location button
    document.getElementById('getCurrentLocation').addEventListener('click', getCurrentLocation);
}

function getCurrentLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;

            document.getElementById('location').value = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
            window.selectedLocation = {lat, lng};

            // Center report map on location
            reportMap.setView([lat, lng], 15);

            // Add marker
            L.marker([lat, lng]).addTo(reportMap);
        });
    } else {
        alert('Geolocation is not supported by this browser.');
    }
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

        // Update heatmap
        updateHeatmap(potholes);

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

function updateHeatmap(potholes) {
    // Create heatmap data (Objective 2)
    const heatData = potholes
        .filter(p => p.status !== 'repaired')
        .map(p => [p.latitude, p.longitude, 1]); // Intensity 1 for each pothole

    if (heatLayer) {
        mainMap.removeLayer(heatLayer);
    }

    heatLayer = L.heatLayer(heatData, {
        radius: 25,
        blur: 15,
        maxZoom: 17,
        gradient: {0.4: 'blue', 0.6: 'lime', 0.8: 'red'}
    }).addTo(mainMap);
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

// Add this function to handle image to base64 conversion
function imageToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

// Update the handleReportSubmit function
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
        if (imageFile) {
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

// Image preview function
function previewImage(event) {
    const previewContainer = document.getElementById('imagePreviewContainer');
    const preview = document.getElementById('imagePreview');
    const analysis = document.getElementById('imageAnalysis');

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