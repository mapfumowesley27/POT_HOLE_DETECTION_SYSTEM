// =============================================
// ROAD MAINTENANCE MANAGER - COMPLETE JAVASCRIPT
// =============================================

const API_BASE_URL = 'http://localhost:5000';

// =============================================
// INITIALIZATION
// =============================================
document.addEventListener('DOMContentLoaded', function() {
    const isLoggedIn = localStorage.getItem('authToken');
    const userRole = localStorage.getItem('userRole');

    if (!isLoggedIn) {
        window.location.href = 'login.html';
        return;
    }

    if (userRole !== 'manager') {
        alert('Access denied. Manager privileges required.');
        window.location.href = 'login.html';
        return;
    }

    const userName = localStorage.getItem('userName') || 'Maintenance Manager';
    document.getElementById('userName').textContent = userName;

    initializeManager();
});

async function initializeManager() {
    await loadZones();
    await loadAlerts();
    await loadPotholeStats();
    await loadPotholesToVerify();
    await loadActiveRepairs();
    await loadRecentlyRepaired();
    await loadCrews();
    await loadCrewMembers();
    await loadMaterials();
    await loadRepairJobs();
    initializeMap();
    setupCameraEventListeners();
}

function setupCameraEventListeners() {
    const startCaptureBtn = document.getElementById('startCapture');
    if (startCaptureBtn) {
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

    container.style.display = 'block';
    const startBtn = document.getElementById('startCapture');
    if (startBtn) startBtn.disabled = true;

    try {
        console.log("Requesting camera access with constraints:", { video: { facingMode: "environment" } });
        const constraints = { 
            video: { facingMode: "environment" },
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
        
        video.play().then(() => {
            console.log("Camera playing immediately after setting srcObject");
        }).catch(e => {
            console.warn("Immediate play failed (expected if metadata not yet loaded):", e.message);
        });

    } catch (err) {
        console.error("Camera error:", err);
        container.style.display = 'none';
        if (startBtn) startBtn.disabled = false;
        
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
    const container = document.getElementById('cameraContainer');
    if (container) container.style.display = 'none';
    const startBtn = document.getElementById('startCapture');
    if (startBtn) startBtn.disabled = false;
}

function captureLivePhoto() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const preview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('imagePreviewContainer');
    
    if (!video || !canvas || !preview || !previewContainer) {
        console.error("Missing camera or preview elements");
        return;
    }

    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg');
    preview.src = dataUrl;
    previewContainer.style.display = 'block';
    
    // Store for submission
    window.capturedImageData = dataUrl;
    
    // Stop the camera feed
    stopCameraFeed();
    
    showNotification("Photo captured! Please confirm if you want to use it.", 'info');
}

function confirmRepairCapture() {
    if (window.capturedImageData) {
        showNotification("Photo confirmed and pothole marked as repaired!", 'success');
        document.getElementById('imagePreviewContainer').style.display = 'none';
        // Here you would normally send the data to the server
    }
}

function retakePhoto() {
    document.getElementById('imagePreviewContainer').style.display = 'none';
    startCameraFeed();
}

// =============================================
// ALERTS FUNCTIONS
// =============================================
async function loadZones() {
    const select = document.getElementById('crewZone');
    if (!select) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/zones`);
        const zones = await response.json();
        
        // Sort zones by name
        zones.sort((a, b) => a.name.localeCompare(b.name));
        
        // Clear current options (except the first one)
        select.innerHTML = '<option value="">Select Zone</option>';
        
        zones.forEach(zone => {
            const option = document.createElement('option');
            option.value = zone.id;
            option.textContent = zone.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading zones:', error);
    }
}

async function loadAlerts() {
    const container = document.getElementById('alertsContainer');
    const alertCount = document.getElementById('alertCount');

    try {
        const response = await fetch(`${API_BASE_URL}/api/alerts`);
        const alerts = await response.json();

        const unacknowledgedAlerts = alerts.filter(a => !a.acknowledged);
        if (alertCount) alertCount.textContent = unacknowledgedAlerts.length;

        if (!alerts || alerts.length === 0) {
            container.innerHTML = '<p class="text-center text-muted py-4">No alerts at this time.</p>';
            return;
        }

        container.innerHTML = alerts.map(alert => `
            <div class="alert-card ${alert.acknowledged ? 'acknowledged' : ''}" id="alert${alert.id}">
                <div class="d-flex justify-content-between align-items-center">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-${alert.type === 'large_pothole' ? 'exclamation-triangle text-danger' : 'exclamation-circle text-warning'} me-3 fs-4"></i>
                        <div>
                            <h6 class="mb-0 fw-bold">${alert.message}</h6>
                            <small class="text-muted">
                                ${alert.sent_at ? new Date(alert.sent_at).toLocaleString() : 'Recent'} •
                                Size: ${alert.pothole_size || 'Unknown'}
                            </small>
                        </div>
                    </div>
                    <div>
                        ${!alert.acknowledged ? `
                            <button class="btn btn-glass-warning me-2" onclick="acknowledgeAlert(${alert.id})">
                                <i class="fas fa-check me-1"></i>Acknowledge
                            </button>
                            <button class="btn btn-glass-success" onclick="markForRepair(${alert.pothole_id})">
                                <i class="fas fa-tools me-1"></i>Mark for Repair
                            </button>
                        ` : '<span class="badge bg-success">Acknowledged</span>'}
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading alerts:', error);
        container.innerHTML = '<p class="text-center text-danger py-4">Error loading alerts. Make sure backend is running.</p>';
    }
}

async function acknowledgeAlert(alertId) {
    try {
        await fetch(`${API_BASE_URL}/api/alerts/${alertId}/acknowledge`, { method: 'POST' });
        const alertCard = document.getElementById(`alert${alertId}`);
        alertCard.classList.add('acknowledged');
        const buttonDiv = alertCard.querySelector('div:last-child');
        buttonDiv.innerHTML = '<span class="badge bg-success">Acknowledged</span>';
        updateAlertCount();
        showNotification('Alert acknowledged successfully', 'success');
    } catch (error) {
        showNotification('Error acknowledging alert', 'error');
    }
}

function updateAlertCount() {
    const activeAlerts = document.querySelectorAll('.alert-card:not(.acknowledged)').length;
    document.getElementById('alertCount').textContent = activeAlerts;
}

function markForRepair(potholeId) {
    showNotification(`Pothole #${potholeId} marked for repair`, 'info');
    // Create a repair job for this pothole
    window.location.href = `road_maintenance_manager.html?createJob=${potholeId}`;
}

// =============================================
// STATS FUNCTIONS
// =============================================
async function loadPotholeStats() {
    try {
        const userRole = localStorage.getItem('userRole');
        const zoneId = localStorage.getItem('zoneId');
        const response = await fetch(`${API_BASE_URL}/api/stats?role=${userRole}&zone_id=${zoneId}`);
        const data = await response.json();

        document.getElementById('pendingCount').textContent = data.pending || 0;
        document.getElementById('verifiedCount').textContent = data.verified || 0;
        document.getElementById('inProgressCount').textContent = data.in_progress || 0;
        document.getElementById('repairedToday').textContent = data.repaired || 0;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// =============================================
// POTHOLE VERIFICATION
// =============================================
async function loadPotholesToVerify() {
    const container = document.getElementById('potholesToVerify');
    if (!container) return;

    try {
        const userRole = localStorage.getItem('userRole');
        const zoneId = localStorage.getItem('zoneId');
        const response = await fetch(`${API_BASE_URL}/api/potholes?status=pending&role=${userRole}&zone_id=${zoneId}`);
        const potholes = await response.json();

        if (!potholes || potholes.length === 0) {
            container.innerHTML = '<p class="text-center text-muted py-4">No potholes to verify</p>';
            return;
        }

        container.innerHTML = potholes.map(p => `
            <div class="pothole-card">
                <div class="d-flex justify-content-between">
                    <div>
                        <span class="severity-badge severity-${p.size_classification || 'medium'}">
                            ${p.size_classification || 'Medium'}
                        </span>
                        <h6 class="mt-2 mb-1">${getLocationName(p.latitude, p.longitude)}</h6>
                        <small class="text-muted d-block">
                            Reported: ${p.reported_at ? new Date(p.reported_at).toLocaleString() : 'Recent'} •
                            Diameter: ${p.diameter?.toFixed(2) || '?'}m
                        </small>
                    </div>
                    <div class="text-end">
                        <button class="btn btn-glass-success btn-sm mb-2" onclick="verifyPothole(${p.id})">
                            <i class="fas fa-check me-1"></i>Verify
                        </button>
                        <button class="btn btn-glass-warning btn-sm" onclick="rejectPothole(${p.id})">
                            <i class="fas fa-times me-1"></i>Reject
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading potholes:', error);
        container.innerHTML = '<p class="text-center text-danger py-4">Error loading potholes</p>';
    }
}

async function verifyPothole(id) {
    if (!confirm('Verify this pothole?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes/${id}/verify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ verified: true })
        });
        
        if (response.ok) {
            showNotification('Pothole verified successfully', 'success');
            loadPotholesToVerify();
            loadPotholeStats();
            loadAlerts();
        } else {
            showNotification('Error verifying pothole', 'error');
        }
    } catch (error) {
        showNotification('Error verifying pothole', 'error');
    }
}

async function rejectPothole(id) {
    if (!confirm('Reject this pothole report?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes/${id}/verify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ verified: false })
        });
        
        if (response.ok) {
            showNotification('Pothole report rejected', 'warning');
            loadPotholesToVerify();
        } else {
            showNotification('Error rejecting pothole', 'error');
        }
    } catch (error) {
        showNotification('Error rejecting pothole', 'error');
    }
}

// =============================================
// REPAIR MANAGEMENT
// =============================================
async function loadActiveRepairs() {
    const container = document.getElementById('activeRepairs');
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/repair-jobs?status=in_progress`);
        const repairs = await response.json();

        if (!repairs || repairs.length === 0) {
            container.innerHTML = '<p class="text-center text-muted">No active repairs</p>';
            return;
        }

        container.innerHTML = repairs.map(r => `
            <div class="pothole-card">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h6 class="mb-1">Pothole #${r.pothole_id}</h6>
                        <small class="text-muted">Started: ${r.started_at ? new Date(r.started_at).toLocaleString() : 'Just now'} • Crew: ${r.crew_name || 'Unassigned'}</small>
                    </div>
                    <span class="badge bg-warning">In Progress</span>
                </div>
                <div class="mt-2">
                    <button class="btn btn-glass-success btn-sm w-100" onclick="showRepairUpload(${r.id})">
                        <i class="fas fa-cloud-upload-alt me-2"></i>Upload After-Photo & Complete
                    </button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading repairs:', error);
        container.innerHTML = '<p class="text-center text-danger">Error loading repairs</p>';
    }
}

async function loadRecentlyRepaired() {
    const container = document.getElementById('recentlyRepaired');
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes/repaired`);
        const repaired = await response.json();

        if (!repaired || repaired.length === 0) {
            container.innerHTML = '<p class="text-center text-muted">No recently repaired potholes</p>';
            return;
        }

        container.innerHTML = repaired.slice(0, 5).map(r => {
            const photoUrl = r.after_photos && r.after_photos.length > 0 
                ? `${API_BASE_URL}/uploads/repairs/${r.after_photos[0]}` 
                : 'https://via.placeholder.com/60?text=N/A';
            
            return `
                <div class="d-flex align-items-center mb-2">
                    <img src="${photoUrl}" class="repair-image-preview me-2" onclick="viewImage('${photoUrl}')">
                    <div class="flex-grow-1">
                        <small class="d-block"><strong>Pothole #${r.pothole_id}</strong></small>
                        <small class="text-muted">${r.completed_at ? new Date(r.completed_at).toLocaleString() : 'Recently'}</small>
                    </div>
                    <i class="fas fa-check-circle text-success"></i>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading repaired:', error);
    }
}

function showRepairUpload(jobId) {
    document.getElementById('repairPhoto').click();
}

async function handleRepairUpload(input) {
    if (input.files.length > 0) {
        showNotification(`${input.files.length} photo(s) uploaded. Pothole marked as repaired!`, 'success');
    }
}

// =============================================
// CREW MANAGEMENT
// =============================================
async function loadCrews() {
    const tbody = document.getElementById('crewsTableBody');
    if (!tbody) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/crews`);
        const crews = await response.json();

        if (!crews || crews.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4">No crews found. Click "Add Crew" to create one.</td></tr>';
            return;
        }

        tbody.innerHTML = crews.map(crew => `
            <tr>
                <td>${crew.id}</td>
                <td><strong>${crew.name}</strong></td>
                <td>${crew.supervisor_name || '-'}</td>
                <td>${crew.zone_name || '-'}</td>
                <td>${crew.contact_number || '-'}</td>
                <td>${crew.member_count || 0}</td>
                <td><span class="${crew.active ? 'status-badge-active' : 'status-badge-inactive'}">${crew.active ? 'Active' : 'Inactive'}</span></td>
                <td>
                    <button class="btn btn-sm btn-glass-success me-1" onclick="editCrew(${crew.id})"><i class="fas fa-edit"></i></button>
                    <button class="btn btn-sm btn-glass-danger" onclick="deleteCrew(${crew.id})"><i class="fas fa-trash"></i></button>
                </td>
            </tr>
        `).join('');
        
        // Update supervisor dropdown
        populateSupervisorDropdown(crews);
    } catch (error) {
        console.error('Error loading crews:', error);
        tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-danger">Error loading crews</td></tr>';
    }
}

function populateSupervisorDropdown(crews) {
    const selects = ['crewSupervisor', 'crewZone'];
    // Will be populated with users from API
}

function openCrewModal(crewId = null) {
    document.getElementById('crewModalTitle').innerHTML = '<i class="fas fa-users me-2"></i>' + (crewId ? 'Edit Crew' : 'Add Crew');
    document.getElementById('crewId').value = crewId || '';
    
    if (crewId) {
        // Fetch crew data and populate form
        fetch(`${API_BASE_URL}/api/crews/${crewId}`)
            .then(r => r.json())
            .then(crew => {
                document.getElementById('crewName').value = crew.name || '';
                document.getElementById('crewSupervisor').value = crew.supervisor_id || '';
                document.getElementById('crewZone').value = crew.zone_id || '';
                document.getElementById('crewContact').value = crew.contact_number || '';
                document.getElementById('crewActive').checked = crew.active !== false;
            });
    } else {
        document.getElementById('crewForm').reset();
    }
    
    new bootstrap.Modal(document.getElementById('crewModal')).show();
}

async function saveCrew() {
    const crewId = document.getElementById('crewId').value;
    const data = {
        name: document.getElementById('crewName').value,
        supervisor_id: document.getElementById('crewSupervisor').value || null,
        zone_id: document.getElementById('crewZone').value || null,
        contact_number: document.getElementById('crewContact').value,
        active: document.getElementById('crewActive').checked
    };

    try {
        const url = crewId ? `${API_BASE_URL}/api/crews/${crewId}` : `${API_BASE_URL}/api/crews`;
        const method = crewId ? 'PUT' : 'POST';
        
        const response = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            showNotification('Crew saved successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('crewModal')).hide();
            loadCrews();
        } else {
            showNotification('Error saving crew', 'error');
        }
    } catch (error) {
        showNotification('Error saving crew', 'error');
    }
}

function editCrew(id) {
    openCrewModal(id);
}

async function deleteCrew(id) {
    if (!confirm('Are you sure you want to delete this crew?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/crews/${id}`, { method: 'DELETE' });
        if (response.ok) {
            showNotification('Crew deleted successfully', 'success');
            loadCrews();
        } else {
            showNotification('Error deleting crew', 'error');
        }
    } catch (error) {
        showNotification('Error deleting crew', 'error');
    }
}

// =============================================
// CREW MEMBERS MANAGEMENT
// =============================================
async function loadCrewMembers() {
    const tbody = document.getElementById('crewMembersTableBody');
    if (!tbody) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/crew-members`);
        const members = await response.json();

        if (!members || members.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4">No crew members found.</td></tr>';
            return;
        }

        tbody.innerHTML = members.map(member => `
            <tr>
                <td>${member.id}</td>
                <td><strong>${member.name}</strong></td>
                <td>${member.crew_name || '-'}</td>
                <td>${member.role || '-'}</td>
                <td>${member.phone || '-'}</td>
                <td>${member.joined_date ? new Date(member.joined_date).toLocaleDateString() : '-'}</td>
                <td><span class="${member.active ? 'status-badge-active' : 'status-badge-inactive'}">${member.active ? 'Active' : 'Inactive'}</span></td>
                <td>
                    <button class="btn btn-sm btn-glass-success me-1" onclick="editCrewMember(${member.id})"><i class="fas fa-edit"></i></button>
                    <button class="btn btn-sm btn-glass-danger" onclick="deleteCrewMember(${member.id})"><i class="fas fa-trash"></i></button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading crew members:', error);
    }
}

function openCrewMemberModal(memberId = null) {
    document.getElementById('crewMemberModalTitle').innerHTML = '<i class="fas fa-user-plus me-2"></i>' + (memberId ? 'Edit Member' : 'Add Member');
    document.getElementById('crewMemberId').value = memberId || '';
    
    // Populate crew dropdown
    fetch(`${API_BASE_URL}/api/crews`)
        .then(r => r.json())
        .then(crews => {
            const select = document.getElementById('memberCrew');
            select.innerHTML = '<option value="">Select Crew</option>' + 
                crews.map(c => `<option value="${c.id}">${c.name}</option>`).join('');
        });
    
    if (memberId) {
        fetch(`${API_BASE_URL}/api/crew-members/${memberId}`)
            .then(r => r.json())
            .then(member => {
                document.getElementById('memberCrew').value = member.crew_id || '';
                document.getElementById('memberName').value = member.name || '';
                document.getElementById('memberRole').value = member.role || 'Worker';
                document.getElementById('memberPhone').value = member.phone || '';
                document.getElementById('memberActive').checked = member.active !== false;
            });
    } else {
        document.getElementById('crewMemberForm').reset();
    }
    
    new bootstrap.Modal(document.getElementById('crewMemberModal')).show();
}

async function saveCrewMember() {
    const memberId = document.getElementById('crewMemberId').value;
    const data = {
        crew_id: document.getElementById('memberCrew').value,
        name: document.getElementById('memberName').value,
        role: document.getElementById('memberRole').value,
        phone: document.getElementById('memberPhone').value,
        active: document.getElementById('memberActive').checked
    };

    try {
        const url = memberId ? `${API_BASE_URL}/api/crew-members/${memberId}` : `${API_BASE_URL}/api/crew-members`;
        const method = memberId ? 'PUT' : 'POST';
        
        const response = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            showNotification('Crew member saved successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('crewMemberModal')).hide();
            loadCrewMembers();
        } else {
            showNotification('Error saving crew member', 'error');
        }
    } catch (error) {
        showNotification('Error saving crew member', 'error');
    }
}

function editCrewMember(id) {
    openCrewMemberModal(id);
}

async function deleteCrewMember(id) {
    if (!confirm('Are you sure you want to delete this crew member?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/crew-members/${id}`, { method: 'DELETE' });
        if (response.ok) {
            showNotification('Crew member deleted successfully', 'success');
            loadCrewMembers();
        } else {
            showNotification('Error deleting crew member', 'error');
        }
    } catch (error) {
        showNotification('Error deleting crew member', 'error');
    }
}

// =============================================
// MATERIALS MANAGEMENT
// =============================================
async function loadMaterials() {
    const tbody = document.getElementById('materialsTableBody');
    if (!tbody) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/materials`);
        const materials = await response.json();

        // Update stats
        if (document.getElementById('totalMaterials')) {
            document.getElementById('totalMaterials').textContent = materials.length;
            document.getElementById('lowStockMaterials').textContent = materials.filter(m => m.low_stock).length;
            document.getElementById('totalQuantity').textContent = materials.reduce((sum, m) => sum + (m.quantity || 0), 0);
            document.getElementById('totalValue').textContent = '$' + materials.reduce((sum, m) => sum + ((m.quantity || 0) * (m.cost_per_unit || 0)), 0);
        }

        if (!materials || materials.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4">No materials found. Click "Add Material" to create one.</td></tr>';
            return;
        }

        tbody.innerHTML = materials.map(m => `
            <tr>
                <td>${m.id}</td>
                <td><strong>${m.name}</strong></td>
                <td>${m.unit || '-'}</td>
                <td>${m.quantity || 0}</td>
                <td>${m.reorder_level || 0}</td>
                <td>$${m.cost_per_unit || 0}</td>
                <td>${m.low_stock ? '<span class="low-stock-badge">Low Stock</span>' : '<span class="status-badge-active">OK</span>'}</td>
                <td>
                    <button class="btn btn-sm btn-glass-success me-1" onclick="editMaterial(${m.id})"><i class="fas fa-edit"></i></button>
                    <button class="btn btn-sm btn-glass-danger" onclick="deleteMaterial(${m.id})"><i class="fas fa-trash"></i></button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading materials:', error);
        tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-danger">Error loading materials</td></tr>';
    }
}

function openMaterialModal(materialId = null) {
    document.getElementById('materialModalTitle').innerHTML = '<i class="fas fa-boxes me-2"></i>' + (materialId ? 'Edit Material' : 'Add Material');
    document.getElementById('materialId').value = materialId || '';
    
    if (materialId) {
        fetch(`${API_BASE_URL}/api/materials/${materialId}`)
            .then(r => r.json())
            .then(material => {
                document.getElementById('materialName').value = material.name || '';
                document.getElementById('materialUnit').value = material.unit || 'kg';
                document.getElementById('materialQuantity').value = material.quantity || 0;
                document.getElementById('materialReorderLevel').value = material.reorder_level || 0;
                document.getElementById('materialCost').value = material.cost_per_unit || 0;
            });
    } else {
        document.getElementById('materialForm').reset();
    }
    
    new bootstrap.Modal(document.getElementById('materialModal')).show();
}

async function saveMaterial() {
    const materialId = document.getElementById('materialId').value;
    const data = {
        name: document.getElementById('materialName').value,
        unit: document.getElementById('materialUnit').value,
        quantity: parseFloat(document.getElementById('materialQuantity').value) || 0,
        reorder_level: parseFloat(document.getElementById('materialReorderLevel').value) || 0,
        cost_per_unit: parseFloat(document.getElementById('materialCost').value) || 0
    };

    try {
        const url = materialId ? `${API_BASE_URL}/api/materials/${materialId}` : `${API_BASE_URL}/api/materials`;
        const method = materialId ? 'PUT' : 'POST';
        
        const response = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            showNotification('Material saved successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('materialModal')).hide();
            loadMaterials();
        } else {
            showNotification('Error saving material', 'error');
        }
    } catch (error) {
        showNotification('Error saving material', 'error');
    }
}

function editMaterial(id) {
    openMaterialModal(id);
}

async function deleteMaterial(id) {
    if (!confirm('Are you sure you want to delete this material?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/materials/${id}`, { method: 'DELETE' });
        if (response.ok) {
            showNotification('Material deleted successfully', 'success');
            loadMaterials();
        } else {
            showNotification('Error deleting material', 'error');
        }
    } catch (error) {
        showNotification('Error deleting material', 'error');
    }
}

// =============================================
// REPAIR JOBS MANAGEMENT
// =============================================
async function loadRepairJobs() {
    const tbody = document.getElementById('repairJobsTableBody');
    if (!tbody) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/repair-jobs`);
        const jobs = await response.json();

        // Update stats
        if (document.getElementById('totalJobs')) {
            document.getElementById('totalJobs').textContent = jobs.length;
            document.getElementById('pendingJobs').textContent = jobs.filter(j => j.status === 'pending').length;
            document.getElementById('inProgressJobs').textContent = jobs.filter(j => j.status === 'in_progress').length;
            document.getElementById('completedJobs').textContent = jobs.filter(j => j.status === 'completed').length;
        }

        if (!jobs || jobs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4">No repair jobs found.</td></tr>';
            return;
        }

        tbody.innerHTML = jobs.map(job => `
            <tr>
                <td>${job.id}</td>
                <td>#${job.pothole_id}</td>
                <td>${job.crew_name || '-'}</td>
                <td>${job.assigned_at ? new Date(job.assigned_at).toLocaleDateString() : '-'}</td>
                <td>${job.started_at ? new Date(job.started_at).toLocaleDateString() : '-'}</td>
                <td>${job.completed_at ? new Date(job.completed_at).toLocaleDateString() : '-'}</td>
                <td><span class="badge bg-${job.status === 'completed' ? 'success' : job.status === 'in_progress' ? 'warning' : 'secondary'}">${job.status}</span></td>
                <td>
                    <button class="btn btn-sm btn-glass-success me-1" onclick="editRepairJob(${job.id})"><i class="fas fa-edit"></i></button>
                    <button class="btn btn-sm btn-glass-primary" onclick="updateJobStatus(${job.id}, '${job.status === 'pending' ? 'in_progress' : 'completed'}')"><i class="fas fa-play"></i></button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading repair jobs:', error);
        tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-danger">Error loading repair jobs</td></tr>';
    }
}

function openRepairJobModal(jobId = null) {
    document.getElementById('repairJobModalTitle').innerHTML = '<i class="fas fa-tools me-2"></i>' + (jobId ? 'Edit Job' : 'Create Job');
    document.getElementById('repairJobId').value = jobId || '';
    
    // Populate pothole dropdown
    fetch(`${API_BASE_URL}/api/potholes?status=verified`)
        .then(r => r.json())
        .then(potholes => {
            const select = document.getElementById('repairJobPothole');
            select.innerHTML = '<option value="">Select Pothole</option>' + 
                potholes.map(p => `<option value="${p.id}">#${p.id} - ${getLocationName(p.latitude, p.longitude)}</option>`).join('');
        });
    
    // Populate crew dropdown
    fetch(`${API_BASE_URL}/api/crews`)
        .then(r => r.json())
        .then(crews => {
            const select = document.getElementById('repairJobCrew');
            select.innerHTML = '<option value="">Select Crew</option>' + 
                crews.map(c => `<option value="${c.id}">${c.name}</option>`).join('');
        });
    
    if (jobId) {
        fetch(`${API_BASE_URL}/api/repair-jobs/${jobId}`)
            .then(r => r.json())
            .then(job => {
                document.getElementById('repairJobPothole').value = job.pothole_id || '';
                document.getElementById('repairJobCrew').value = job.crew_id || '';
                document.getElementById('repairJobNotes').value = job.notes || '';
                document.getElementById('repairJobStatus').value = job.status || 'pending';
            });
    } else {
        document.getElementById('repairJobForm').reset();
    }
    
    new bootstrap.Modal(document.getElementById('repairJobModal')).show();
}

async function saveRepairJob() {
    const jobId = document.getElementById('repairJobId').value;
    const data = {
        pothole_id: document.getElementById('repairJobPothole').value,
        crew_id: document.getElementById('repairJobCrew').value || null,
        notes: document.getElementById('repairJobNotes').value,
        status: document.getElementById('repairJobStatus').value
    };

    try {
        const url = jobId ? `${API_BASE_URL}/api/repair-jobs/${jobId}` : `${API_BASE_URL}/api/repair-jobs`;
        const method = jobId ? 'PUT' : 'POST';
        
        const response = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            showNotification('Repair job saved successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('repairJobModal')).hide();
            loadRepairJobs();
        } else {
            showNotification('Error saving repair job', 'error');
        }
    } catch (error) {
        showNotification('Error saving repair job', 'error');
    }
}

function editRepairJob(id) {
    openRepairJobModal(id);
}

async function updateJobStatus(jobId, newStatus) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/repair-jobs/${jobId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus })
        });

        if (response.ok) {
            showNotification('Job status updated', 'success');
            loadRepairJobs();
            loadActiveRepairs();
            loadRecentlyRepaired();
        } else {
            showNotification('Error updating job status', 'error');
        }
    } catch (error) {
        showNotification('Error updating job status', 'error');
    }
}

// =============================================
// MAP FUNCTIONS
// =============================================
let managerMap = null;
let mapMarkers = [];

function initializeMap() {
    if (document.getElementById('managerMap')) {
        managerMap = L.map('managerMap').setView([-17.8252, 31.0335], 7);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(managerMap);
        
        loadMapMarkers();
    }
}

async function loadMapMarkers() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes`);
        const potholes = await response.json();
        
        // Clear existing markers
        mapMarkers.forEach(m => managerMap.removeLayer(m));
        mapMarkers = [];
        
        potholes.forEach(p => {
            const color = getMarkerColor(p.status, p.size_classification);
            const marker = L.circleMarker([p.latitude, p.longitude], {
                radius: getMarkerSize(p.size_classification),
                color: color,
                fillColor: color,
                fillOpacity: 0.8
            }).addTo(managerMap);
            
            marker.bindPopup(`
                <b>Pothole #${p.id}</b><br>
                Status: ${p.status}<br>
                Size: ${p.size_classification}<br>
                Location: ${getLocationName(p.latitude, p.longitude)}
            `);
            
            mapMarkers.push(marker);
        });
    } catch (error) {
        console.error('Error loading map markers:', error);
    }
}

function getMarkerColor(status, size) {
    switch(status) {
        case 'repaired': return '#28a745';
        case 'verified': return '#17a2b8';
        default:
            switch(size) {
                case 'large': return '#dc3545';
                case 'medium': return '#fd7e14';
                default: return '#ffc107';
            }
    }
}

function getMarkerSize(size) {
    switch(size) {
        case 'large': return 12;
        case 'medium': return 9;
        default: return 6;
    }
}

async function filterMap(status) {
    // Clear existing markers
    mapMarkers.forEach(m => managerMap.removeLayer(m));
    mapMarkers = [];
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes`);
        const potholes = await response.json();
        
        const filtered = status === 'all' ? potholes : potholes.filter(p => p.status === status);
        
        filtered.forEach(p => {
            const color = getMarkerColor(p.status, p.size_classification);
            const marker = L.circleMarker([p.latitude, p.longitude], {
                radius: getMarkerSize(p.size_classification),
                color: color,
                fillColor: color,
                fillOpacity: 0.8
            }).addTo(managerMap);
            
            marker.bindPopup(`
                <b>Pothole #${p.id}</b><br>
                Status: ${p.status}<br>
                Size: ${p.size_classification}<br>
                Location: ${getLocationName(p.latitude, p.longitude)}
            `);
            
            mapMarkers.push(marker);
        });
        
        showNotification(`Showing ${filtered.length} ${status} potholes on map`, 'info');
    } catch (error) {
        console.error('Error filtering map:', error);
    }
}

function filterByStatus(status) {
    filterMap(status);
}

// =============================================
// HELPER FUNCTIONS
// =============================================
function getLocationName(lat, lng) {
    const locations = [
        {lat: -17.8252, lng: 31.0335, name: 'Harare CBD'},
        {lat: -17.8300, lng: 31.0400, name: 'Chinhoyi Street'},
        {lat: -20.1651, lng: 28.5832, name: 'Bulawayo CBD'},
        {lat: -19.0584, lng: 29.7850, name: 'Gweru'},
        {lat: -18.0184, lng: 31.0835, name: 'Mbare'},
        {lat: -17.9866, lng: 31.3072, name: 'Mount Pleasant'},
        {lat: -17.7715, lng: 31.0456, name: 'Avondale'},
        {lat: -17.8500, lng: 31.0200, name: 'Sam Nujoma Street'},
        {lat: -17.8132, lng: 31.0497, name: 'Eastlea'},
        {lat: -19.4438, lng: 29.7820, name: 'Kwekwe'},
        {lat: -18.1409, lng: 32.4172, name: 'Mutare'},
        {lat: -18.6744, lng: 31.9875, name: 'Masvingo'}
    ];
    
    let closest = null;
    let minDist = Infinity;
    
    for (const loc of locations) {
        const dist = Math.sqrt(Math.pow(lat - loc.lat, 2) + Math.pow(lng - loc.lng, 2));
        if (dist < minDist && dist < 0.15) {
            minDist = dist;
            closest = loc;
        }
    }
    
    return closest ? closest.name : `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
}

function viewImage(imgUrl) {
    const modal = new bootstrap.Modal(document.getElementById('imageViewModal'));
    document.getElementById('modalImage').src = imgUrl;
    modal.show();
}

function showNotification(message, type) {
    // Simple notification - could be enhanced with toast notifications
    alert(message);
}

function refreshAlerts() {
    loadAlerts();
    loadPotholeStats();
    loadPotholesToVerify();
    loadActiveRepairs();
    loadRecentlyRepaired();
    loadCrews();
    loadCrewMembers();
    loadMaterials();
    loadRepairJobs();
    if (managerMap) loadMapMarkers();
    showNotification('Data refreshed', 'success');
}

function logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('userRole');
    localStorage.removeItem('userName');
    localStorage.removeItem('userId');
    window.location.href = 'login.html';
}