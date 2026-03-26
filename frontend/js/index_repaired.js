// =============================================
// REPAIRED ROADS SECTION (index.html)
// =============================================

// Function to load and display repaired roads on index.html
async function loadRepairedRoads() {
    const container = document.getElementById('repairedRoadsContainer');
    if (!container) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes/repaired`);
        const repaired = await response.json();
        
        if (repaired.length === 0) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i class="fas fa-check-circle text-success mb-3" style="font-size: 3rem; opacity: 0.5;"></i>
                    <p class="text-muted">No repaired roads to display yet.</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = repaired.map(r => {
            const photoUrl = r.after_photos && r.after_photos.length > 0 
                ? `${API_BASE_URL}/uploads/repairs/${r.after_photos[0]}` 
                : 'https://via.placeholder.com/400x300?text=No+Photo';
            
            const location = r.location_name || getLocationNameFromCoords(r.latitude, r.longitude);
            
            return `
                <div class="col-md-6 col-lg-4 mb-4">
                    <div class="repaired-road-card">
                        <div class="position-relative">
                            <img src="${photoUrl}" class="repaired-road-image" 
                                 onerror="this.src='https://via.placeholder.com/400x300?text=No+Photo'"
                                 onclick="viewRepairedImage('${photoUrl}')">
                            <div class="position-absolute" style="bottom: 10px; left: 10px;">
                                <span class="location-label">
                                    <i class="fas fa-map-marker-alt"></i> ${location}
                                </span>
                            </div>
                            <div class="position-absolute" style="top: 10px; right: 10px;">
                                <span class="badge bg-success">
                                    <i class="fas fa-check"></i> Repaired
                                </span>
                            </div>
                        </div>
                        <div class="mt-3">
                            <small class="text-muted">
                                <i class="fas fa-calendar me-1"></i>
                                ${r.completed_at ? new Date(r.completed_at).toLocaleDateString() : 'Recently'}
                                ${r.repaired_by ? '<br><i class="fas fa-user me-1"></i> By: ' + r.repaired_by : ''}
                            </small>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        console.error('Error loading repaired roads:', error);
        if (container) {
            container.innerHTML = '<p class="text-center text-danger py-4">Error loading repaired roads</p>';
        }
    }
}

// Helper function to get location name from coordinates
function getLocationNameFromCoords(lat, lng) {
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

function viewRepairedImage(imgUrl) {
    const modal = new bootstrap.Modal(document.getElementById('imageViewModal'));
    document.getElementById('modalImage').src = imgUrl;
    modal.show();
}

// =============================================
// HEATMAP FILTER FUNCTIONS (main.js enhancement)
// =============================================

let currentHeatmapFilter = 'all';
let heatmapLayer = null;

async function filterHeatmap(status, btn) {
    currentHeatmapFilter = status;
    
    // Update button states
    document.querySelectorAll('.heatmap-filter-btn').forEach(b => {
        b.classList.remove('active');
    });
    if (btn) {
        btn.classList.add('active');
    }
    
    await updateHeatmapWithFilter();
}

async function updateHeatmapWithFilter() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes`);
        const potholes = await response.json();
        
        // Filter based on status
        let filteredPotholes = potholes;
        if (currentHeatmapFilter !== 'all') {
            filteredPotholes = potholes.filter(p => p.status === currentHeatmapFilter);
        }
        
        // Remove existing heat layer
        if (heatmapLayer) {
            heatMap.removeLayer(heatmapLayer);
        }
        
        // Recreate heatmap data
        const heatData = filteredPotholes.map(p => [p.latitude, p.longitude, 1]);
        
        setTimeout(() => {
            heatMap.invalidateSize();
        }, 100);
        
        heatmapLayer = L.heatLayer(heatData, {
            radius: 30,
            blur: 20,
            maxZoom: 17,
            gradient: {0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        }).addTo(heatMap);
        
    } catch (error) {
        console.error('Error filtering heatmap:', error);
    }
}

// Add this to main.js - Enhanced map marker filtering
let currentMainMapFilter = 'all';
let mainMapMarkers = [];

async function filterMainMap(status) {
    currentMainMapFilter = status;
    
    // Update button states
    document.querySelectorAll('#mainMap + div .btn, .map-filters .btn').forEach(btn => {
        btn.classList.remove('active');
    });
    if (event && event.target) {
        event.target.classList.add('active');
    }
    
    await loadMapWithFilter();
}

async function loadMapWithFilter() {
    try {
        // Clear existing markers
        mainMapMarkers.forEach(marker => mainMap.removeLayer(marker));
        mainMapMarkers = [];
        
        const response = await fetch(`${API_BASE_URL}/api/potholes`);
        const potholes = await response.json();
        
        potholes.forEach(pothole => {
            if (currentMainMapFilter === 'all' || pothole.status === currentMainMapFilter) {
                const color = getMarkerColorForStatus(pothole.status, pothole.size_classification);
                
                const marker = L.circleMarker([pothole.latitude, pothole.longitude], {
                    radius: getMarkerSize(pothole.size_classification),
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.8
                }).addTo(mainMap);
                
                marker.bindPopup(`
                    <b>Pothole Report #${pothole.id}</b><br>
                    Size: ${pothole.size_classification || 'Unknown'}<br>
                    Status: ${pothole.status}<br>
                    Diameter: ${pothole.diameter?.toFixed(2) || '?'}m<br>
                    Reported: ${new Date(pothole.reported_at).toLocaleDateString()}
                `);
                
                mainMapMarkers.push(marker);
            }
        });
        
    } catch (error) {
        console.error('Error loading map with filter:', error);
    }
}

function getMarkerColorForStatus(status, size) {
    switch(status) {
        case 'repaired':
            return '#28a745';  // Green
        case 'verified':
            return '#17a2b8';  // Teal
        default:
            // For pending - use size-based colors
            switch(size) {
                case 'large':
                    return '#dc3545';  // Red
                case 'medium':
                    return '#fd7e14';  // Orange
                default:
                    return '#ffc107';  // Yellow
            }
    }
}

// Initialize filters on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check if we have the repaired roads container
    if (document.getElementById('repairedRoadsContainer')) {
        loadRepairedRoads();
    }
    
    // Set up heatmap filter buttons if they exist
    const heatmapFilterBtns = document.querySelectorAll('.heatmap-filter-btn');
    if (heatmapFilterBtns.length > 0) {
        heatmapFilterBtns[0].addEventListener('click', () => filterHeatmap('all'));
        heatmapFilterBtns[1].addEventListener('click', () => filterHeatmap('verified'));
        heatmapFilterBtns[2].addEventListener('click', () => filterHeatmap('repaired'));
        heatmapFilterBtns[3].addEventListener('click', () => filterHeatmap('pending'));
    }
    
    // Set up main map filter buttons
    const mapFilterBtns = document.querySelectorAll('.map-filter-btn');
    if (mapFilterBtns.length > 0) {
        mapFilterBtns[0].addEventListener('click', () => filterMainMap('all'));
        mapFilterBtns[1].addEventListener('click', () => filterMainMap('verified'));
        mapFilterBtns[2].addEventListener('click', () => filterMainMap('repaired'));
        mapFilterBtns[3].addEventListener('click', () => filterMainMap('pending'));
    }
});
