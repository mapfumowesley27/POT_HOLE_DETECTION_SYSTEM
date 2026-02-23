const API_BASE_URL = window.APP_CONFIG ? window.APP_CONFIG.API_BASE_URL : 'http://localhost:5000';

let allPotholes = [];
let activeSize = 'all';
let activeStatus = 'all';
let showAnnotated = true;
let currentModalPotholeId = null;
const locationCache = {};

document.addEventListener('DOMContentLoaded', function () {
    loadGalleryData();
    setupFilters();
});

async function loadGalleryData() {
    const loadingState = document.getElementById('loadingState');
    const emptyState = document.getElementById('emptyState');
    const grid = document.getElementById('galleryGrid');

    loadingState.style.display = 'block';
    emptyState.style.display = 'none';
    grid.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes`);
        const potholes = await response.json();

        allPotholes = potholes.filter(p => p.image_path);

        updateStats(potholes);
        renderGallery();
        resolveLocations();
    } catch (error) {
        console.error('Error loading gallery data:', error);
        grid.innerHTML = `
            <div class="col-12 text-center py-5">
                <p class="text-danger">Failed to load data. Is the backend running?</p>
                <button class="btn btn-primary mt-2" onclick="loadGalleryData()">Retry</button>
            </div>`;
    } finally {
        loadingState.style.display = 'none';
    }
}

function updateStats(potholes) {
    const withImages = potholes.filter(p => p.image_path);
    document.getElementById('totalImages').textContent = withImages.length;
    document.getElementById('smallCount').textContent = withImages.filter(p => p.size_classification === 'small').length;
    document.getElementById('mediumCount').textContent = withImages.filter(p => p.size_classification === 'medium').length;
    document.getElementById('largeCount').textContent = withImages.filter(p => p.size_classification === 'large').length;
}

async function reverseGeocode(lat, lon) {
    const key = `${lat.toFixed(4)},${lon.toFixed(4)}`;
    if (locationCache[key]) return locationCache[key];

    try {
        const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&zoom=16&addressdetails=1`,
            { headers: { 'Accept-Language': 'en' } }
        );
        const data = await response.json();

        let placeName = 'Unknown location';
        if (data.address) {
            const a = data.address;
            const parts = [
                a.road || a.pedestrian || a.footway || '',
                a.suburb || a.neighbourhood || a.quarter || '',
                a.city || a.town || a.village || a.county || ''
            ].filter(Boolean);
            placeName = parts.join(', ') || data.display_name || placeName;
        } else if (data.display_name) {
            placeName = data.display_name.split(',').slice(0, 3).join(',');
        }

        locationCache[key] = placeName;
        return placeName;
    } catch (error) {
        console.error('Reverse geocoding error:', error);
        return 'Unknown location';
    }
}

async function resolveLocations() {
    const filtered = getFilteredPotholes();

    for (let i = 0; i < filtered.length; i++) {
        const pothole = filtered[i];
        const placeName = await reverseGeocode(pothole.latitude, pothole.longitude);

        const locationEl = document.getElementById(`location-${pothole.id}`);
        if (locationEl) {
            locationEl.textContent = placeName;
            locationEl.title = `${placeName} (${pothole.latitude.toFixed(4)}, ${pothole.longitude.toFixed(4)})`;
        }

        // Rate limit: ~1 request per second for Nominatim (skip delay for cached results)
        const cacheKey = `${pothole.latitude.toFixed(4)},${pothole.longitude.toFixed(4)}`;
        const wasCached = locationCache[cacheKey] === placeName;
        if (!wasCached && i < filtered.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 1100));
        }
    }
}

function getFilteredPotholes() {
    let filtered = allPotholes;
    if (activeSize !== 'all') {
        filtered = filtered.filter(p => p.size_classification === activeSize);
    }
    if (activeStatus !== 'all') {
        filtered = filtered.filter(p => p.status === activeStatus);
    }
    return filtered;
}

function renderGallery() {
    const grid = document.getElementById('galleryGrid');
    const emptyState = document.getElementById('emptyState');

    const filtered = getFilteredPotholes();

    if (filtered.length === 0) {
        grid.innerHTML = '';
        emptyState.style.display = 'block';
        return;
    }

    emptyState.style.display = 'none';

    grid.innerHTML = filtered.map(pothole => {
        const imageUrl = showAnnotated
            ? `${API_BASE_URL}/api/potholes/${pothole.id}/annotated-image`
            : `${API_BASE_URL}/api/potholes/${pothole.id}/original-image`;

        const sizeClass = pothole.size_classification || 'unknown';
        const diameter = pothole.diameter ? pothole.diameter.toFixed(2) + 'm' : '—';
        const confidence = pothole.confidence_score ? (pothole.confidence_score * 100).toFixed(0) + '%' : '—';
        const date = pothole.reported_at ? new Date(pothole.reported_at).toLocaleDateString() : '—';
        const status = pothole.status || 'pending';
        const coords = `${pothole.latitude.toFixed(4)}, ${pothole.longitude.toFixed(4)}`;
        const cachedKey = `${pothole.latitude.toFixed(4)},${pothole.longitude.toFixed(4)}`;
        const cachedLocation = locationCache[cachedKey] || 'Loading...';

        return `
            <div class="col-lg-3 col-md-4 col-sm-6" id="card-wrapper-${pothole.id}">
                <div class="gallery-card">
                    <div class="gallery-card-image" onclick="openModal(${pothole.id})">
                        <span class="id-badge">#${pothole.id}</span>
                        <span class="size-badge ${sizeClass}">${sizeClass}</span>
                        <img src="${imageUrl}" alt="Pothole #${pothole.id}"
                             onerror="this.parentElement.innerHTML='<div class=\'image-placeholder\'>🚫</div>'">
                    </div>
                    <div class="gallery-card-body">
                        <div class="info-row location-row">
                            <span class="info-label">📍 Location</span>
                            <span class="info-value location-name" id="location-${pothole.id}" title="${coords}">${cachedLocation}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Coordinates</span>
                            <span class="info-value coords-value">${coords}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Diameter</span>
                            <span class="info-value">${diameter}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Confidence</span>
                            <span class="info-value">${confidence}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Reported</span>
                            <span class="info-value">${date}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Status</span>
                            <span class="status-badge ${status}">${status}</span>
                        </div>
                        <div class="card-actions">
                            <button class="btn btn-sm btn-outline-danger delete-btn" onclick="deletePothole(${pothole.id}, event)" title="Delete this report">
                                🗑️ Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>`;
    }).join('');
}

async function openModal(potholeId) {
    const pothole = allPotholes.find(p => p.id === potholeId);
    if (!pothole) return;

    currentModalPotholeId = potholeId;

    const imageUrl = showAnnotated
        ? `${API_BASE_URL}/api/potholes/${pothole.id}/annotated-image`
        : `${API_BASE_URL}/api/potholes/${pothole.id}/original-image`;

    document.getElementById('modalImage').src = imageUrl;
    document.getElementById('modalTitle').textContent = `Pothole #${pothole.id} — Detection Result`;

    const sizeClass = pothole.size_classification || 'unknown';
    const sizeBadge = document.getElementById('modalSize');
    sizeBadge.textContent = sizeClass.toUpperCase();
    sizeBadge.className = `badge size-badge ${sizeClass}`;

    document.getElementById('modalDiameter').textContent =
        pothole.diameter ? pothole.diameter.toFixed(2) + 'm' : '—';
    document.getElementById('modalConfidence').textContent =
        pothole.confidence_score ? (pothole.confidence_score * 100).toFixed(1) + '%' : '—';
    document.getElementById('modalCoords').textContent =
        `${pothole.latitude.toFixed(4)}, ${pothole.longitude.toFixed(4)}`;

    // Show place name
    const locationEl = document.getElementById('modalLocation');
    locationEl.textContent = 'Resolving location...';
    const placeName = await reverseGeocode(pothole.latitude, pothole.longitude);
    locationEl.textContent = `${placeName} (${pothole.latitude.toFixed(4)}, ${pothole.longitude.toFixed(4)})`;

    const modal = new bootstrap.Modal(document.getElementById('imageModal'));
    modal.show();
}

async function deletePothole(potholeId, event) {
    if (event) event.stopPropagation();

    if (!confirm(`Are you sure you want to delete Pothole #${potholeId}? This will permanently remove the image and record.`)) {
        return false;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/potholes/${potholeId}`, {
            method: 'DELETE'
        });
        const data = await response.json();

        if (data.success) {
            // Remove from local array
            allPotholes = allPotholes.filter(p => p.id !== potholeId);

            // Animate card removal
            const cardWrapper = document.getElementById(`card-wrapper-${potholeId}`);
            if (cardWrapper) {
                cardWrapper.style.transition = 'opacity 0.3s, transform 0.3s';
                cardWrapper.style.opacity = '0';
                cardWrapper.style.transform = 'scale(0.8)';
                setTimeout(() => {
                    renderGallery();
                    updateStats(allPotholes);
                }, 300);
            } else {
                renderGallery();
                updateStats(allPotholes);
            }
            return true;
        } else {
            alert('Error deleting pothole: ' + (data.error || 'Unknown error'));
            return false;
        }
    } catch (error) {
        console.error('Error deleting pothole:', error);
        alert('Failed to delete pothole. Is the backend running?');
        return false;
    }
}

async function deletePotholeFromModal() {
    if (!currentModalPotholeId) return;

    const deleted = await deletePothole(currentModalPotholeId, null);

    if (deleted) {
        const modalEl = document.getElementById('imageModal');
        const modal = bootstrap.Modal.getInstance(modalEl);
        if (modal) modal.hide();
        currentModalPotholeId = null;
    }
}

function setupFilters() {
    document.querySelectorAll('#sizeFilter .btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('#sizeFilter .btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            activeSize = this.dataset.size;
            renderGallery();
            resolveLocations();
        });
    });

    document.querySelectorAll('#statusFilter .btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('#statusFilter .btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            activeStatus = this.dataset.status;
            renderGallery();
            resolveLocations();
        });
    });

    document.getElementById('viewAnnotated').addEventListener('click', function () {
        showAnnotated = true;
        this.classList.add('active');
        document.getElementById('viewOriginal').classList.remove('active');
        renderGallery();
    });

    document.getElementById('viewOriginal').addEventListener('click', function () {
        showAnnotated = false;
        this.classList.add('active');
        document.getElementById('viewAnnotated').classList.remove('active');
        renderGallery();
    });
}
