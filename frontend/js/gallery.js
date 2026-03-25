const API_BASE_URL = window.APP_CONFIG ? window.APP_CONFIG.API_BASE_URL : 'http://localhost:5000';

let allPotholes = [];
let activeSize = 'all';
let activeStatus = 'all';
let showAnnotated = true;
let currentModalPotholeId = null;
const locationCache = {};

// Folder & Pagination State
let currentFolder = null; // null means viewing folders
let itemsPerPage = 12;
let currentPage = 1;

document.addEventListener('DOMContentLoaded', function () {
    const isLoggedIn = localStorage.getItem('authToken');
    const userRole = localStorage.getItem('userRole');

    if (!isLoggedIn || userRole !== 'manager') {
        alert('Access denied. Maintenance Manager privileges required.');
        window.location.href = 'login.html';
        return;
    }

    loadGalleryData();
    setupFilters();
    setupBreadcrumbs();
    setupViewModes();
});

function setupViewModes() {
    const viewFoldersBtn = document.getElementById('viewFoldersMode');
    const viewAllBtn = document.getElementById('viewAllMode');

    if (viewFoldersBtn && viewAllBtn) {
        viewFoldersBtn.addEventListener('click', () => {
            currentFolder = null;
            viewFoldersBtn.classList.add('active');
            viewAllBtn.classList.remove('active');
            renderGallery();
            resolveLocations();
        });

        viewAllBtn.addEventListener('click', () => {
            currentFolder = 'all'; // Special value for "Show All"
            viewFoldersBtn.classList.remove('active');
            viewAllBtn.classList.add('active');
            renderGallery();
            resolveLocations();
        });
    }
}

async function loadGalleryData() {
    const loadingState = document.getElementById('loadingState');
    const emptyState = document.getElementById('emptyState');
    const grid = document.getElementById('galleryGrid');
    const folderGrid = document.getElementById('foldersGrid');

    loadingState.style.display = 'block';
    emptyState.style.display = 'none';
    grid.innerHTML = '';
    folderGrid.innerHTML = '';

    try {
        console.log('Fetching potholes from:', `${API_BASE_URL}/api/potholes`);
        const response = await fetch(`${API_BASE_URL}/api/potholes`);
        const potholes = await response.json();
        console.log(`Loaded ${potholes.length} potholes from backend`);

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
    console.log('Rendering gallery... Current view mode:', currentFolder === null ? 'Folders' : (currentFolder === 'all' ? 'All Images' : `Folder: ${currentFolder}`));
    const grid = document.getElementById('galleryGrid');
    const folderGrid = document.getElementById('foldersGrid');
    const emptyState = document.getElementById('emptyState');
    const breadcrumb = document.getElementById('galleryBreadcrumb');
    const pagination = document.getElementById('paginationContainer');

    const filtered = getFilteredPotholes();

    if (filtered.length === 0) {
        grid.innerHTML = '';
        folderGrid.innerHTML = '';
        emptyState.style.display = 'block';
        breadcrumb.style.display = 'none';
        pagination.style.setProperty('display', 'none', 'important');
        return;
    }

    emptyState.style.display = 'none';

    if (currentFolder === null) {
        // Render Folders view
        renderFolders(filtered);
        grid.style.display = 'none';
        folderGrid.style.display = 'flex';
        breadcrumb.style.display = 'none';
        pagination.style.setProperty('display', 'none', 'important');
    } else {
        // Render Images (either specific folder or all)
        const folderItems = currentFolder === 'all' 
            ? filtered 
            : filtered.filter(p => getFolderName(p) === currentFolder);
        
        // Pagination logic
        const totalPages = Math.ceil(folderItems.length / itemsPerPage);
        if (currentPage > totalPages) currentPage = Math.max(1, totalPages);
        
        const start = (currentPage - 1) * itemsPerPage;
        const pagedItems = folderItems.slice(start, start + itemsPerPage);

        renderImages(pagedItems);
        
        grid.style.display = 'flex';
        folderGrid.style.display = 'none';
        
        if (currentFolder === 'all') {
            breadcrumb.style.display = 'none';
        } else {
            breadcrumb.style.display = 'block';
            document.getElementById('currentFolderName').textContent = currentFolder;
        }
        
        if (totalPages > 1) {
            renderPagination(totalPages);
            pagination.style.setProperty('display', 'flex', 'important');
        } else {
            pagination.style.setProperty('display', 'none', 'important');
        }
    }
}

function getFolderName(pothole) {
    if (!pothole.reported_at) return 'Unknown Date';
    const date = new Date(pothole.reported_at);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

function renderFolders(potholes) {
    const folderGrid = document.getElementById('foldersGrid');
    const groups = {};
    
    potholes.forEach(p => {
        const name = getFolderName(p);
        if (!groups[name]) groups[name] = [];
        groups[name].push(p);
    });

    const folderNames = Object.keys(groups).sort((a, b) => new Date(b) - new Date(a));

    folderGrid.innerHTML = folderNames.map(name => {
        const count = groups[name].length;
        return `
            <div class="col-lg-3 col-md-4 col-sm-6">
                <div class="folder-card" onclick="openFolder('${name}')">
                    <span class="folder-icon">📂</span>
                    <div class="folder-name">${name}</div>
                    <div class="folder-count">${count} ${count === 1 ? 'image' : 'images'}</div>
                </div>
            </div>`;
    }).join('');
}

function openFolder(name) {
    currentFolder = name;
    currentPage = 1;
    
    // Update view mode buttons
    const viewFoldersBtn = document.getElementById('viewFoldersMode');
    const viewAllBtn = document.getElementById('viewAllMode');
    if (viewFoldersBtn) viewFoldersBtn.classList.add('active');
    if (viewAllBtn) viewAllBtn.classList.remove('active');

    renderGallery();
    resolveLocations();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setupBreadcrumbs() {
    document.getElementById('backToFolders').addEventListener('click', function(e) {
        e.preventDefault();
        currentFolder = null;
        renderGallery();
    });
}

function renderPagination(totalPages) {
    const list = document.getElementById('paginationList');
    let html = '';
    
    // Previous
    html += `
        <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage - 1}, event)">Previous</a>
        </li>`;
    
    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || (i >= currentPage - 2 && i <= currentPage + 2)) {
            html += `
                <li class="page-item ${currentPage === i ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="changePage(${i}, event)">${i}</a>
                </li>`;
        } else if (i === currentPage - 3 || i === currentPage + 3) {
            html += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
        }
    }
    
    // Next
    html += `
        <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage + 1}, event)">Next</a>
        </li>`;
        
    list.innerHTML = html;
}

function changePage(page, event) {
    if (event) event.preventDefault();
    currentPage = page;
    renderGallery();
    resolveLocations();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderImages(potholes) {
    const grid = document.getElementById('galleryGrid');
    grid.innerHTML = potholes.map(pothole => {
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
                             onload="console.log('Successfully loaded image for pothole #${pothole.id}')"
                             onerror="console.error('Failed to load image for pothole #${pothole.id} from: ' + this.src); this.parentElement.innerHTML='<div class=\'image-placeholder\'>🚫<br><small style=\'font-size:10px\'>Load Error</small></div>'">
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
        resolveLocations();
    });

    document.getElementById('viewOriginal').addEventListener('click', function () {
        showAnnotated = false;
        this.classList.add('active');
        document.getElementById('viewAnnotated').classList.remove('active');
        renderGallery();
        resolveLocations();
    });
}
