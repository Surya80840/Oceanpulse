/* OceanPulse front-end.  Pure vanilla JS + Leaflet, no build step. */

const API = {
    health: '/api/health',
    bootstrap: '/api/bootstrap',
    grid: '/api/grid',
    species: '/api/species',
    queryBbox: '/api/query/bbox',
    querySpecies: '/api/query/species',
};

const state = {
    map: null,
    heatLayer: null,
    gridLayer: null,
    topMarkersLayer: null,
    bboxRect: null,
    drawControl: null,
    gridInfo: null,
    speciesList: [],
    currentResult: null,
};

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initUI();
    loadGrid();
    loadSpecies();
    pollHealth();
});

/* -------------------------------------------------------------------- */
/* Map                                                                   */
/* -------------------------------------------------------------------- */
function initMap() {
    const map = L.map('map', {
        zoomControl: true,
        center: [39.0, -120.0],
        zoom: 5,
        preferCanvas: true,
    });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 19,
    }).addTo(map);
    state.map = map;

    state.heatLayer = L.layerGroup().addTo(map);
    state.gridLayer = L.layerGroup();          // shown only when no heatmap
    state.topMarkersLayer = L.layerGroup().addTo(map);

    // Drawing controls - rectangle only
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    state.drawControl = new L.Control.Draw({
        draw: {
            rectangle: { shapeOptions: { color: '#22d3ee', weight: 2, fillOpacity: 0.08 } },
            polygon: false, polyline: false, circle: false, marker: false, circlemarker: false,
        },
        edit: { featureGroup: drawnItems, remove: true },
    });
    map.addControl(state.drawControl);

    map.on(L.Draw.Event.CREATED, (e) => {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        const b = e.layer.getBounds();
        setBBoxInputs(b.getSouth(), b.getNorth(), b.getWest(), b.getEast());
        setTab('bbox');
        if (state.bboxRect) { state.bboxRect.remove(); }
        state.bboxRect = e.layer;
    });
}

function setBBoxInputs(latMin, latMax, lonMin, lonMax) {
    document.getElementById('lat-min').value = clamp(latMin, 30, 48).toFixed(2);
    document.getElementById('lat-max').value = clamp(latMax, 30, 48).toFixed(2);
    document.getElementById('lon-min').value = clamp(lonMin, -124, -116).toFixed(2);
    document.getElementById('lon-max').value = clamp(lonMax, -124, -116).toFixed(2);
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

/* -------------------------------------------------------------------- */
/* UI wiring                                                             */
/* -------------------------------------------------------------------- */
function initUI() {
    document.querySelectorAll('.tab').forEach(btn => {
        btn.addEventListener('click', () => setTab(btn.dataset.tab));
    });
    document.getElementById('run-bbox').addEventListener('click', runBboxQuery);
    document.getElementById('run-species').addEventListener('click', runSpeciesQuery);
}

function setTab(name) {
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === name);
    });
    document.querySelectorAll('.tab-panel').forEach(p => {
        p.classList.toggle('hidden', p.dataset.panel !== name);
    });
}

function showOverlay(text) {
    document.getElementById('overlay-text').textContent = text;
    document.getElementById('overlay').classList.remove('hidden');
}
function hideOverlay() {
    document.getElementById('overlay').classList.add('hidden');
}

/* -------------------------------------------------------------------- */
/* Bootstrap / status                                                    */
/* -------------------------------------------------------------------- */
async function loadGrid() {
    const res = await fetch(API.grid);
    state.gridInfo = await res.json();
    drawGridBounds();
}

function drawGridBounds() {
    const g = state.gridInfo;
    const rect = L.rectangle([[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]], {
        color: '#38bdf8', weight: 1.5, fill: false, dashArray: '5,6', opacity: 0.8,
    });
    rect.addTo(state.map);
    state.map.fitBounds(rect.getBounds(), { padding: [40, 40] });
}

async function loadSpecies() {
    const res = await fetch(API.species);
    const data = await res.json();
    state.speciesList = data.species;
    const dl = document.getElementById('species-list');
    dl.innerHTML = state.speciesList.map(
        s => `<option value="${s.id}" label="${escapeHtml(s.display_name)}">${escapeHtml(s.display_name)}</option>`
    ).join('');
}

async function pollHealth() {
    try {
        const res = await fetch(API.health);
        const h = await res.json();
        const dot = document.getElementById('status-dot');
        const txt = document.getElementById('status-text');
        if (h.status === 'ready') {
            dot.className = 'status-dot ready';
            const refresh = h.last_refresh ? ` | last refresh ${h.last_refresh.replace('T',' ').slice(0,16)}` : '';
            txt.textContent = `${h.n_species} species, ${h.feature_store_days} days cached${refresh}`;
            return;
        } else if (h.status === 'error') {
            dot.className = 'status-dot error';
            const bs = await (await fetch(API.bootstrap)).json();
            txt.textContent = `error: ${bs.message || 'unknown'}`;
        } else {
            dot.className = 'status-dot';
            const bs = await (await fetch(API.bootstrap)).json();
            const done = bs.done || 0, total = bs.total || 30;
            txt.textContent = `${h.status}: ${done}/${total} days`;
        }
    } catch (err) {
        document.getElementById('status-text').textContent = 'backend unreachable';
    }
    setTimeout(pollHealth, 3000);
}

/* -------------------------------------------------------------------- */
/* Queries                                                               */
/* -------------------------------------------------------------------- */
async function runBboxQuery() {
    const body = {
        lat_min: parseFloat(document.getElementById('lat-min').value),
        lat_max: parseFloat(document.getElementById('lat-max').value),
        lon_min: parseFloat(document.getElementById('lon-min').value),
        lon_max: parseFloat(document.getElementById('lon-max').value),
        top_k: 10,
    };
    if (body.lat_min >= body.lat_max || body.lon_min >= body.lon_max) {
        alert('Please provide a valid bounding box (min < max).');
        return;
    }
    showOverlay('Scoring every grid cell and all 369 species...');
    try {
        const res = await fetch(API.queryBbox, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        if (!res.ok) { throw new Error(await res.text()); }
        const data = await res.json();
        renderBboxResult(data, body);
    } catch (err) {
        alert('Query failed: ' + err.message);
    } finally {
        hideOverlay();
    }
}

async function runSpeciesQuery() {
    const input = document.getElementById('species-input').value.trim();
    const byId = state.speciesList.find(s => s.id === input);
    const byDisplay = state.speciesList.find(s => s.display_name.toLowerCase() === input.toLowerCase());
    const pick = byId || byDisplay;
    if (!pick) {
        alert('Pick a species from the dropdown.');
        return;
    }
    showOverlay(`Scoring ${pick.display_name} across every cell...`);
    try {
        const res = await fetch(API.querySpecies, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({species: pick.id, top_k: 10}),
        });
        if (!res.ok) { throw new Error(await res.text()); }
        const data = await res.json();
        renderSpeciesResult(data);
    } catch (err) {
        alert('Query failed: ' + err.message);
    } finally {
        hideOverlay();
    }
}

/* -------------------------------------------------------------------- */
/* Result rendering                                                      */
/* -------------------------------------------------------------------- */
function clearMapLayers() {
    state.heatLayer.clearLayers();
    state.topMarkersLayer.clearLayers();
}

function renderBboxResult(data, query) {
    clearMapLayers();

    // Draw the bbox
    if (state.bboxRect) { state.bboxRect.remove(); state.bboxRect = null; }
    state.bboxRect = L.rectangle(
        [[query.lat_min, query.lon_min], [query.lat_max, query.lon_max]],
        { color: '#22d3ee', weight: 2, fill: false }
    ).addTo(state.map);
    state.map.fitBounds(state.bboxRect.getBounds(), { padding: [50, 50] });

    // Heatmap: probability of top-1 species inside bbox
    drawHeatmap(data.heatmap);

    // Side panel
    document.getElementById('results-title').textContent = 'Top species in bbox';
    document.getElementById('results-meta').innerHTML =
        `forecast ${formatDate(data.forecast_start)} &rarr; ${formatDate(data.forecast_end)} &middot; ${data.n_cells} cells` +
        (data.heatmap_species ? ` &middot; heatmap: <b>${escapeHtml(prettyName(data.heatmap_species))}</b>` : '');

    const list = document.getElementById('rank-list');
    list.innerHTML = '';
    data.top_species.forEach((s, i) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="idx">${i + 1}</span>
            <span class="name">${escapeHtml(s.display_name)}<small>${escapeHtml(s.species)} &middot; max prob ${fmtProb(s.max_probability)}</small></span>
            <span class="prob">${fmtProb(s.mean_probability)}</span>
        `;
        li.addEventListener('click', async () => {
            // Clicking a species in bbox mode triggers a global species query
            document.getElementById('species-input').value = s.species;
            setTab('species');
            await runSpeciesQuery();
        });
        list.appendChild(li);
    });
}

function renderSpeciesResult(data) {
    clearMapLayers();
    if (state.bboxRect) { state.bboxRect.remove(); state.bboxRect = null; }

    // Full-grid heatmap
    drawHeatmap(data.heatmap);

    // Top-N hotspots: big markers
    data.top_locations.forEach((loc, i) => {
        const marker = L.circleMarker([loc.lat, loc.lon], {
            radius: 10, color: '#f8fafc', weight: 2,
            fillColor: probColor(loc.probability), fillOpacity: 0.95,
        }).bindTooltip(
            `#${i + 1} &middot; (${loc.lat.toFixed(2)}, ${loc.lon.toFixed(2)})<br>` +
            `probability: <b>${fmtProb(loc.probability)}</b><br>` +
            `n_obs in cell: ${loc.n_obs_in_cell}`,
            { sticky: true }
        );
        marker.addTo(state.topMarkersLayer);
    });
    if (data.top_locations.length) {
        const bounds = L.latLngBounds(data.top_locations.map(l => [l.lat, l.lon]));
        state.map.fitBounds(bounds.pad(0.35));
    }

    document.getElementById('results-title').textContent = data.display_name;
    document.getElementById('results-meta').innerHTML =
        `forecast ${formatDate(data.forecast_start)} &rarr; ${formatDate(data.forecast_end)} &middot; ${data.heatmap.length} cells scored`;

    const list = document.getElementById('rank-list');
    list.innerHTML = '';
    data.top_locations.forEach((loc, i) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="idx">${i + 1}</span>
            <span class="name">${loc.lat.toFixed(2)}, ${loc.lon.toFixed(2)}
                <small>n_obs in cell: ${loc.n_obs_in_cell}</small></span>
            <span class="prob">${fmtProb(loc.probability)}</span>
        `;
        li.addEventListener('click', () => {
            state.map.flyTo([loc.lat, loc.lon], 7, { duration: 0.8 });
            document.querySelectorAll('#rank-list li').forEach(x => x.classList.remove('highlight'));
            li.classList.add('highlight');
        });
        list.appendChild(li);
    });
}

function drawHeatmap(cells) {
    if (!cells || !cells.length) return;
    const step = state.gridInfo.lat_step / 2;
    cells.forEach(c => {
        // land / no-observation cells get dimmed a bit
        const dim = c.n_obs_in_cell === 0 ? 0.28 : 0.75;
        const rect = L.rectangle(
            [[c.lat - step, c.lon - step], [c.lat + step, c.lon + step]],
            {
                stroke: false,
                fill: true,
                fillColor: probColor(c.probability),
                fillOpacity: dim,
            }
        );
        rect.bindTooltip(
            `(${c.lat.toFixed(2)}, ${c.lon.toFixed(2)})<br>probability <b>${fmtProb(c.probability)}</b>` +
            `<br>n_obs in cell: ${c.n_obs_in_cell}`,
            { sticky: true },
        );
        rect.addTo(state.heatLayer);
    });
}

/* -------------------------------------------------------------------- */
/* helpers                                                              */
/* -------------------------------------------------------------------- */
function probColor(p) {
    // five-stop ramp matching the legend gradient
    const stops = [
        [0.00, [30, 64, 175]],
        [0.25, [14, 165, 233]],
        [0.50, [34, 197, 94]],
        [0.75, [250, 204, 21]],
        [1.00, [239, 68, 68]],
    ];
    const v = Math.max(0, Math.min(1, p));
    for (let i = 0; i < stops.length - 1; i++) {
        const [a, ca] = stops[i];
        const [b, cb] = stops[i + 1];
        if (v >= a && v <= b) {
            const t = (v - a) / (b - a || 1);
            const r = Math.round(ca[0] + (cb[0] - ca[0]) * t);
            const g = Math.round(ca[1] + (cb[1] - ca[1]) * t);
            const bl = Math.round(ca[2] + (cb[2] - ca[2]) * t);
            return `rgb(${r},${g},${bl})`;
        }
    }
    return '#888';
}

function fmtProb(p) {
    if (p == null) return '-';
    if (p < 0.0001) return p.toExponential(1);
    return p.toFixed(3);
}

function formatDate(s) {
    if (!s) return '-';
    return s.slice(0, 10);
}

function prettyName(s) {
    return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
        {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
    ));
}
