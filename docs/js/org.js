let siteData = null;
let currentOrg = null;
let orgChart = null;
let venueDatasets = [];
let sortCol = 'total';
let sortDir = 'desc';
let allYears = [];
let columnWidthsOrg = {};

// Load saved column widths from localStorage
try {
    const saved = localStorage.getItem('pub_lister_col_widths_org');
    if (saved) columnWidthsOrg = JSON.parse(saved);
} catch (e) {
    columnWidthsOrg = {};
}

function getDefaultWidthOrg(colKey) {
    if (columnWidthsOrg[colKey]) return columnWidthsOrg[colKey];
    if (colKey === 'venue') return 220;
    if (colKey === 'total') return 180;
    return 85;
}

const COLOR_PALETTE = [
    '#2563eb', '#dc2626', '#16a34a', '#d97706', '#9333ea', 
    '#0891b2', '#ea580c', '#4f46e5', '#059669', '#be123c', 
    '#854d0e', '#57534e', '#ec4899', '#14b8a6', '#6366f1'
];

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('data/site_data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        siteData = await response.json();
        initOrgPage();
    } catch (err) {
        console.error('Failed to load site_data.json:', err);
        document.querySelector('main').innerHTML = `
            <div class="card" style="color: #dc2626; text-align: center; padding: 3rem;">
                <h2>Error Loading Data</h2>
                <p style="margin-top: 0.5rem;">Could not load <code>data/site_data.json</code>.</p>
            </div>
        `;
    }
});

function initOrgPage() {
    setupGlobalSearch();

    const urlParams = new URLSearchParams(window.location.search);
    const orgId = urlParams.get('id');

    if (!orgId || !siteData.organisations[orgId]) {
        showNotFound();
        return;
    }

    currentOrg = siteData.organisations[orgId];
    renderOrgHeader();

    // Determine all active years across all venues for this org
    const yearSet = new Set();
    Object.values(currentOrg.venues).forEach(v => {
        if (v.years) {
            Object.keys(v.years).forEach(y => yearSet.add(y));
        }
    });
    allYears = Array.from(yearSet).sort();

    renderChartAndToggles();
    setupTableAndListeners();
}

function showNotFound() {
    document.querySelector('main').innerHTML = `
        <a href="index.html" class="back-link">← Back to Dashboard</a>
        <div class="card" style="text-align: center; padding: 3rem;">
            <h2>Institution Not Found</h2>
            <p style="color:var(--text-muted); margin-top:0.5rem;">The requested institution ID could not be found in the dataset.</p>
        </div>
    `;
}

function renderOrgHeader() {
    document.title = `${currentOrg.canonical_name} - Publication Lister`;
    document.getElementById('orgCanonicalName').textContent = currentOrg.canonical_name;
    document.getElementById('orgCanonicalId').textContent = currentOrg.canonical_id;
    
    const badge = document.getElementById('orgEntityTypeBadge');
    badge.textContent = currentOrg.entity_type;
    badge.className = `badge badge-${currentOrg.entity_type || 'OTHER'}`;

    document.getElementById('orgTotalPubs').textContent = currentOrg.total_publications.toLocaleString();

    // Known Aliases
    const aliases = currentOrg.known_aliases || [];
    document.getElementById('aliasCount').textContent = aliases.length;
    const aliasContainer = document.getElementById('aliasTagsContainer');
    aliasContainer.innerHTML = aliases.map(a => `<span class="alias-tag">${escapeHtml(a)}</span>`).join('');

    // External links
    const externalContainer = document.getElementById('externalLinksContainer');
    let extHtml = '';
    if (currentOrg.openalex_source_ids && currentOrg.openalex_source_ids.length > 0) {
        currentOrg.openalex_source_ids.forEach(sid => {
            extHtml += `<a href="https://openalex.org/sources/${sid}" target="_blank" rel="noopener" class="btn btn-outline" style="padding:0.25rem 0.6rem; font-size:0.75rem;">OpenAlex Source (${sid}) ↗</a>`;
        });
    }
    externalContainer.innerHTML = extHtml;
}

function renderChartAndToggles() {
    const togglesContainer = document.getElementById('venueTogglesContainer');
    togglesContainer.innerHTML = '';
    venueDatasets = [];

    const venueEntries = Object.entries(currentOrg.venues);

    venueEntries.forEach(([venueName, vData], index) => {
        const color = COLOR_PALETTE[index % COLOR_PALETTE.length];
        const dataPoints = allYears.map(yr => (vData.years && vData.years[yr]) ? vData.years[yr] : 0);

        const dataset = {
            label: venueName,
            data: dataPoints,
            borderColor: color,
            backgroundColor: color,
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 7,
            fill: false,
            tension: 0.2,
            hidden: false
        };
        venueDatasets.push(dataset);

        // Create toggle element
        const toggleItem = document.createElement('label');
        toggleItem.className = 'venue-toggle-item';
        toggleItem.style.color = color;
        toggleItem.innerHTML = `
            <input type="checkbox" checked data-index="${index}">
            <span>${venueName} (${vData.total})</span>
        `;
        togglesContainer.appendChild(toggleItem);
    });

    // Chart init
    const ctx = document.getElementById('orgChart').getContext('2d');
    if (orgChart) {
        orgChart.destroy();
    }

    orgChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allYears,
            datasets: venueDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` ${context.dataset.label}: ${context.raw.toLocaleString()} pubs`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        callback: function(val) {
                            return Number.isInteger(val) ? val : '';
                        }
                    }
                }
            }
        }
    });

    // Toggle event listener
    togglesContainer.querySelectorAll('input[type="checkbox"]').forEach(chk => {
        chk.addEventListener('change', (e) => {
            const idx = parseInt(e.target.getAttribute('data-index'));
            const isChecked = e.target.checked;
            orgChart.setDatasetVisibility(idx, isChecked);
            orgChart.update();
        });
    });
}

function setupTableAndListeners() {
    document.getElementById('downloadOrgCsvBtn').addEventListener('click', downloadOrgCsv);
    renderOrgTable();
}

function renderOrgTable() {
    const headerRow = document.getElementById('orgTableHeaderRow');
    const tbody = document.getElementById('orgTableBody');

    // Headers with column resizers
    let headerHtml = `
        <th data-col="venue" style="width: ${getDefaultWidthOrg('venue')}px;">
            <span>Venue ${getSortIcon('venue')}</span>
            <div class="col-resizer" title="Drag to resize column"></div>
        </th>
        <th data-col="total" style="width: ${getDefaultWidthOrg('total')}px;">
            <span>Total Publications ${getSortIcon('total')}</span>
            <div class="col-resizer" title="Drag to resize column"></div>
        </th>
    `;
    allYears.forEach(yr => {
        headerHtml += `
            <th data-col="${yr}" style="width: ${getDefaultWidthOrg(yr)}px;">
                <span>${yr} ${getSortIcon(yr)}</span>
                <div class="col-resizer" title="Drag to resize column"></div>
            </th>
        `;
    });
    headerRow.innerHTML = headerHtml;

    headerRow.querySelectorAll('th').forEach(th => {
        const col = th.getAttribute('data-col');
        const resizer = th.querySelector('.col-resizer');

        th.addEventListener('click', (e) => {
            if (e.target.classList.contains('col-resizer')) return;
            if (sortCol === col) {
                sortDir = sortDir === 'asc' ? 'desc' : 'asc';
            } else {
                sortCol = col;
                sortDir = col === 'venue' ? 'asc' : 'desc';
            }
            renderOrgTable();
        });

        if (resizer) {
            setupResizerOrg(resizer, th, col, 'pub_lister_col_widths_org');
        }
    });

    // Data rows
    const rows = Object.entries(currentOrg.venues).map(([venueName, vData]) => {
        return {
            venue: venueName,
            total: vData.total || 0,
            years: vData.years || {}
        };
    });

    // Sort
    rows.sort((a, b) => {
        let valA, valB;
        if (sortCol === 'venue') {
            valA = a.venue;
            valB = b.venue;
        } else if (sortCol === 'total') {
            valA = a.total;
            valB = b.total;
        } else {
            valA = a.years[sortCol] || 0;
            valB = b.years[sortCol] || 0;
        }

        if (valA < valB) return sortDir === 'asc' ? -1 : 1;
        if (valA > valB) return sortDir === 'asc' ? 1 : -1;
        return 0;
    });

    tbody.innerHTML = '';
    rows.forEach(r => {
        const tr = document.createElement('tr');
        let rowHtml = `
            <td title="${escapeHtml(r.venue)}">
                <a href="index.html?venue=${encodeURIComponent(r.venue)}" class="org-link">
                    ${escapeHtml(r.venue)}
                </a>
            </td>
            <td><strong>${r.total.toLocaleString()}</strong></td>
        `;

        allYears.forEach(yr => {
            const cnt = r.years[yr] || 0;
            rowHtml += `<td>${cnt ? cnt.toLocaleString() : '-'}</td>`;
        });

        tr.innerHTML = rowHtml;
        tbody.appendChild(tr);
    });
}

function setupResizerOrg(resizer, th, colKey, storageKey) {
    resizer.addEventListener('click', (e) => e.stopPropagation());

    resizer.addEventListener('mousedown', (e) => {
        e.stopPropagation();
        e.preventDefault();

        const startX = e.pageX;
        const startWidth = th.offsetWidth;
        const minWidth = colKey === 'venue' ? 120 : 60;

        resizer.classList.add('resizing');
        document.body.classList.add('column-resizing');

        const onMouseMove = (moveEvent) => {
            const deltaX = moveEvent.pageX - startX;
            const newWidth = Math.max(minWidth, startWidth + deltaX);
            th.style.width = `${newWidth}px`;
            columnWidthsOrg[colKey] = newWidth;
        };

        const onMouseUp = () => {
            resizer.classList.remove('resizing');
            document.body.classList.remove('column-resizing');
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
            try {
                localStorage.setItem(storageKey, JSON.stringify(columnWidthsOrg));
            } catch (err) {}
        };

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
    });
}

function getSortIcon(col) {
    if (sortCol !== col) return `<span class="sort-icon">↕</span>`;
    return sortDir === 'asc' ? `<span class="sort-icon">▲</span>` : `<span class="sort-icon">▼</span>`;
}

function downloadOrgCsv() {
    if (!currentOrg) return;

    const headers = ['Venue', 'Total Publications', ...allYears];
    const csvLines = [headers.map(h => `"${h}"`).join(',')];

    Object.entries(currentOrg.venues).forEach(([venueName, vData]) => {
        const line = [
            `"${venueName}"`,
            vData.total || 0,
            ...allYears.map(yr => (vData.years && vData.years[yr]) ? vData.years[yr] : 0)
        ];
        csvLines.push(line.join(','));
    });

    const csvContent = csvLines.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${currentOrg.canonical_id}_publications.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function setupGlobalSearch() {
    const input = document.getElementById('globalSearchInput');
    const resultsContainer = document.getElementById('globalSearchResults');

    input.addEventListener('input', () => {
        const query = input.value.toLowerCase().trim();
        if (query.length < 2) {
            resultsContainer.style.display = 'none';
            return;
        }

        const orgs = Object.values(siteData.organisations);
        const matches = orgs.filter(o => {
            const nameMatch = o.canonical_name.toLowerCase().includes(query);
            const aliasMatch = (o.known_aliases || []).some(a => a.toLowerCase().includes(query));
            return nameMatch || aliasMatch;
        }).slice(0, 8);

        if (matches.length === 0) {
            resultsContainer.innerHTML = '<div style="color:var(--text-muted);">No matching institutions found</div>';
        } else {
            resultsContainer.innerHTML = matches.map(m => `
                <div data-id="${m.canonical_id}">
                    <strong>${escapeHtml(m.canonical_name)}</strong>
                    <span class="badge badge-${m.entity_type}">${m.entity_type}</span>
                    <span style="font-size:0.75rem; color:var(--text-muted); float:right;">${m.total_publications} pubs</span>
                </div>
            `).join('');

            resultsContainer.querySelectorAll('div[data-id]').forEach(el => {
                el.addEventListener('click', () => {
                    const orgId = el.getAttribute('data-id');
                    window.location.href = `organisation.html?id=${encodeURIComponent(orgId)}`;
                });
            });
        }
        resultsContainer.style.display = 'block';
    });

    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !resultsContainer.contains(e.target)) {
            resultsContainer.style.display = 'none';
        }
    });
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
