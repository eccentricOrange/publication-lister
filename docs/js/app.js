let siteData = null;
let activeVenueKey = null;
let currentChart = null;
let currentRows = [];
let filteredRows = [];
let sortColumn = 'total';
let sortDirection = 'desc';

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('data/site_data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        siteData = await response.json();
        initApp();
    } catch (err) {
        console.error('Failed to load site_data.json:', err);
        document.querySelector('main').innerHTML = `
            <div class="card" style="color: #dc2626; text-align: center; padding: 3rem;">
                <h2>Error Loading Data</h2>
                <p style="margin-top: 0.5rem;">Could not load <code>data/site_data.json</code>. Please ensure the build script has run.</p>
                <code style="display:inline-block; margin-top:1rem; padding:0.5rem; background:#fee2e2;">python main.py build-visualisation</code>
            </div>
        `;
    }
});

function initApp() {
    const venueKeys = Object.keys(siteData.venues);
    if (venueKeys.length === 0) return;

    // Check URL query string for venue
    const urlParams = new URLSearchParams(window.location.search);
    const requestedVenue = urlParams.get('venue');
    activeVenueKey = (requestedVenue && siteData.venues[requestedVenue]) ? requestedVenue : venueKeys[0];

    renderTabs(venueKeys);
    setupGlobalSearch();
    setupTableListeners();

    // Initial load
    switchVenue(activeVenueKey);
}

function renderTabs(venueKeys) {
    const container = document.getElementById('venueTabsContainer');
    container.innerHTML = '';

    venueKeys.forEach(vKey => {
        const btn = document.createElement('button');
        btn.className = `tab-btn ${vKey === activeVenueKey ? 'active' : ''}`;
        btn.textContent = vKey;
        btn.addEventListener('click', () => switchVenue(vKey));
        container.appendChild(btn);
    });
}

function switchVenue(vKey) {
    activeVenueKey = vKey;

    // Update active tab UI
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.textContent === vKey);
    });

    // Update URL query string without reloading page
    const newUrl = new URL(window.location.href);
    newUrl.searchParams.set('venue', vKey);
    window.history.pushState({ venue: vKey }, '', newUrl);

    const venue = siteData.venues[vKey];
    currentRows = venue.rows || [];

    // Render Stats
    renderStats(venue);

    // Render Chart
    renderChart(venue);

    // Render Table
    sortColumn = 'total';
    sortDirection = 'desc';
    applyFiltersAndSort();
}

function renderStats(venue) {
    document.getElementById('statTotalPubs').textContent = venue.total_publications.toLocaleString();
    document.getElementById('statYearRange').textContent = `Spanning ${venue.year_range}`;
    document.getElementById('statTotalInsts').textContent = venue.total_institutions.toLocaleString();

    if (venue.top_institution) {
        document.getElementById('statTopInstName').textContent = venue.top_institution.canonical_name;
        document.getElementById('statTopInstName').title = venue.top_institution.canonical_name;
        document.getElementById('statTopInstCount').textContent = `${venue.top_institution.total.toLocaleString()} publications`;
    } else {
        document.getElementById('statTopInstName').textContent = 'N/A';
        document.getElementById('statTopInstCount').textContent = '-';
    }

    const link = document.getElementById('statVenueLink');
    if (venue.venue_url) {
        link.href = venue.venue_url;
        link.style.display = 'inline';
    } else {
        link.style.display = 'none';
    }

    document.getElementById('statVenueFreq').textContent = `Frequency: ${venue.frequency || 'Annual'}`;
    document.getElementById('chartVenueTitle').textContent = `${venue.venue} Publication Trend over Time`;
}

function renderChart(venue) {
    const ctx = document.getElementById('venueChart').getContext('2d');
    if (currentChart) {
        currentChart.destroy();
    }

    const years = venue.years;
    const totals = years.map(y => venue.yearly_totals[y] || 0);

    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: `${venue.venue} Total Publications`,
                data: totals,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                borderWidth: 3,
                pointBackgroundColor: '#2563eb',
                pointRadius: 5,
                pointHoverRadius: 8,
                fill: true,
                tension: 0.2
            }]
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
                            return ` ${context.raw.toLocaleString()} publications`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

function setupTableListeners() {
    document.getElementById('tableSearchInput').addEventListener('input', () => applyFiltersAndSort());
    document.getElementById('typeFilterSelect').addEventListener('change', () => applyFiltersAndSort());
    document.getElementById('downloadCsvBtn').addEventListener('click', () => downloadCsv());
}

function applyFiltersAndSort() {
    const searchTerm = document.getElementById('tableSearchInput').value.toLowerCase().trim();
    const selectedType = document.getElementById('typeFilterSelect').value;

    const venue = siteData.venues[activeVenueKey];
    if (!venue) return;

    filteredRows = currentRows.filter(row => {
        // Search filter (name + known aliases)
        let matchesSearch = true;
        if (searchTerm) {
            const nameMatch = row.canonical_name.toLowerCase().includes(searchTerm);
            const aliasMatch = (row.known_aliases || []).some(a => a.toLowerCase().includes(searchTerm));
            matchesSearch = nameMatch || aliasMatch;
        }

        // Type filter
        let matchesType = true;
        if (selectedType !== 'ALL') {
            matchesType = (row.entity_type === selectedType);
        }

        return matchesSearch && matchesType;
    });

    // Sort
    filteredRows.sort((a, b) => {
        let valA, valB;
        if (sortColumn === 'name') {
            valA = a.canonical_name.toLowerCase();
            valB = b.canonical_name.toLowerCase();
        } else if (sortColumn === 'type') {
            valA = a.entity_type;
            valB = b.entity_type;
        } else if (sortColumn === 'total') {
            valA = a.total || 0;
            valB = b.total || 0;
        } else {
            // Year column
            valA = (a.years && a.years[sortColumn]) || 0;
            valB = (b.years && b.years[sortColumn]) || 0;
        }

        if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
        if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
        return 0;
    });

    renderTable(venue);
}

function renderTable(venue) {
    const headerRow = document.getElementById('tableHeaderRow');
    const tbody = document.getElementById('tableBody');

    // Headers
    let headerHtml = `
        <th data-col="name">Institution ${getSortIcon('name')}</th>
        <th data-col="type">Type ${getSortIcon('type')}</th>
        <th data-col="total">Total ${getSortIcon('total')}</th>
    `;
    venue.years.forEach(yr => {
        headerHtml += `<th data-col="${yr}">${yr} ${getSortIcon(yr)}</th>`;
    });
    headerRow.innerHTML = headerHtml;

    // Attach click listeners to headers for sorting
    headerRow.querySelectorAll('th').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.getAttribute('data-col');
            if (sortColumn === col) {
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                sortColumn = col;
                sortDirection = (col === 'name' || col === 'type') ? 'asc' : 'desc';
            }
            applyFiltersAndSort();
        });
    });

    // Rows
    tbody.innerHTML = '';
    if (filteredRows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="${3 + venue.years.length}" style="text-align:center; padding:2rem; color: var(--text-muted);">No institutions matching criteria</td></tr>`;
        return;
    }

    filteredRows.forEach(row => {
        const tr = document.createElement('tr');
        const badgeClass = `badge-${row.entity_type || 'OTHER'}`;

        let rowHtml = `
            <td>
                <a href="organisation.html?id=${encodeURIComponent(row.canonical_id)}" class="org-link">
                    ${escapeHtml(row.canonical_name)}
                </a>
            </td>
            <td><span class="badge ${badgeClass}">${row.entity_type}</span></td>
            <td><strong>${(row.total || 0).toLocaleString()}</strong></td>
        `;

        venue.years.forEach(yr => {
            const cnt = (row.years && row.years[yr]) ? row.years[yr] : 0;
            rowHtml += `<td>${cnt ? cnt.toLocaleString() : '-'}</td>`;
        });

        tr.innerHTML = rowHtml;
        tbody.appendChild(tr);
    });
}

function getSortIcon(col) {
    if (sortColumn !== col) return `<span class="sort-icon">↕</span>`;
    return sortDirection === 'asc' ? `<span class="sort-icon">▲</span>` : `<span class="sort-icon">▼</span>`;
}

function downloadCsv() {
    const venue = siteData.venues[activeVenueKey];
    if (!venue || filteredRows.length === 0) return;

    const headers = ['Canonical ID', 'Institution Name', 'Entity Type', 'Total Publications', ...venue.years];
    const csvLines = [headers.map(h => `"${h}"`).join(',')];

    filteredRows.forEach(row => {
        const line = [
            `"${row.canonical_id}"`,
            `"${row.canonical_name.replace(/"/g, '""')}"`,
            `"${row.entity_type}"`,
            row.total || 0,
            ...venue.years.map(yr => (row.years && row.years[yr]) ? row.years[yr] : 0)
        ];
        csvLines.push(line.join(','));
    });

    const csvContent = csvLines.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${activeVenueKey}_affiliations.csv`);
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

