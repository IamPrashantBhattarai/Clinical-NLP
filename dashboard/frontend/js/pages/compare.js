// Compare page — full model-vs-feature-set comparison table + charts.

mountShell({
  activePage: 'compare.html',
  title: 'Model Comparison',
  subtitle: 'Every model × feature-set combination, side by side',
  actions: `
    <a href="index.html" class="btn btn--ghost">
      <i data-lucide="arrow-left"></i>
      <span>Back to overview</span>
    </a>
  `,
});

const root = document.getElementById('page-content');

const METRICS = [
  { key: 'accuracy',  label: 'Accuracy'  },
  { key: 'precision', label: 'Precision' },
  { key: 'recall',    label: 'Recall'    },
  { key: 'f1',        label: 'F1'        },
  { key: 'roc_auc',   label: 'ROC-AUC'   },
  { key: 'pr_auc',    label: 'PR-AUC'    },
];

root.innerHTML = `
  <div class="grid grid-cols-2" style="margin-bottom: var(--space-8);">
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Metric Distribution</div>
          <div class="card-subtitle">All runs — pick a metric to compare</div>
        </div>
        <select id="metric-select" class="form-select" style="max-width: 180px;">
          ${METRICS.map(m => `<option value="${m.key}">${m.label}</option>`).join('')}
        </select>
      </div>
      <div class="chart-container"><canvas id="metric-chart"></canvas></div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Best Model — Full Profile</div>
          <div class="card-subtitle">Radar of every metric for the leading run</div>
        </div>
      </div>
      <div class="chart-container"><canvas id="radar-chart"></canvas></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header">
      <div>
        <div class="card-title">All Runs</div>
        <div class="card-subtitle">Sorted by ROC-AUC · best row highlighted</div>
      </div>
      <div style="display: flex; gap: var(--space-2); align-items: center;">
        <select id="filter-feature" class="form-select" style="max-width: 200px;">
          <option value="">All feature sets</option>
        </select>
        <select id="filter-model" class="form-select" style="max-width: 200px;">
          <option value="">All models</option>
        </select>
      </div>
    </div>
    <div class="table-wrap">
      <div class="table-scroll">
        <table class="table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Feature Set</th>
              ${METRICS.map(m => `<th>${m.label}</th>`).join('')}
            </tr>
          </thead>
          <tbody id="results-tbody">
            <tr><td colspan="${METRICS.length + 2}" style="text-align: center; color: var(--text-muted); padding: var(--space-6);">Loading…</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
`;

lucide.createIcons();

// ─────────────────────────────────────────────
// State + data fetch
// ─────────────────────────────────────────────

let ALL_RESULTS = [];
let BEST = null;
let metricChart = null;
let radarChart = null;

function fmt(v) {
  return (typeof v === 'number') ? v.toFixed(3) : '—';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function prettyModel(m) {
  return String(m).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function rowKey(r) {
  return `${r.model}__${r.feature_type}`;
}

function renderTable() {
  const fModel = document.getElementById('filter-model').value;
  const fFeat = document.getElementById('filter-feature').value;
  const rows = ALL_RESULTS
    .filter(r => !fModel || r.model === fModel)
    .filter(r => !fFeat || r.feature_type === fFeat)
    .slice()
    .sort((a, b) => b.roc_auc - a.roc_auc);

  const bestKey = BEST ? rowKey(BEST) : null;
  const tbody = document.getElementById('results-tbody');
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="${METRICS.length + 2}" style="text-align: center; color: var(--text-muted); padding: var(--space-6);">No rows match the current filters.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const isBest = rowKey(r) === bestKey;
    return `
      <tr class="${isBest ? 'is-best' : ''}">
        <td style="font-weight: 600;">
          ${isBest ? '<i data-lucide="trophy" style="width:14px;height:14px;color:var(--color-primary);vertical-align:-2px;margin-right:4px;"></i>' : ''}
          ${escapeHtml(prettyModel(r.model))}
        </td>
        <td><span class="badge">${escapeHtml(r.feature_type)}</span></td>
        ${METRICS.map(m => `<td class="metric-cell">${fmt(r[m.key])}</td>`).join('')}
      </tr>
    `;
  }).join('');
  lucide.createIcons();
}

function buildFilters() {
  const models = [...new Set(ALL_RESULTS.map(r => r.model))];
  const feats = [...new Set(ALL_RESULTS.map(r => r.feature_type))];
  const fm = document.getElementById('filter-model');
  const ff = document.getElementById('filter-feature');
  models.forEach(m => {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = prettyModel(m);
    fm.appendChild(o);
  });
  feats.forEach(f => {
    const o = document.createElement('option');
    o.value = f;
    o.textContent = f;
    ff.appendChild(o);
  });
  fm.addEventListener('change', renderTable);
  ff.addEventListener('change', renderTable);
}

function renderMetricChart(metricKey) {
  // Grouped bar: x = feature sets, one series per model
  const feats = [...new Set(ALL_RESULTS.map(r => r.feature_type))];
  const models = [...new Set(ALL_RESULTS.map(r => r.model))];
  const datasets = models.map((m, i) => ({
    label: prettyModel(m),
    data: feats.map(f => {
      const hit = ALL_RESULTS.find(r => r.model === m && r.feature_type === f);
      return hit ? hit[metricKey] : 0;
    }),
    backgroundColor: charts.CHART_PALETTE[i % charts.CHART_PALETTE.length],
    borderRadius: 4,
  }));

  if (metricChart) metricChart.destroy();
  metricChart = charts.createBarChart('metric-chart', feats, datasets, {
    scales: {
      y: { beginAtZero: true, max: 1, grid: { color: '#f1f5f9' } },
      x: { grid: { display: false } },
    },
    plugins: { legend: { position: 'top', align: 'end' } },
  });
}

function renderRadar() {
  if (!BEST) return;
  const ctx = document.getElementById('radar-chart');
  if (!ctx) return;
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: METRICS.map(m => m.label),
      datasets: [{
        label: `${prettyModel(BEST.model)} · ${BEST.feature_type}`,
        data: METRICS.map(m => BEST[m.key]),
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderColor: charts.CHART_COLORS.primary,
        borderWidth: 2,
        pointBackgroundColor: charts.CHART_COLORS.primary,
        pointRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          max: 1,
          ticks: { stepSize: 0.2, backdropColor: 'transparent' },
          grid: { color: '#e2e8f0' },
          angleLines: { color: '#e2e8f0' },
          pointLabels: { font: { size: 12, weight: '500' } },
        },
      },
      plugins: { legend: { position: 'top', align: 'end' } },
    },
  });
}

api.results().then(data => {
  ALL_RESULTS = data.results || [];
  BEST = data.best || ALL_RESULTS[0] || null;

  if (ALL_RESULTS.length === 0) {
    document.getElementById('results-tbody').innerHTML = `
      <tr><td colspan="${METRICS.length + 2}" style="text-align: center; color: var(--text-muted); padding: var(--space-6);">
        No results yet. Run the pipeline to populate this page.
      </td></tr>`;
    return;
  }

  buildFilters();
  renderTable();
  renderMetricChart('roc_auc');
  renderRadar();

  document.getElementById('metric-select').addEventListener('change', e => {
    renderMetricChart(e.target.value);
  });
}).catch(err => {
  console.error('Failed to load results:', err);
  document.getElementById('results-tbody').innerHTML = `
    <tr><td colspan="${METRICS.length + 2}" style="text-align: center; color: var(--color-danger); padding: var(--space-6);">
      Failed to load results: ${escapeHtml(err.message)}
    </td></tr>`;
});
