// Fairness page — per-attribute group metrics & disparity summary.

mountShell({
  activePage: 'fairness.html',
  title: 'Fairness Audit',
  subtitle: 'Group-level performance across gender, insurance, and age',
});

const root = document.getElementById('page-content');

const DISPARITY_THRESHOLD = 0.1;  // > 10% gap is flagged

function disparityLevel(v) {
  const x = Math.abs(v || 0);
  if (x < 0.05) return 'low';
  if (x < DISPARITY_THRESHOLD) return 'moderate';
  return 'high';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function fmt(v) { return (typeof v === 'number') ? v.toFixed(3) : '—'; }

root.innerHTML = `
  <div id="fair-summary" class="grid grid-cols-3" style="margin-bottom: var(--space-8);"></div>
  <div id="fair-attrs"></div>
  <div id="fair-empty" class="card empty-state" style="display: none;">
    <i data-lucide="scale"></i>
    <div class="empty-state-title">No fairness data</div>
    <div>Run the pipeline with fairness audit enabled to populate this page.</div>
  </div>
`;

lucide.createIcons();

function renderSummary(attrs) {
  const container = document.getElementById('fair-summary');
  container.innerHTML = attrs.map(a => {
    const maxDisp = Math.max(
      a.demographic_parity_difference || 0,
      a.equalized_odds_difference || 0,
      a.fpr_difference || 0,
      a.fnr_difference || 0,
    );
    const level = disparityLevel(maxDisp);
    const mod = level === 'high' ? 'stat-card--danger'
              : level === 'moderate' ? 'stat-card--warning'
              : 'stat-card--success';
    const palette = {
      high:     { bg: 'rgb(239 68 68 / 0.1)',  fg: 'var(--color-danger)'  },
      moderate: { bg: 'rgb(245 158 11 / 0.1)', fg: 'var(--color-warning)' },
      low:      { bg: 'rgb(16 185 129 / 0.1)', fg: 'var(--color-success)' },
    }[level];
    return `
      <div class="stat-card ${mod}">
        <div class="stat-card-icon" style="background: ${palette.bg}; color: ${palette.fg};">
          <i data-lucide="users"></i>
        </div>
        <div class="stat-card-label">${escapeHtml(a.attribute)}</div>
        <div class="stat-card-value">${fmt(maxDisp)}</div>
        <div class="stat-card-delta">max disparity · ${a.groups.length} groups</div>
      </div>
    `;
  }).join('');
  lucide.createIcons();
}

function renderAttribute(attr) {
  const metricKeys = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'selection_rate'];
  const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'Sel. rate'];
  const chartId = `chart-${attr.attribute}`;

  const disparityBadges = [
    ['Demographic parity Δ', attr.demographic_parity_difference],
    ['Equalized odds Δ',     attr.equalized_odds_difference],
    ['FPR Δ',                attr.fpr_difference],
    ['FNR Δ',                attr.fnr_difference],
  ].map(([label, v]) => {
    const lvl = disparityLevel(v);
    const cls = lvl === 'high' ? 'badge--danger' : lvl === 'moderate' ? 'badge--warning' : 'badge--success';
    return `<span class="badge ${cls}" title="${label}">${label}: ${fmt(v)}</span>`;
  }).join(' ');

  const tableRows = attr.groups.map(g => `
    <tr>
      <td style="font-weight: 600;">${escapeHtml(g.group)}</td>
      ${metricKeys.map(k => `<td class="metric-cell">${fmt(g[k])}</td>`).join('')}
    </tr>
  `).join('');

  return `
    <div class="card" style="margin-bottom: var(--space-6);">
      <div class="card-header">
        <div>
          <div class="card-title">${escapeHtml(attr.attribute)}</div>
          <div class="card-subtitle">${attr.groups.length} groups · disparity thresholds at 5% / 10%</div>
        </div>
        <div style="display: flex; gap: var(--space-2); flex-wrap: wrap; justify-content: flex-end;">
          ${disparityBadges}
        </div>
      </div>

      <div class="grid grid-cols-2" style="align-items: start; gap: var(--space-6);">
        <div>
          <div class="table-wrap">
            <div class="table-scroll">
              <table class="table">
                <thead>
                  <tr>
                    <th>Group</th>
                    ${metricLabels.map(l => `<th>${l}</th>`).join('')}
                  </tr>
                </thead>
                <tbody>${tableRows}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="chart-container"><canvas id="${chartId}"></canvas></div>
      </div>
    </div>
  `;
}

function renderGroupChart(attr) {
  const chartId = `chart-${attr.attribute}`;
  const labels = attr.groups.map(g => g.group);
  const keys = [
    ['accuracy',  'Accuracy',  charts.CHART_COLORS.primary],
    ['precision', 'Precision', charts.CHART_COLORS.accent],
    ['recall',    'Recall',    charts.CHART_COLORS.success],
    ['f1',        'F1',        charts.CHART_COLORS.warning],
  ];
  const datasets = keys.map(([k, label, color]) => ({
    label,
    data: attr.groups.map(g => g[k] || 0),
    backgroundColor: color,
    borderRadius: 4,
  }));
  charts.createBarChart(chartId, labels, datasets, {
    scales: {
      y: { beginAtZero: true, max: 1, grid: { color: '#f1f5f9' } },
      x: { grid: { display: false } },
    },
    plugins: { legend: { position: 'top', align: 'end' } },
  });
}

api.fairness().then(data => {
  const attrs = data.attributes || [];
  if (attrs.length === 0) {
    document.getElementById('fair-empty').style.display = 'block';
    return;
  }
  renderSummary(attrs);
  const host = document.getElementById('fair-attrs');
  host.innerHTML = attrs.map(renderAttribute).join('');
  lucide.createIcons();
  attrs.forEach(renderGroupChart);
}).catch(err => {
  console.error('Failed to load fairness:', err);
  const empty = document.getElementById('fair-empty');
  empty.style.display = 'block';
  empty.innerHTML = `
    <i data-lucide="alert-circle"></i>
    <div class="empty-state-title">Failed to load fairness data</div>
    <div>${escapeHtml(err.message)}</div>
  `;
  lucide.createIcons();
});
