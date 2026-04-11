// Overview page — top-level dashboard with KPIs and a model comparison preview.

mountShell({
  activePage: 'index.html',
  title: 'Dashboard Overview',
  subtitle: 'Clinical readmission prediction · 30-day risk · synthetic MIMIC-IV',
  actions: `
    <a href="predict.html" class="btn btn--primary">
      <i data-lucide="zap"></i>
      <span>New Prediction</span>
    </a>
  `,
});

const root = document.getElementById('page-content');

root.innerHTML = `
  <div class="grid grid-cols-4" style="margin-bottom: var(--space-8);">
    <div class="stat-card stat-card--accent">
      <div class="stat-card-icon"><i data-lucide="database"></i></div>
      <div class="stat-card-label">Patients</div>
      <div class="stat-card-value" id="kpi-patients">—</div>
      <div class="stat-card-delta">Synthetic MIMIC-IV cohort</div>
    </div>
    <div class="stat-card stat-card--success">
      <div class="stat-card-icon" style="background: rgb(16 185 129 / 0.1); color: var(--color-success);">
        <i data-lucide="cpu"></i>
      </div>
      <div class="stat-card-label">Models Trained</div>
      <div class="stat-card-value" id="kpi-models">—</div>
      <div class="stat-card-delta">LR · RF · XGBoost · LightGBM</div>
    </div>
    <div class="stat-card stat-card--warning">
      <div class="stat-card-icon" style="background: rgb(245 158 11 / 0.1); color: var(--color-warning);">
        <i data-lucide="layers"></i>
      </div>
      <div class="stat-card-label">Feature Sets</div>
      <div class="stat-card-value" id="kpi-features">—</div>
      <div class="stat-card-delta">TF-IDF · LDA · BERT · structured</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-icon"><i data-lucide="trophy"></i></div>
      <div class="stat-card-label">Best ROC-AUC</div>
      <div class="stat-card-value" id="kpi-best">—</div>
      <div class="stat-card-delta" id="kpi-best-meta">—</div>
    </div>
  </div>

  <div class="grid grid-cols-2" style="margin-bottom: var(--space-8);">
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Best Model — Performance</div>
          <div class="card-subtitle">Metrics across the top model & feature combination</div>
        </div>
      </div>
      <div class="chart-container"><canvas id="best-metrics-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">ROC-AUC by Model</div>
          <div class="card-subtitle">Best feature set per model</div>
        </div>
      </div>
      <div class="chart-container"><canvas id="model-rocauc-chart"></canvas></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header">
      <div>
        <div class="card-title">Pipeline Stages</div>
        <div class="card-subtitle">End-to-end clinical NLP workflow</div>
      </div>
    </div>
    <div class="grid grid-cols-3">
      ${[
        ['file-text', 'Data Loading', 'Discharge notes + admissions + demographics merged'],
        ['filter', 'Preprocessing', 'PHI removal, tokenization, clinical stopwords'],
        ['layers', 'Feature Engineering', 'TF-IDF, LDA, BERT, structured, combined'],
        ['target', 'Hyperparameter Tuning', 'RandomizedSearchCV with 5-fold CV'],
        ['scissors', 'Feature Selection', 'Univariate / RFE / SHAP / L1'],
        ['sparkles', 'SHAP Explainability', 'Per-patient + global feature importance'],
      ].map(([icon, title, desc]) => `
        <div style="display: flex; gap: var(--space-3); align-items: flex-start;">
          <div class="stat-card-icon" style="margin: 0; width: 36px; height: 36px;">
            <i data-lucide="${icon}"></i>
          </div>
          <div>
            <div style="font-weight: 600; font-size: var(--text-sm); color: var(--text-primary);">${title}</div>
            <div style="font-size: var(--text-xs); color: var(--text-muted); margin-top: 2px;">${desc}</div>
          </div>
        </div>
      `).join('')}
    </div>
  </div>
`;

lucide.createIcons();

// Load data
api.results().then(data => {
  const results = data.results || [];
  const best = data.best || results[0];

  document.getElementById('kpi-patients').textContent = '200';
  document.getElementById('kpi-models').textContent = data.n_models || '—';
  document.getElementById('kpi-features').textContent = data.n_feature_sets || '—';
  document.getElementById('kpi-best').textContent = best ? best.roc_auc.toFixed(3) : '—';
  document.getElementById('kpi-best-meta').textContent = best ? `${best.model} · ${best.feature_type}` : '—';

  // Best model metrics chart (radar-ish bar)
  if (best) {
    charts.createBarChart('best-metrics-chart',
      ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'],
      [{
        label: `${best.model} (${best.feature_type})`,
        data: [best.accuracy, best.precision, best.recall, best.f1, best.roc_auc, best.pr_auc],
        backgroundColor: charts.CHART_COLORS.primaryLight,
        borderColor: charts.CHART_COLORS.primary,
        borderWidth: 2,
        borderRadius: 6,
      }],
      { scales: { y: { beginAtZero: true, max: 1, grid: { color: '#f1f5f9' } }, x: { grid: { display: false } } } }
    );
  }

  // ROC-AUC by model — best feature set per model
  const byModel = {};
  results.forEach(r => {
    if (!byModel[r.model] || byModel[r.model].roc_auc < r.roc_auc) {
      byModel[r.model] = r;
    }
  });
  const labels = Object.keys(byModel);
  const data2 = labels.map(m => byModel[m].roc_auc);

  charts.createBarChart('model-rocauc-chart', labels, [{
    label: 'ROC-AUC',
    data: data2,
    backgroundColor: labels.map((_, i) => charts.CHART_PALETTE[i % charts.CHART_PALETTE.length]),
    borderRadius: 6,
  }], {
    indexAxis: 'y',
    plugins: { legend: { display: false } },
    scales: {
      x: { beginAtZero: true, max: 1, grid: { color: '#f1f5f9' } },
      y: { grid: { display: false } },
    },
  });
}).catch(err => {
  console.error('Failed to load results:', err);
});
