// Topics page — BERTopic clusters, weight word clouds, and readmission-rate chart.

mountShell({
  activePage: 'topics.html',
  title: 'Topic Modeling',
  subtitle: 'Latent clinical topics discovered in discharge notes',
});

const root = document.getElementById('page-content');

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function fmt(v, d = 3) { return (typeof v === 'number') ? v.toFixed(d) : '—'; }

function wordSizeClass(rankFromTop) {
  if (rankFromTop === 0) return 'topic-word--xl';
  if (rankFromTop <= 2) return 'topic-word--lg';
  return '';
}

root.innerHTML = `
  <div class="grid grid-cols-3" style="margin-bottom: var(--space-8);">
    <div class="stat-card stat-card--accent">
      <div class="stat-card-icon" style="background: rgb(139 92 246 / 0.1); color: var(--color-accent);">
        <i data-lucide="tags"></i>
      </div>
      <div class="stat-card-label">Topics Discovered</div>
      <div class="stat-card-value" id="kpi-ntopics">—</div>
      <div class="stat-card-delta">BERTopic · c-TF-IDF</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-icon"><i data-lucide="gauge"></i></div>
      <div class="stat-card-label">Coherence (c_v)</div>
      <div class="stat-card-value" id="kpi-coherence">—</div>
      <div class="stat-card-delta">Higher = more semantically tight</div>
    </div>
    <div class="stat-card stat-card--warning">
      <div class="stat-card-icon" style="background: rgb(245 158 11 / 0.1); color: var(--color-warning);">
        <i data-lucide="activity"></i>
      </div>
      <div class="stat-card-label">Max Readmission Rate</div>
      <div class="stat-card-value" id="kpi-maxrate">—</div>
      <div class="stat-card-delta" id="kpi-maxrate-label">—</div>
    </div>
  </div>

  <div class="card" style="margin-bottom: var(--space-8);">
    <div class="card-header">
      <div>
        <div class="card-title">Readmission Rate by Topic</div>
        <div class="card-subtitle">Which clinical topics correlate with higher 30-day readmission</div>
      </div>
    </div>
    <div class="chart-container"><canvas id="topic-rate-chart"></canvas></div>
  </div>

  <div class="card-header" style="margin-bottom: var(--space-4);">
    <div>
      <div class="card-title" style="font-size: var(--text-lg);">Topic Word Clouds</div>
      <div class="card-subtitle">Top terms per topic — size reflects c-TF-IDF weight</div>
    </div>
  </div>

  <div id="topic-grid" class="grid grid-cols-2"></div>

  <div id="topic-empty" class="card empty-state" style="display: none;">
    <i data-lucide="tags"></i>
    <div class="empty-state-title">No topics yet</div>
    <div>Run the topic modeling stage of the pipeline to populate this page.</div>
  </div>
`;

lucide.createIcons();

function renderTopicCards(topics) {
  const host = document.getElementById('topic-grid');
  host.innerHTML = topics.map(t => {
    const words = (t.words || []).map((w, i) => `
      <span class="topic-word ${wordSizeClass(i)}">${escapeHtml(w.word)}</span>
    `).join('');
    const rate = (typeof t.readmission_rate === 'number')
      ? `<span class="badge ${t.readmission_rate > 0.3 ? 'badge--danger' : t.readmission_rate > 0.2 ? 'badge--warning' : 'badge--success'}">
           readmit ${(t.readmission_rate * 100).toFixed(1)}%
         </span>`
      : '';
    return `
      <div class="topic-card">
        <div class="topic-card-header">
          <div>
            <div class="topic-card-title">Topic ${t.topic_id} · ${escapeHtml(t.label || '')}</div>
          </div>
          ${rate}
        </div>
        <div class="topic-words">${words}</div>
      </div>
    `;
  }).join('');
}

function renderRateChart(topics) {
  const sorted = topics.slice().sort((a, b) => (b.readmission_rate || 0) - (a.readmission_rate || 0));
  const labels = sorted.map(t => `#${t.topic_id} ${String(t.label || '').split(/[\/·]/)[0].trim().slice(0, 18)}`);
  const rates = sorted.map(t => t.readmission_rate || 0);
  const colors = rates.map(r =>
    r > 0.3 ? charts.CHART_COLORS.danger
    : r > 0.2 ? charts.CHART_COLORS.warning
    : charts.CHART_COLORS.success
  );
  charts.createBarChart('topic-rate-chart', labels, [{
    label: '30-day readmission rate',
    data: rates,
    backgroundColor: colors,
    borderRadius: 6,
  }], {
    indexAxis: 'y',
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `${(ctx.raw * 100).toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        beginAtZero: true,
        max: Math.min(1, Math.max(0.5, Math.max(...rates) * 1.2)),
        grid: { color: '#f1f5f9' },
        ticks: { callback: v => `${(v * 100).toFixed(0)}%` },
      },
      y: { grid: { display: false } },
    },
  });
}

api.topics().then(data => {
  const topics = data.topics || [];
  if (topics.length === 0) {
    document.getElementById('topic-empty').style.display = 'block';
    return;
  }

  document.getElementById('kpi-ntopics').textContent = data.n_topics || topics.length;
  document.getElementById('kpi-coherence').textContent =
    (typeof data.coherence_score === 'number') ? fmt(data.coherence_score, 4) : '—';

  const withRate = topics.filter(t => typeof t.readmission_rate === 'number');
  if (withRate.length > 0) {
    const top = withRate.reduce((a, b) => (a.readmission_rate > b.readmission_rate ? a : b));
    document.getElementById('kpi-maxrate').textContent = `${(top.readmission_rate * 100).toFixed(1)}%`;
    document.getElementById('kpi-maxrate-label').textContent = `Topic ${top.topic_id} · ${top.label || ''}`;
  }

  renderTopicCards(topics);
  renderRateChart(topics);
}).catch(err => {
  console.error('Failed to load topics:', err);
  const empty = document.getElementById('topic-empty');
  empty.style.display = 'block';
  empty.innerHTML = `
    <i data-lucide="alert-circle"></i>
    <div class="empty-state-title">Failed to load topics</div>
    <div>${escapeHtml(err.message)}</div>
  `;
  lucide.createIcons();
});
