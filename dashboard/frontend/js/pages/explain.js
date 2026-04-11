// Explain page — SHAP per-patient explanations.

mountShell({
  activePage: 'explain.html',
  title: 'SHAP Explainability',
  subtitle: 'See which features push the prediction toward (or away from) readmission',
});

const root = document.getElementById('page-content');

const SAMPLE_NOTE = `Patient is a 72-year-old female with severe chronic obstructive pulmonary disease (COPD), congestive heart failure with reduced ejection fraction (EF 25%), atrial fibrillation on warfarin, and chronic kidney disease. Admitted for acute COPD exacerbation requiring BiPAP support. Course complicated by acute kidney injury and decompensated heart failure. Treated with IV steroids, antibiotics, diuresis, and rate control. Length of stay 9 days. Discharged on home oxygen with close pulmonology follow-up.`;

root.innerHTML = `
  <div class="card" style="margin-bottom: var(--space-6);">
    <div class="card-header">
      <div>
        <div class="card-title">Discharge Note</div>
        <div class="card-subtitle">Provide a note to compute SHAP feature contributions</div>
      </div>
      <button type="button" id="load-sample" class="btn btn--ghost">
        <i data-lucide="file-plus"></i>
        <span>Load sample</span>
      </button>
    </div>

    <form id="explain-form">
      <div class="form-group">
        <textarea id="note-text" class="form-textarea" placeholder="Paste a clinical note here…" required></textarea>
      </div>

      <div class="form-row" style="max-width: 480px;">
        <div class="form-group">
          <label class="form-label" for="top-n">Top N features</label>
          <input id="top-n" class="form-input" type="number" min="3" max="30" value="10" />
        </div>
      </div>

      <button type="submit" id="explain-btn" class="btn btn--primary">
        <i data-lucide="sparkles"></i>
        <span>Compute SHAP</span>
      </button>
    </form>
  </div>

  <div id="explain-result" style="display: none;">
    <div class="grid grid-cols-2" style="margin-bottom: var(--space-6);">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Prediction</div>
            <div class="card-subtitle">Computed by <span id="ex-model">—</span></div>
          </div>
        </div>
        <div style="display: flex; align-items: baseline; gap: var(--space-3);">
          <div id="ex-prob" style="font-size: var(--text-4xl); font-weight: 700; color: var(--text-primary);">—</div>
          <div style="font-size: var(--text-sm); color: var(--text-muted);">readmission probability</div>
        </div>
        <div style="margin-top: var(--space-3); font-size: var(--text-xs); color: var(--text-muted); font-family: var(--font-mono);">
          base value = <span id="ex-base">—</span>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">SHAP Direction Summary</div>
            <div class="card-subtitle">How features balance out</div>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4);">
          <div>
            <div style="font-size: var(--text-xs); text-transform: uppercase; color: var(--text-muted); margin-bottom: var(--space-1);">Push ↑ risk</div>
            <div id="ex-up" style="font-size: var(--text-2xl); font-weight: 700; color: var(--risk-high);">—</div>
          </div>
          <div>
            <div style="font-size: var(--text-xs); text-transform: uppercase; color: var(--text-muted); margin-bottom: var(--space-1);">Push ↓ risk</div>
            <div id="ex-down" style="font-size: var(--text-2xl); font-weight: 700; color: var(--risk-low);">—</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Top Feature Contributions</div>
          <div class="card-subtitle">Red bars push probability up, green bars pull it down</div>
        </div>
      </div>
      <div id="shap-list" class="shap-list"></div>
    </div>
  </div>

  <div id="explain-empty" class="card empty-state">
    <i data-lucide="sparkles"></i>
    <div class="empty-state-title">No explanation yet</div>
    <div>Submit a clinical note above to compute SHAP values.</div>
  </div>
`;

lucide.createIcons();

const form = document.getElementById('explain-form');
const noteEl = document.getElementById('note-text');
const btn = document.getElementById('explain-btn');

document.getElementById('load-sample').addEventListener('click', () => {
  noteEl.value = SAMPLE_NOTE;
  noteEl.focus();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = noteEl.value.trim();
  if (!text) return;
  const topN = parseInt(document.getElementById('top-n').value, 10) || 10;

  setLoading(true);
  try {
    const res = await api.explain({ text, top_n: topN });
    renderResult(res);
  } catch (err) {
    renderError(err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(loading) {
  btn.disabled = loading;
  btn.innerHTML = loading
    ? `<span class="spinner"></span><span>Computing…</span>`
    : `<i data-lucide="sparkles"></i><span>Compute SHAP</span>`;
  if (window.lucide) lucide.createIcons();
}

function renderResult(res) {
  document.getElementById('explain-empty').style.display = 'none';
  document.getElementById('explain-result').style.display = 'block';

  document.getElementById('ex-prob').textContent = `${(res.probability * 100).toFixed(1)}%`;
  document.getElementById('ex-base').textContent = res.base_value.toFixed(3);
  document.getElementById('ex-model').textContent = res.model_name;

  const ups = res.top_features.filter(f => f.shap > 0).length;
  const downs = res.top_features.filter(f => f.shap < 0).length;
  document.getElementById('ex-up').textContent = ups;
  document.getElementById('ex-down').textContent = downs;

  // Bars: scale by max abs SHAP
  const maxAbs = Math.max(...res.top_features.map(f => Math.abs(f.shap)), 1e-9);
  const list = document.getElementById('shap-list');
  list.innerHTML = res.top_features.map(f => {
    const pct = (Math.abs(f.shap) / maxAbs) * 50; // half the bar (split center)
    const positive = f.shap > 0;
    const fillClass = positive ? 'shap-row-bar-fill--positive' : 'shap-row-bar-fill--negative';
    const sign = positive ? '+' : '−';
    return `
      <div class="shap-row">
        <div class="shap-row-name">${escapeHtml(f.feature)}</div>
        <div class="shap-row-bar">
          <div class="shap-row-bar-center"></div>
          <div class="shap-row-bar-fill ${fillClass}" style="width: ${pct}%;"></div>
        </div>
        <div class="shap-row-value">${sign}${Math.abs(f.shap).toFixed(3)}</div>
      </div>
    `;
  }).join('');
}

function renderError(msg) {
  document.getElementById('explain-result').style.display = 'none';
  const empty = document.getElementById('explain-empty');
  empty.style.display = 'block';
  empty.innerHTML = `
    <i data-lucide="alert-circle"></i>
    <div class="empty-state-title">Explanation failed</div>
    <div>${escapeHtml(msg || 'Unknown error')}</div>
  `;
  if (window.lucide) lucide.createIcons();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}
