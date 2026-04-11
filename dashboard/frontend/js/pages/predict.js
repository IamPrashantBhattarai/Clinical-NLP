// Predict page — clinical note in, readmission risk out.

mountShell({
  activePage: 'predict.html',
  title: 'Predict Readmission Risk',
  subtitle: 'Paste a discharge note and structured features to score 30-day readmission risk',
});

const root = document.getElementById('page-content');

const SAMPLE_NOTE = `Patient is a 67-year-old male admitted with acute exacerbation of congestive heart failure. History of NYHA class III CHF, type 2 diabetes mellitus, chronic kidney disease stage 3, and hypertension. On admission, patient presented with worsening dyspnea, orthopnea, and lower extremity edema. BNP 1850. Treated with IV furosemide, ACE inhibitor titration, and beta-blocker optimization. Echo showed EF 30%. Discharged on day 6 in stable condition with adjusted medication regimen. Follow-up with cardiology in 1 week.`;

root.innerHTML = `
  <div class="grid grid-cols-2" style="align-items: start; gap: var(--space-6);">
    <!-- LEFT: Input form -->
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Patient Input</div>
          <div class="card-subtitle">Discharge note + structured features</div>
        </div>
        <button type="button" id="load-sample" class="btn btn--ghost">
          <i data-lucide="file-plus"></i>
          <span>Load sample</span>
        </button>
      </div>

      <form id="predict-form">
        <div class="form-group">
          <label class="form-label" for="note-text">Discharge Note</label>
          <textarea
            id="note-text"
            class="form-textarea"
            placeholder="Paste a clinical discharge note here…"
            required></textarea>
          <div class="form-help">Free text — will be cleaned & vectorized server-side.</div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="form-label" for="age">Age</label>
            <input id="age" class="form-input" type="number" min="0" max="120" value="65" />
          </div>
          <div class="form-group">
            <label class="form-label" for="gender">Gender</label>
            <select id="gender" class="form-select">
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="form-label" for="insurance">Insurance</label>
            <select id="insurance" class="form-select">
              <option value="Medicare">Medicare</option>
              <option value="Medicaid">Medicaid</option>
              <option value="Private">Private</option>
              <option value="Other">Other</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label" for="los">Length of Stay (days)</label>
            <input id="los" class="form-input" type="number" min="0" step="0.5" value="6" />
          </div>
        </div>

        <div style="display: flex; gap: var(--space-3); margin-top: var(--space-4);">
          <button type="submit" id="predict-btn" class="btn btn--primary">
            <i data-lucide="zap"></i>
            <span>Predict Risk</span>
          </button>
          <button type="reset" id="reset-btn" class="btn btn--secondary">
            <i data-lucide="rotate-ccw"></i>
            <span>Reset</span>
          </button>
        </div>
      </form>
    </div>

    <!-- RIGHT: Risk gauge + result -->
    <div class="risk-display" id="risk-display">
      <div class="risk-circle" id="risk-circle">
        <svg viewBox="0 0 200 200">
          <circle class="risk-circle-bg" cx="100" cy="100" r="84"></circle>
          <circle
            class="risk-circle-fg"
            id="risk-circle-fg"
            cx="100" cy="100" r="84"
            stroke-dasharray="527.8"
            stroke-dashoffset="527.8"></circle>
        </svg>
        <div class="risk-circle-text">
          <div class="risk-circle-value" id="risk-value">—</div>
          <div class="risk-circle-label">Probability</div>
        </div>
      </div>

      <div id="risk-badge-wrap" style="margin-bottom: var(--space-4);">
        <span class="risk-badge" id="risk-badge">
          <i data-lucide="circle-help"></i>
          <span>Awaiting input</span>
        </span>
      </div>

      <div id="risk-meta" style="font-size: var(--text-sm); color: var(--text-secondary);">
        Submit a note to see the model's prediction.
      </div>

      <div id="risk-details" style="display: none; margin-top: var(--space-6); padding-top: var(--space-6); border-top: 1px solid var(--border-color); text-align: left;">
        <div style="display: grid; grid-template-columns: auto 1fr; gap: var(--space-2) var(--space-4); font-size: var(--text-sm);">
          <div style="color: var(--text-muted);">Model</div>
          <div id="detail-model" style="font-weight: 600;">—</div>
          <div style="color: var(--text-muted);">Feature set</div>
          <div id="detail-features" style="font-weight: 600;">—</div>
          <div style="color: var(--text-muted);">Threshold</div>
          <div id="detail-threshold" style="font-family: var(--font-mono);">—</div>
          <div style="color: var(--text-muted);">Predicted label</div>
          <div id="detail-label" style="font-weight: 600;">—</div>
        </div>

        <div style="margin-top: var(--space-4); display: flex; gap: var(--space-2);">
          <a href="explain.html" class="btn btn--secondary" style="flex: 1;">
            <i data-lucide="sparkles"></i>
            <span>Explain this prediction</span>
          </a>
        </div>
      </div>
    </div>
  </div>
`;

lucide.createIcons();

// ─────────────────────────────────────────────
// Form interactions
// ─────────────────────────────────────────────

const form = document.getElementById('predict-form');
const noteEl = document.getElementById('note-text');
const btn = document.getElementById('predict-btn');

document.getElementById('load-sample').addEventListener('click', () => {
  noteEl.value = SAMPLE_NOTE;
  noteEl.focus();
});

document.getElementById('reset-btn').addEventListener('click', () => {
  setTimeout(() => resetGauge(), 0);
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = noteEl.value.trim();
  if (!text) return;

  const payload = {
    text,
    age: parseInt(document.getElementById('age').value, 10) || null,
    gender: document.getElementById('gender').value || null,
    insurance: document.getElementById('insurance').value || null,
    los_days: parseFloat(document.getElementById('los').value) || null,
  };

  setLoading(true);
  try {
    const res = await api.predict(payload);
    renderResult(res);
  } catch (err) {
    renderError(err.message);
  } finally {
    setLoading(false);
  }
});

// ─────────────────────────────────────────────
// Gauge rendering
// ─────────────────────────────────────────────

const CIRCUMFERENCE = 2 * Math.PI * 84; // ~527.8

function setLoading(loading) {
  btn.disabled = loading;
  btn.innerHTML = loading
    ? `<span class="spinner"></span><span>Scoring…</span>`
    : `<i data-lucide="zap"></i><span>Predict Risk</span>`;
  if (window.lucide) lucide.createIcons();
}

function resetGauge() {
  document.getElementById('risk-value').textContent = '—';
  document.getElementById('risk-circle-fg').setAttribute('stroke-dashoffset', CIRCUMFERENCE);
  document.getElementById('risk-circle').className = 'risk-circle';
  const badge = document.getElementById('risk-badge');
  badge.className = 'risk-badge';
  badge.innerHTML = `<i data-lucide="circle-help"></i><span>Awaiting input</span>`;
  document.getElementById('risk-meta').textContent = 'Submit a note to see the model\'s prediction.';
  document.getElementById('risk-details').style.display = 'none';
  if (window.lucide) lucide.createIcons();
}

function renderResult(res) {
  const prob = res.probability;
  const pct = (prob * 100).toFixed(1);

  // Animate circle
  const offset = CIRCUMFERENCE * (1 - prob);
  const circle = document.getElementById('risk-circle');
  const fg = document.getElementById('risk-circle-fg');
  circle.className = `risk-circle risk-circle--${res.risk_level.toLowerCase()}`;
  fg.setAttribute('stroke-dashoffset', offset);

  document.getElementById('risk-value').textContent = `${pct}%`;

  // Badge
  const badge = document.getElementById('risk-badge');
  const level = res.risk_level.toLowerCase();
  const iconMap = { low: 'check-circle', moderate: 'alert-triangle', high: 'alert-octagon' };
  badge.className = `risk-badge risk-badge--${level}`;
  badge.innerHTML = `<i data-lucide="${iconMap[level] || 'info'}"></i><span>${res.risk_level} risk</span>`;

  // Meta
  document.getElementById('risk-meta').textContent =
    `Predicted by ${res.model_name} on ${res.feature_type} features.`;

  // Details
  document.getElementById('risk-details').style.display = 'block';
  document.getElementById('detail-model').textContent = res.model_name;
  document.getElementById('detail-features').textContent = res.feature_type;
  document.getElementById('detail-threshold').textContent = res.threshold.toFixed(2);
  document.getElementById('detail-label').textContent =
    res.predicted_label === 1 ? 'Will be readmitted' : 'No readmission';

  if (window.lucide) lucide.createIcons();
}

function renderError(msg) {
  resetGauge();
  const badge = document.getElementById('risk-badge');
  badge.className = 'risk-badge risk-badge--high';
  badge.innerHTML = `<i data-lucide="alert-circle"></i><span>Error</span>`;
  document.getElementById('risk-meta').textContent = msg || 'Prediction failed.';
  if (window.lucide) lucide.createIcons();
}
