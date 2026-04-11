// Renders the sidebar nav and highlights the active page.
const NAV_ITEMS = [
  { section: 'Dashboard', items: [
    { href: 'index.html',     label: 'Overview',          icon: 'layout-dashboard' },
    { href: 'predict.html',   label: 'Predict',           icon: 'activity' },
    { href: 'explain.html',   label: 'Explain (SHAP)',    icon: 'sparkles' },
  ]},
  { section: 'Analysis', items: [
    { href: 'compare.html',   label: 'Model Comparison',  icon: 'bar-chart-3' },
    { href: 'fairness.html',  label: 'Fairness Audit',    icon: 'scale' },
    { href: 'topics.html',    label: 'Topics',            icon: 'tags' },
  ]},
];

function renderSidebar(activePage) {
  const sections = NAV_ITEMS.map(section => {
    const items = section.items.map(item => {
      const isActive = item.href === activePage;
      return `
        <a href="${item.href}" class="nav-item ${isActive ? 'active' : ''}">
          <i data-lucide="${item.icon}"></i>
          <span>${item.label}</span>
        </a>
      `;
    }).join('');
    return `
      <div class="sidebar-section">
        <div class="sidebar-section-title">${section.section}</div>
        <nav class="nav-list">${items}</nav>
      </div>
    `;
  }).join('');

  return `
    <aside class="sidebar">
      <div class="sidebar-brand">
        <div class="sidebar-brand-icon">CN</div>
        <div class="sidebar-brand-text">
          <span class="sidebar-brand-name">Clinical NLP</span>
          <span class="sidebar-brand-tag">Readmission Predictor</span>
        </div>
      </div>
      ${sections}
      <div class="sidebar-footer">
        <span class="status-dot"></span>
        <span id="api-status">API ready</span>
      </div>
    </aside>
  `;
}

function mountShell({ activePage, title, subtitle, actions = '' }) {
  document.body.innerHTML = `
    <div class="app">
      ${renderSidebar(activePage)}
      <main class="main">
        <header class="page-header">
          <div>
            <div class="page-header-title">${title}</div>
            ${subtitle ? `<div class="page-header-subtitle">${subtitle}</div>` : ''}
          </div>
          <div class="page-header-actions">${actions}</div>
        </header>
        <div class="page-content" id="page-content"></div>
      </main>
    </div>
  `;
  if (window.lucide) lucide.createIcons();

  // Health check
  api.health()
    .then(h => {
      const el = document.getElementById('api-status');
      if (el) el.textContent = `API ready · ${h.models_loaded} model${h.models_loaded === 1 ? '' : 's'}`;
    })
    .catch(() => {
      const el = document.getElementById('api-status');
      if (el) el.textContent = 'API offline';
      const dot = document.querySelector('.status-dot');
      if (dot) dot.style.background = 'var(--color-danger)';
    });
}

window.mountShell = mountShell;
