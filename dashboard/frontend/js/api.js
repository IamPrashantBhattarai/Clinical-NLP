// Tiny fetch wrapper for the dashboard API.
const API_BASE = '/api';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const opts = {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  };
  if (opts.body && typeof opts.body !== 'string') {
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data.detail) detail = data.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

const api = {
  health: () => request('/health'),
  models: () => request('/models'),
  predict: (payload) => request('/predict', { method: 'POST', body: payload }),
  explain: (payload) => request('/explain', { method: 'POST', body: payload }),
  results: () => request('/results'),
  fairness: () => request('/fairness'),
  topics: () => request('/topics'),
  figures: () => request('/figures'),
  figureUrl: (filename) => `${API_BASE}/figures/${filename}`,
};

window.api = api;
