'use strict';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { status: null, results: [], filter: 'all', analyzing: false };
const format = (n) => Number(n).toLocaleString('en-IN');

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

async function api(path, options = {}) {
  const response = await fetch(`/api/${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(x => x.msg).join('; ') : 'Request failed');
  }
  return data;
}

function fail(error) {
  $('#error-banner').textContent = error.message || String(error);
  $('#error-banner').classList.remove('hidden');
}

function clearError() { $('#error-banner').classList.add('hidden'); }
async function refresh() {
  const data = await api('status');
  state.status = data;
  $('#connection-text').textContent = 'Local backend connected';
  $('#connection-dot').classList.remove('offline');
  $('#model-pill').textContent = data.model_ready
    ? `${data.backend === 'transformer' ? 'Multilingual transformer' : 'Research baseline'} ready`
    : 'Train a model to begin';
  $('#analyze-button').disabled = !data.model_ready;

}

function getComments() { return $('#comments').value.split(/\r?\n/).map(s => s.trim()).filter(Boolean); }
function countComments() { $('#input-count').textContent = `${getComments().length} comments`; }

async function analyze() {
  if (state.analyzing || !state.status?.model_ready) return;
  clearError(); const comments = getComments();
  if (!comments.length) return fail(new Error('Add at least one comment.'));
  if (comments.length > 500) return fail(new Error('Please analyze up to 500 comments per batch.'));
  state.analyzing = true;
  $('#results').setAttribute('aria-busy', 'true');
  const button = $('#analyze-button'); button.disabled = true; button.textContent = 'Analyzing…';
  try {
    const data = await api('analyze', { method: 'POST', body: JSON.stringify({ comments }) });
    state.results = data.results;
    $('#stat-count').textContent = format(data.summary.analyzed);
    $('#stat-flagged').textContent = `${data.summary.flagged_percent}%`;
    $('#stat-flagged-note').textContent = `${format(data.summary.flagged)} comments above the threshold`;
    $('#stat-average').replaceChildren(document.createTextNode(data.summary.mean_likelihood_score.toFixed(1)), node('small', '', ' / 10'));
    $('#stat-uncertain').textContent = format(data.summary.uncertain);
    $('#results-count').textContent = format(data.results.length);
    $('#threshold-note').textContent = `Threshold ${data.threshold.toFixed(2)}`;
    $('#export-results').disabled = false;
    renderResults();
  } catch (error) { fail(error); }
  finally { state.analyzing = false; $('#results').setAttribute('aria-busy', 'false'); button.disabled = !state.status?.model_ready; button.replaceChildren(document.createTextNode('Analyze comments '), node('span', '', '↗')); }
}

function renderResults() {
  const root = $('#results'); root.replaceChildren();
  const filtered = state.results.filter(r => state.filter === 'all' || state.filter === 'toxic' && r.label === 'toxic' || state.filter === 'uncertain' && r.uncertain);
  if (!filtered.length) { root.append(node('div', 'empty-small', state.results.length ? 'No comments match this filter.' : 'Analyze some comments to see results.')); return; }
  const list = node('div', 'result-list');
  for (const result of filtered) {
    const row = node('article', 'result-row');
    const top = node('div', 'result-top'); const labels = node('div');
    labels.append(node('span', `label${result.label === 'toxic' ? ' flagged' : ''}`, result.label === 'toxic' ? 'Flagged' : 'Not flagged'));
    if (result.uncertain) labels.append(node('span', 'label uncertain', 'Needs context'));
    const scoreBox = node('div'); const score = node('span', 'score', result.score.toFixed(1)); score.append(node('small', '', ' / 10'));
    const bar = node('div', `meter${result.label === 'toxic' ? ' flagged-meter' : ''}`); const fill = node('span'); fill.style.width = `${result.score * 10}%`; bar.append(fill);
    scoreBox.append(score, bar); top.append(labels, scoreBox); row.append(top, node('p', 'comment-text', result.comment));
    if (result.truncated) row.append(node('p', 'muted small', 'Only the beginning of this comment was analyzed. Review the full text.'));
    if (result.evidence?.length) {
      const details = node('details', 'evidence'); details.append(node('summary', '', 'Associated model features'));
      details.append(node('p', '', 'Positive linear feature contributions; these are learned associations, not verified abusive phrases.'));
      result.evidence.forEach(item => { const chip = node('span', '', `${item.type}: ${item.feature}`); details.append(chip); }); row.append(details);
    }
    list.append(row);
  }
  root.append(list);
}

function exportResults() {
  const safe = (value) => { let s = String(value); if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`; return `"${s.replaceAll('"', '""')}"`; };
  const rows = [['comment', 'predicted_label', 'toxicity_probability', 'likelihood_score_0_10', 'needs_context', 'truncated'], ...state.results.map(r => [r.comment, r.label, r.probability, r.score, r.uncertain, !!r.truncated])];
  const blob = new Blob(['\ufeff' + rows.map(r => r.map(safe).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'mozhi_predictions.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

$$('[data-filter]').forEach(el => el.addEventListener('click', () => { state.filter = el.dataset.filter; $$('[data-filter]').forEach(button => button.classList.toggle('active', button === el)); renderResults(); }));
$('#comments').addEventListener('input', countComments);
$('#comments').addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.isComposing) {
    event.preventDefault();
    if (!event.repeat) analyze();
  }
});
$('#clear-comments').addEventListener('click', () => { $('#comments').value = ''; countComments(); $('#comments').focus(); });
$('#load-examples').addEventListener('click', () => {
  $('#comments').value = ['Indha video romba helpful ah irukku. Nandri!', 'The explanation could have been better.', 'You are an idiot. Stop posting this garbage.', 'Super acting thalaiva, semma performance!', 'I disagree with your opinion, but thanks for explaining.'].join('\n'); countComments();
});
$('#analyze-button').addEventListener('click', analyze);
$('#export-results').addEventListener('click', exportResults);
refresh().catch(error => {
  $('#connection-text').textContent = 'Backend unavailable'; $('#connection-dot').classList.add('offline'); fail(error);
});
