'use strict';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { status: null, results: [], filter: 'all', audit: { rows: [] }, visibleReviews: 10, job: null, toast: null };
const format = (n) => Number(n).toLocaleString('en-IN');
const percent = (n) => `${(100 * n).toFixed(1)}%`;

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
function toast(message) {
  clearTimeout(state.toast);
  $('#toast').textContent = message;
  $('#toast').classList.remove('hidden');
  state.toast = setTimeout(() => $('#toast').classList.add('hidden'), 3500);
}

function navigate(page) {
  $$('.page').forEach(el => el.classList.toggle('hidden', el.id !== `page-${page}`));
  $$('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.page === page));
  $('#breadcrumb').textContent = { analyze: 'COMMENT ANALYSIS', dataset: 'DATASET STUDIO', experiments: 'EXPERIMENTS', review: 'LABEL REVIEW' }[page];
  if (page === 'review') loadReviews().catch(fail);
}

function busy(active) {
  for (const id of ['fetch-dataset', 'import-dataset', 'train-models', 'audit-labels']) $( `#${id}`).disabled = active;
  $$('.review-actions button').forEach(button => { button.disabled = active; });
  $('#job-banner').classList.toggle('hidden', !active);
}

async function watchJob(id) {
  state.job = id;
  busy(true);
  try {
    const job = await api(`jobs/${id}`);
    $('#job-text').textContent = job.message;
    if (job.status === 'running') {
      setTimeout(() => watchJob(id), 1200);
      return;
    }
    state.job = null;
    busy(false);
    if (job.status === 'failed') throw new Error(job.message);
    await refresh();
    await loadReviews();
    toast({ dataset: 'Dataset loaded and checked.', training: 'Training complete. Model and report updated.', audit: 'Review queue is ready.' }[job.kind] || 'Complete.');
  } catch (error) {
    state.job = null;
    busy(false);
    fail(error);
  }
}

async function launch(path, payload = {}) {
  clearError(); busy(true);
  try {
    const job = await api(path, { method: 'POST', body: JSON.stringify(payload) });
    await watchJob(job.id);
  } catch (error) { busy(false); fail(error); }
}

async function refresh() {
  const data = await api('status');
  state.status = data;
  $('#connection-text').textContent = 'Local backend connected';
  $('#connection-dot').classList.remove('offline');
  $('#model-pill').textContent = data.model_ready ? `●  ${data.backend === 'transformer' ? 'Multilingual transformer' : 'Research baseline'} ready` : '○  Train a model to begin';
  $('#analyze-button').disabled = !data.model_ready;
  renderDataset(data.dataset);
  renderExperiments(data.report, data.model_stale);
  if (data.active_job && !state.job) watchJob(data.active_job.id);
}

function renderDataset(info) {
  const root = $('#dataset-details'); root.replaceChildren();
  if (!info) {
    $('#dataset-name').textContent = 'No dataset loaded';
    $('#dataset-state').textContent = 'Awaiting data';
    root.textContent = 'Load data here to retrain or audit labels. A packaged model can analyze comments without the training corpus.';
    return;
  }
  $('#dataset-name').textContent = info.name;
  $('#dataset-state').textContent = 'Published / imported labels · unverified';
  const grid = node('div', 'data-metrics');
  const counts = info.class_counts;
  for (const [value, label] of [[info.saved_rows, 'Rows after split checks'], [counts.non_toxic, 'Non-toxic labels'], [counts.toxic, 'Toxic labels'], [info.reviewed_count || 0, 'Training labels reviewed']]) {
    const item = node('div'); item.append(node('strong', '', format(value)), node('span', '', label)); grid.append(item);
  }
  const bar = node('div', 'class-distribution'); const fill = node('span');
  fill.style.width = `${100 * counts.non_toxic / (counts.non_toxic + counts.toxic)}%`; bar.append(fill);
  const note = node('p', 'muted small', `${format(info.corrected_count || 0)} labels corrected. ${format(info.excluded?.other_language || 0)} other-language rows excluded; ${format(info.excluded?.duplicate_rows || 0)} duplicates removed; ${format(info.excluded?.conflicting_duplicate_rows || 0)} conflicting rows excluded.`);
  root.append(grid, bar, note);
  if (info.split?.overlap_rows_excluded) root.append(node('p', 'muted small', `${info.split.overlap_rows_excluded} lower-priority rows excluded to keep variant groups out of multiple splits.`));
}

function renderExperiments(report, stale) {
  if (!report) return;
  const root = $('#experiment-table'); root.replaceChildren();
  const table = node('table'); const head = node('thead'); const headers = node('tr');
  for (const title of ['Model', 'Val. macro-F1', 'Test macro-F1', 'Precision', 'Recall', 'False positive rate']) headers.append(node('th', '', title));
  head.append(headers); table.append(head);
  const body = node('tbody');
  report.experiments.forEach((experiment, index) => {
    const row = node('tr', index === report.selected_index ? 'selected-row' : '');
    const name = node('td', '', experiment.name);
    if (index === report.selected_index) name.append(node('span', 'count-pill', 'Selected'));
    row.append(name);
    for (const value of [experiment.validation.macro_f1, experiment.test.macro_f1, experiment.test.precision, experiment.test.recall, experiment.test.false_positive_rate]) row.append(node('td', '', percent(value)));
    body.append(row);
  });
  table.append(body); root.append(table);
  $('#experiment-meta').textContent = `${new Date(report.created_at).toLocaleString()} · ${format(report.split.counts.test.rows)} test comments${stale ? ' · Dataset changed: retrain to update this model' : ''}`;
  const splits = $('#split-details'); splits.replaceChildren(node('p', 'muted small', report.split.strategy));
  for (const [key, value] of Object.entries(report.split.counts)) {
    const row = node('div', 'split-row'); row.append(node('span', '', key[0].toUpperCase() + key.slice(1)), node('strong', '', `${format(value.rows)} comments`)); splits.append(row);
  }
  const robustness = $('#robustness-details'); robustness.replaceChildren();
  const selected = report.experiments[report.selected_index];
  robustness.append(node('p', 'muted small', Object.keys(selected.robustness).length ? 'Selected model · prediction consistency after synthetic changes' : 'No synthetic spelling evaluation in this report. Run the multilingual diagnostics and inspect the downloaded report for language and script metrics.'));
  for (const [key, value] of Object.entries(selected.robustness)) {
    const row = node('div', 'split-row'); row.append(node('span', '', key === 'repeat' ? 'Repeated letters' : 'Inserted punctuation'), node('strong', '', value ? `${percent(value.prediction_consistency)} · ${format(value.count)} examples` : 'No applicable examples')); robustness.append(row);
  }
  robustness.append(node('p', 'muted small', 'Consistency measures whether predictions stay the same. It does not show that either prediction is correct.'));
}

function getComments() { return $('#comments').value.split(/\r?\n/).map(s => s.trim()).filter(Boolean); }
function countComments() { $('#input-count').textContent = `${getComments().length} comments`; }

async function analyze() {
  clearError(); const comments = getComments();
  if (!comments.length) return fail(new Error('Add at least one comment.'));
  if (comments.length > 500) return fail(new Error('Please analyze up to 500 comments per batch.'));
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
  finally { button.disabled = !state.status?.model_ready; button.replaceChildren(document.createTextNode('Analyze comments '), node('span', '', '↗')); }
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

async function loadReviews() {
  state.audit = await api('reviews'); renderReviews();
}

function renderReviews() {
  const root = $('#review-list'); root.replaceChildren();
  const rows = state.audit.rows || [];
  $('#queue-badge').textContent = rows.length;
  $('#review-count').textContent = rows.length;
  $('#review-pagination').classList.toggle('hidden', rows.length <= state.visibleReviews);
  if (!rows.length) { root.append(node('div', 'empty-small', state.audit.method ? 'Your queue is empty. Retrain after corrections, or run another audit.' : 'Load your dataset and run a label audit to start reviewing.')); return; }
  $('#review-meta').textContent = `${format(state.audit.flagged_count)} potentially mislabeled training comments · showing up to 100 for review`;
  for (const item of rows.slice(0, state.visibleReviews)) {
    const row = node('article', 'review-row'); const content = node('div');
    const current = item.label === 1 ? 'toxic' : 'non-toxic';
    content.append(node('span', `label${item.label === 1 ? ' flagged' : ''}`, `Current label: ${current}`), node('p', 'comment-text', item.comment), node('div', 'review-note', `Suggested: ${item.suggested_label ? 'toxic' : 'non-toxic'} · Out-of-fold toxicity probability ${percent(item.probability)} · Original label: ${item.original_label}`));
    const actions = node('div', 'review-actions');
    for (const [label, value] of [[`Keep ${current}`, item.label], [`Change to ${item.label ? 'non-toxic' : 'toxic'}`, 1 - item.label]]) {
      const button = node('button', 'secondary', label); button.disabled = !!state.job;
      button.addEventListener('click', async () => {
        clearError(); actions.querySelectorAll('button').forEach(el => { el.disabled = true; });
        try {
          await api('reviews', { method: 'POST', body: JSON.stringify({ revision: state.audit.revision, decisions: [{ id: item.id, label: value }] }) });
          await loadReviews(); await refresh(); toast('Review saved. Retrain when your review batch is complete.');
        } catch (error) { fail(error); await loadReviews(); }
      }); actions.append(button);
    }
    row.append(content, actions); root.append(row);
  }
}

function exportResults() {
  const safe = (value) => { let s = String(value); if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`; return `"${s.replaceAll('"', '""')}"`; };
  const rows = [['comment', 'predicted_label', 'toxicity_probability', 'likelihood_score_0_10', 'needs_context', 'truncated'], ...state.results.map(r => [r.comment, r.label, r.probability, r.score, r.uncertain, !!r.truncated])];
  const blob = new Blob(['\ufeff' + rows.map(r => r.map(safe).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'mozhi_predictions.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

$$('.nav-item').forEach(el => el.addEventListener('click', () => navigate(el.dataset.page)));
$$('[data-filter]').forEach(el => el.addEventListener('click', () => { state.filter = el.dataset.filter; $$('[data-filter]').forEach(button => button.classList.toggle('active', button === el)); renderResults(); }));
$('#comments').addEventListener('input', countComments);
$('#clear-comments').addEventListener('click', () => { $('#comments').value = ''; countComments(); });
$('#load-examples').addEventListener('click', () => {
  $('#comments').value = ['Indha video romba helpful ah irukku. Nandri!', 'The explanation could have been better.', 'You are an idiot. Stop posting this garbage.', 'Super acting thalaiva, semma performance!', 'I disagree with your opinion, but thanks for explaining.'].join('\n'); countComments();
});
$('#analyze-button').addEventListener('click', analyze);
$('#export-results').addEventListener('click', exportResults);
$('#fetch-dataset').addEventListener('click', () => launch('datasets/research'));
$('#train-models').addEventListener('click', () => launch('train', { augment: $('#augment').checked }));
$('#audit-labels').addEventListener('click', () => launch('audit'));
$('#review-more').addEventListener('click', () => { state.visibleReviews += 10; renderReviews(); });
$('#dataset-file').addEventListener('change', () => { $('#file-name').textContent = $('#dataset-file').files[0]?.name || 'Choose a labeled dataset'; });
$('#import-dataset').addEventListener('click', async () => {
  clearError();
  try {
    const file = $('#dataset-file').files[0];
    if (!file) throw new Error('Choose a labeled CSV or TSV first.');
    if (file.size > 20_000_000) throw new Error('File is too large. Use a dataset under 20 MB.');
    const mappingText = $('#label-mapping').value.trim();
    const mapping = mappingText ? JSON.parse(mappingText) : null;
    if (mapping && (Array.isArray(mapping) || typeof mapping !== 'object')) throw new Error('Label mapping must be a JSON object.');
    await launch('datasets/import', { filename: file.name, content: await file.text(), text_column: $('#text-column').value.trim(), label_column: $('#label-column').value.trim(), mapping });
  } catch (error) { fail(error); }
});

Promise.all([refresh(), loadReviews()]).catch(error => {
  $('#connection-text').textContent = 'Backend unavailable'; $('#connection-dot').classList.add('offline'); fail(error);
});
