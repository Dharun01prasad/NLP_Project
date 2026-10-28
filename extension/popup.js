'use strict';
const $ = (id) => document.getElementById(id);
let predictions = [];
let tabId;
function element(tag, className, text) { const e = document.createElement(tag); e.className = className; e.textContent = text; return e; }
function message(text, error = false) { $('message').textContent = text; $('message').className = `message${error ? ' error' : ''}`; }
async function background(type, comments) {
  const response = await chrome.runtime.sendMessage({ type, comments });
  if (!response?.ok) throw new Error(response?.error || 'Could not reach the extension background worker.');
  return response.data;
}
function watchVideo(url) {
  try { const parsed = new URL(url); return ['www.youtube.com', 'youtube.com', 'm.youtube.com'].includes(parsed.hostname) && parsed.pathname === '/watch' && parsed.searchParams.has('v'); } catch { return false; }
}
async function initialize() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !watchVideo(tab.url)) { $('backend').textContent = 'Open a YouTube watch page'; return; }
    tabId = tab.id;
    await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
    const data = await background('MOZHI_STATUS');
    $('backend').textContent = data.model_ready ? '● Local model connected' : '○ Train a model in the lab';
    $('analyze').disabled = !data.model_ready;
    if (!data.model_ready) message('Open the lab, load a labeled dataset, and train a model.');
  } catch (error) { $('backend').textContent = 'Backend unavailable'; message(error.message, true); }
}
function render(data) {
  $('analysis').classList.remove('hidden');
  $('count').textContent = data.summary.analyzed;
  $('flagged').textContent = `${data.summary.flagged_percent}%`;
  $('average').textContent = data.summary.mean_likelihood_score.toFixed(1);
  $('comments').replaceChildren();
  predictions = data.results;
  for (const result of data.results) {
    const row = element('article', 'comment', ''); const top = element('div', 'comment-top', '');
    top.append(element('span', `label${result.label === 'toxic' ? ' flagged' : ''}`, result.label === 'toxic' ? 'Flagged' : 'Not flagged'), element('span', 'score', `${result.score.toFixed(1)} / 10`));
    row.append(top, element('p', '', result.comment));
    if (result.uncertain) row.append(element('span', 'context', 'Needs context · near model threshold'));
    const meter = element('div', 'meter', ''); const fill = element('span', '', ''); fill.style.width = `${result.score * 10}%`; meter.append(fill); row.append(meter);
    $('comments').append(row);
  }
}
$('analyze').addEventListener('click', async () => {
  $('analyze').disabled = true; $('analysis').classList.add('hidden'); predictions = [];
  try {
    message('Reading comments currently loaded on this page…');
    const batch = await chrome.tabs.sendMessage(tabId, { type: 'MOZHI_COLLECT' });
    if (batch.error) throw new Error(batch.error);
    if (!batch.comments.length) throw new Error('No loaded comments found. Scroll to the comment section, then try again. YouTube markup changes may also require updating the selectors in content.js.');
    $('video-title').textContent = batch.title;
    message(`Analyzing ${batch.comments.length} comments locally…`);
    const data = await background('MOZHI_ANALYZE', batch.comments);
    const current = await chrome.tabs.sendMessage(tabId, { type: 'MOZHI_VIDEO_ID' });
    if (current.videoId !== batch.videoId) throw new Error('The video changed during analysis. Analyze the new page again.');
    render(data);
    message(`${data.summary.analyzed} comments analyzed${batch.truncated ? ` out of ${batch.loaded} loaded` : ''}. This is a sample, not the entire comment section.`);
  } catch (error) { message(error.message, true); }
  finally { $('analyze').disabled = false; }
});
$('open-lab').addEventListener('click', () => chrome.tabs.create({ url: 'http://127.0.0.1:8000' }));
$('export').addEventListener('click', () => {
  if (!predictions.length) return;
  const quote = value => { let s = String(value); if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`; return `"${s.replaceAll('"', '""')}"`; };
  const rows = [['comment', 'predicted_label', 'probability', 'likelihood_0_10', 'needs_context'], ...predictions.map(r => [r.comment, r.label, r.probability, r.score, r.uncertain])];
  const blob = new Blob(['\ufeff' + rows.map(row => row.map(quote).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'mozhi_youtube_predictions.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
initialize();
