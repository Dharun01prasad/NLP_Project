'use strict';
const $ = id => document.getElementById(id);
let predictions = [], tabId, enabled = false, modelReady = false;
function element(tag, className, text) { const item = document.createElement(tag); item.className = className; item.textContent = text; return item; }
function message(text, error = false) { $('message').textContent = text; $('message').className = `message${error ? ' error' : ''}`; }
function watchVideo(url) {
  try { const parsed = new URL(url); return parsed.protocol === 'https:' && ['www.youtube.com', 'youtube.com', 'm.youtube.com'].includes(parsed.hostname) && parsed.pathname === '/watch' && !!parsed.searchParams.get('v')?.trim(); }
  catch { return false; }
}
async function background(type, comments) {
  const response = await chrome.runtime.sendMessage({ type, comments });
  if (!response?.ok) throw new Error(response?.error || 'Could not reach the extension background worker.');
  return response.data;
}
function toggleDisplay() {
  $('enabled').checked = enabled;
  $('toggle-status').textContent = enabled ? (tabId ? 'ON · analyzing rendered comments' : 'ON · open a watch video') : 'OFF · no scanning';
  $('toggle-status').classList.toggle('on', enabled);
  $('analyze').disabled = !enabled || !tabId || !modelReady;
  if (!enabled) { $('backend').textContent = 'OFF'; $('analysis').classList.add('hidden'); predictions = []; }
}
async function attach(tab) {
  if (!tab || !watchVideo(tab.url)) {
    tabId = undefined; modelReady = false;
    $('backend').textContent = 'Open a YouTube watch page';
    message('Mozhi runs on YouTube watch pages only.'); toggleDisplay(); return;
  }
  tabId = tab.id;
  await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
  await chrome.tabs.sendMessage(tabId, { type: 'MOZHI_ENABLE' });
  const status = await background('MOZHI_STATUS');
  modelReady = !!status.model_ready;
  $('backend').textContent = modelReady ? '● Local model connected' : '○ No trained model';
  if (!modelReady) message('No trained model is available. Start the backend with a trained model.', true);
  else message('Mozhi is ON. Newly rendered comments will be analyzed.');
  toggleDisplay();
}
async function initialize() {
  try {
    enabled = (await chrome.storage.local.get('mozhiEnabled')).mozhiEnabled === true;
    toggleDisplay();
    if (!enabled) return;
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await attach(tab);
  } catch (error) { $('backend').textContent = 'Backend unavailable'; message(error.message, true); }
}
$('enabled').addEventListener('change', async event => {
  const next = event.target.checked;
  $('enabled').disabled = true;
  try {
    await chrome.storage.local.set({ mozhiEnabled: next });
    enabled = next; toggleDisplay();
    if (!next) {
      modelReady = false;
      if (tabId) { try { await chrome.tabs.sendMessage(tabId, { type: 'MOZHI_DISABLE' }); } catch { /* Page no longer exists. */ } }
      message('Mozhi is OFF. Page indicators have been removed.');
    } else {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      await attach(tab);
    }
  } catch (error) { $('backend').textContent = 'Backend unavailable'; message(error.message, true); }
  finally { $('enabled').disabled = false; toggleDisplay(); }
});
function render(data) {
  $('analysis').classList.remove('hidden');
  $('count').textContent = data.summary.analyzed;
  $('flagged').textContent = `${data.summary.flagged_percent}%`;
  $('average').textContent = data.summary.mean_likelihood_score.toFixed(1);
  $('comments').replaceChildren(); predictions = data.results;
  for (const result of data.results) {
    const row = element('article', 'comment', ''), top = element('div', 'comment-top', '');
    top.append(element('span', `label${result.label === 'toxic' ? ' flagged' : ''}`, result.label === 'toxic' ? 'Flagged' : 'Not flagged'), element('span', 'score', `${result.score.toFixed(1)} / 10`));
    row.append(top, element('p', '', result.comment));
    if (result.uncertain) row.append(element('span', 'context', 'Needs context · near model threshold'));
    const meter = element('div', 'meter', ''), fill = element('span', '', '');
    fill.style.width = `${result.score * 10}%`; meter.append(fill); row.append(meter);
    $('comments').append(row);
  }
}
$('analyze').addEventListener('click', async () => {
  $('analyze').disabled = true; $('analysis').classList.add('hidden'); predictions = [];
  try {
    if (!enabled || !tabId) throw new Error('Turn Mozhi ON on a YouTube watch page first.');
    message('Reading comments currently loaded on this page…');
    const batch = await chrome.tabs.sendMessage(tabId, { type: 'MOZHI_COLLECT' });
    if (batch.error) throw new Error(batch.error);
    if (!batch.comments.length) throw new Error('No loaded comments found. Scroll to the comments, then try again.');
    $('video-title').textContent = batch.title;
    message(`Analyzing ${batch.comments.length} rendered comments locally…`);
    const results = [];
    for (let offset = 0; offset < batch.comments.length; offset += 100) {
      if (!enabled) throw new Error('Mozhi was switched OFF.');
      const data = await background('MOZHI_ANALYZE', batch.comments.slice(offset, offset + 100));
      if (data.results.length !== Math.min(100, batch.comments.length - offset)) throw new Error('Unexpected backend result count.');
      results.push(...data.results);
    }
    const current = await chrome.tabs.sendMessage(tabId, { type: 'MOZHI_VIDEO_ID' });
    if (current.videoId !== batch.videoId) throw new Error('The video changed during analysis. Analyze the new page again.');
    const flagged = results.filter(result => result.label === 'toxic').length;
    render({ results, summary: { analyzed: results.length, flagged_percent: +(100 * flagged / results.length).toFixed(1),
      mean_likelihood_score: results.reduce((sum, result) => sum + result.score, 0) / results.length } });
    message(`${results.length} comments analyzed${batch.truncated ? ` out of ${batch.loaded} loaded` : ''}. This is a sample of currently rendered comments.`);
  } catch (error) { message(error.message, true); }
  finally { $('analyze').disabled = !enabled || !modelReady; }
});
$('open-lab').addEventListener('click', () => chrome.tabs.create({ url: 'http://127.0.0.1:8000' }));
$('export').addEventListener('click', () => {
  if (!predictions.length) return;
  const quote = value => { let text = String(value); if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`; return `"${text.replaceAll('"', '""')}"`; };
  const rows = [['comment', 'predicted_label', 'probability', 'likelihood_0_10', 'needs_context'], ...predictions.map(result => [result.comment, result.label, result.probability, result.score, result.uncertain])];
  const blob = new Blob(['\ufeff' + rows.map(row => row.map(quote).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'mozhi_youtube_predictions.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
initialize();
