'use strict';
const BACKEND = 'http://127.0.0.1:8000';
const activeRequests = new Set();

function watchVideo(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' && ['youtube.com', 'www.youtube.com', 'm.youtube.com'].includes(parsed.hostname) &&
      parsed.pathname === '/watch' && !!parsed.searchParams.get('v')?.trim();
  } catch { return false; }
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && changes.mozhiEnabled?.newValue !== true) {
    for (const controller of activeRequests) controller.abort();
  }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete' || !watchVideo(tab.url)) return;
  chrome.storage.local.get('mozhiEnabled').then(async value => {
    if (value.mozhiEnabled !== true) return;
    try {
      await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
      await chrome.tabs.sendMessage(tabId, { type: 'MOZHI_ENABLE' });
    } catch { /* activeTab grants can end after navigation; opening the popup grants access again. */ }
  });
});

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id !== chrome.runtime.id || !['MOZHI_STATUS', 'MOZHI_ANALYZE'].includes(message?.type)) return false;
  (async () => {
    let controller, timeout;
    try {
      if ((await chrome.storage.local.get('mozhiEnabled')).mozhiEnabled !== true) throw new Error('Turn Mozhi ON first.');
      const path = message.type === 'MOZHI_STATUS' ? '/api/status' : '/api/analyze';
      const options = {};
      if (message.type === 'MOZHI_ANALYZE') {
        const comments = message.comments;
        if (!Array.isArray(comments) || !comments.length || comments.length > 100 ||
            comments.some(text => typeof text !== 'string' || !text.trim() || text.length > 10000) ||
            comments.reduce((sum, text) => sum + text.length, 0) > 1_000_000) throw new Error('Invalid comment batch.');
        Object.assign(options, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ comments }) });
      }
      controller = new AbortController(); activeRequests.add(controller);
      timeout = setTimeout(() => controller.abort(), 35000);
      options.signal = controller.signal;
      const response = await fetch(BACKEND + path, options);
      const data = await response.json();
      if ((await chrome.storage.local.get('mozhiEnabled')).mozhiEnabled !== true) throw new Error('Mozhi was switched OFF.');
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The local backend rejected this request.');
      respond({ ok: true, data });
    } catch (error) {
      const description = error.name === 'TypeError' ? 'Local backend unavailable. Start it with python run.py.' :
        error.name === 'AbortError' ? 'Analysis stopped or timed out.' : error.message;
      respond({ ok: false, error: description });
    } finally {
      clearTimeout(timeout);
      if (controller) activeRequests.delete(controller);
    }
  })();
  return true;
});
