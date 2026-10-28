'use strict';
const BACKEND = 'http://127.0.0.1:8000';

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id !== chrome.runtime.id || !['MOZHI_STATUS', 'MOZHI_ANALYZE'].includes(message?.type)) return false;
  (async () => {
    try {
      let options = { signal: AbortSignal.timeout(35000) };
      const path = message.type === 'MOZHI_STATUS' ? '/api/status' : '/api/analyze';
      if (message.type === 'MOZHI_ANALYZE') {
        const comments = message.comments;
        if (!Array.isArray(comments) || !comments.length || comments.length > 500 || comments.some(s => typeof s !== 'string' || !s.trim() || s.length > 10000)) throw new Error('Invalid comment batch.');
        options = { ...options, method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ comments }) };
      }
      const response = await fetch(BACKEND + path, options);
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The local backend rejected this request.');
      respond({ ok: true, data });
    } catch (error) {
      const message = error.name === 'TypeError' ? 'Start the local backend with python run.py, then try again.' : error.name === 'TimeoutError' ? 'Analysis timed out. Try fewer loaded comments.' : error.message;
      respond({ ok: false, error: message });
    }
  })();
  return true;
});
