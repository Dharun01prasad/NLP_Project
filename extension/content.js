(() => {
  'use strict';
  if (globalThis.__mozhiInstalled) return;
  globalThis.__mozhiInstalled = true;
  const SELECTOR = 'ytd-comment-thread-renderer #content-text, ytd-comment-view-model #content-text, ytd-comment-renderer #content-text';
  const DEBOUNCE = 1400, MAX_BATCH = 100, DEBUG = false;
  const state = { enabled: false, video: null, generation: 0, records: new Map(), pending: new Set(), stale: new Map(),
    observer: null, timer: null, busy: false, panel: null, error: '', serial: 0, storageSeen: false };
  const debug = error => { if (DEBUG) console.warn('Mozhi:', error); };
  const textOf = element => (element.innerText || element.textContent || '').trim();
  const comments = () => [...document.querySelectorAll(SELECTOR)];
  function videoId() {
    try {
      const url = new URL(location.href), id = url.searchParams.get('v');
      return url.protocol === 'https:' && ['youtube.com', 'www.youtube.com', 'm.youtube.com'].includes(url.hostname) &&
        url.pathname === '/watch' && id?.trim() ? id.trim() : null;
    } catch { return null; }
  }
  function hash(text) {
    let value = 2166136261;
    for (let i = 0; i < text.length; i++) value = Math.imul(value ^ text.charCodeAt(i), 16777619);
    return (value >>> 0).toString(16);
  }
  const isDark = () => document.documentElement.hasAttribute('dark');
  function shadow(host) {
    const root = host.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = `:host{all:initial;display:inline-block;font:12px/1.4 Arial,sans-serif;color:#263d36;margin:3px 0 5px}
      :host(.dark){color:#edf5ef}*{box-sizing:border-box}
      summary{cursor:pointer;list-style:none}summary::-webkit-details-marker{display:none}
      summary:focus-visible{outline:2px solid #78adca;outline-offset:2px}
      .badge{display:inline-flex;gap:5px;align-items:center;padding:3px 7px;border-radius:9px;border:1px solid currentColor;background:#f4faf4;color:#22734e;font:inherit;white-space:nowrap}
      :host(.dark) .badge{background:#19352c}.level-1{color:#668331;background:#f8f9db}.level-2{color:#9a7025;background:#fff3cf}
      .level-3{color:#ba6225;background:#fff0dd}.level-4{color:#a63832;background:#fce5e3}
      :host(.dark) .level-1,:host(.dark) .level-2,:host(.dark) .level-3,:host(.dark) .level-4{background:#342d2a}
      details{display:inline-block}.more{padding:5px 7px;border:1px solid #b9c9be;border-radius:5px;background:#fff;max-width:300px;white-space:normal}
      :host(.dark) .more{background:#24362f}.context{margin-left:5px;font-size:10px}
      .panel{position:fixed;right:16px;bottom:16px;z-index:2147483640;width:220px;max-width:calc(100vw - 32px);padding:11px 13px;border:1px solid #91b3a3;border-radius:9px;background:#fff;color:#263d36;box-shadow:0 4px 18px #0002;font:12px/1.5 Arial,sans-serif;pointer-events:none}
      :host(.dark) .panel{background:#20322a;color:#edf5ef}.panel strong,.panel span{display:block}.error{color:#a63832;margin-top:5px}.note{font-size:10px;opacity:.75;margin-top:5px}`;
    root.append(style);
    return root;
  }
  const theme = host => host.classList.toggle('dark', isDark());
  function emoji(score) {
    if (score < 2) return ['😊', 0];
    if (score < 4) return ['🙂', 1];
    if (score < 6) return ['😐', 2];
    if (score < 8) return ['😠', 3];
    return ['🤬', 4];
  }
  function totals() {
    let analyzed = 0, flagged = 0, scoreSum = 0;
    for (const record of state.records.values()) {
      if (!record.result || !record.element.isConnected) continue;
      analyzed++;
      flagged += Number(record.result.label === 'toxic');
      scoreSum += record.result.score;
    }
    return { analyzed, flagged, scoreSum };
  }
  function summary() {
    if (!state.enabled || !state.video || !document.body) return;
    if (!state.panel) {
      state.panel = document.createElement('span'); state.panel.className = 'mozhi-summary-host';
      const panel = document.createElement('div'); panel.className = 'panel';
      shadow(state.panel).append(panel); document.body.append(state.panel);
    }
    theme(state.panel);
    const { analyzed, flagged, scoreSum } = totals();
    const panel = state.panel.shadowRoot.querySelector('.panel'); panel.replaceChildren();
    const heading = document.createElement('strong'); heading.textContent = 'Mozhi · Comment insights';
    const stats = document.createElement('span');
    stats.textContent = `${analyzed} analyzed · ${flagged} flagged (${analyzed ? (100 * flagged / analyzed).toFixed(1) : '0.0'}%) · average ${analyzed ? (scoreSum / analyzed).toFixed(1) : '0.0'}/10`;
    const note = document.createElement('span'); note.className = 'note';
    note.textContent = 'Only comments currently rendered by YouTube. Scores are toxicity likelihood.';
    panel.append(heading, stats, note);
    if (state.error) { const error = document.createElement('span'); error.className = 'error'; error.textContent = state.error; panel.append(error); }
  }
  function indicator(record, result) {
    if (!record.element.isConnected || record.host) return;
    const host = document.createElement('span'); host.className = 'mozhi-indicator-host';
    const root = shadow(host), details = document.createElement('details'), badge = document.createElement('summary');
    const [icon, level] = emoji(result.score), percent = Math.round(result.probability * 100);
    const label = `Toxicity likelihood ${percent} percent, ${result.score.toFixed(1)} out of 10${result.uncertain ? ', needs context' : ''}`;
    badge.className = `badge level-${level}`; badge.setAttribute('aria-label', label); badge.title = label;
    badge.textContent = `${icon} ${percent}%`;
    if (result.uncertain) { const context = document.createElement('span'); context.className = 'context'; context.textContent = 'Needs context'; badge.append(context); }
    const more = document.createElement('div'); more.className = 'more';
    more.textContent = `Toxicity likelihood: ${percent}% · ${result.score.toFixed(1)}/10 · ${result.label === 'toxic' ? 'Flagged' : 'Not flagged'}. Model prediction, not verified abuse severity.`;
    details.append(badge, more); root.append(details); theme(host);
    record.element.insertAdjacentElement('afterend', host); record.host = host;
  }
  function removeRecord(record) { record.host?.remove(); record.host = null; state.pending.delete(record); }
  function clear() {
    state.generation++; clearTimeout(state.timer); state.timer = null;
    state.observer?.disconnect(); state.observer = null;
    for (const record of state.records.values()) removeRecord(record);
    state.records.clear(); state.pending.clear(); state.stale.clear(); state.panel?.remove(); state.panel = null;
    state.error = ''; state.busy = false; state.video = null;
  }
  function schedule() {
    if (!state.enabled || !state.video || !state.pending.size || state.error) return;
    clearTimeout(state.timer); state.timer = setTimeout(flush, DEBOUNCE);
  }
  function observe() {
    if (!state.enabled || !state.video || !document.body || state.observer) return;
    state.observer = new MutationObserver(changes => {
      if (!state.enabled) return;
      if (videoId() !== state.video || changes.some(change => [...change.addedNodes, ...change.removedNodes].some(node =>
        node.nodeType === 1 && !node.classList?.contains('mozhi-indicator-host') && !node.classList?.contains('mozhi-summary-host')))) scan();
    });
    state.observer.observe(document.body, { childList: true, subtree: true });
  }
  function scan() {
    if (!state.enabled) return;
    const current = videoId();
    if (current !== state.video) {
      const stale = current ? new Map([...state.records].map(([element, record]) => [element, record.text])) : new Map();
      clear(); if (!current) return;
      state.video = current; state.stale = stale; observe();
    }
    if (!current) return;
    for (const [element, record] of state.records) {
      const text = element.isConnected ? textOf(element) : '';
      if (!text || text !== record.text) { removeRecord(record); state.records.delete(element); }
      else if (record.host) theme(record.host);
    }
    let discovered = false;
    for (const element of comments()) {
      if (state.records.has(element)) continue;
      const text = textOf(element); if (!text || text.length > 10000) continue;
      if (state.stale.has(element)) {
        if (state.stale.get(element) === text) continue;
        state.stale.delete(element);
      }
      const record = { element, text, key: `${++state.serial}:${hash(text)}`, host: null, result: null };
      state.records.set(element, record); state.pending.add(record); discovered = true;
    }
    for (const element of state.stale.keys()) if (!element.isConnected) state.stale.delete(element);
    if (discovered) state.error = '';
    summary(); schedule();
  }
  async function flush() {
    state.timer = null;
    if (!state.enabled || !state.video || state.busy || !state.pending.size) return;
    const generation = state.generation, video = state.video, batch = [];
    let characters = 0; state.busy = true;
    for (const record of state.pending) {
      if (batch.length >= MAX_BATCH || characters + record.text.length > 1_000_000) break;
      state.pending.delete(record);
      if (!record.element.isConnected || state.records.get(record.element) !== record) continue;
      batch.push(record); characters += record.text.length;
    }
    if (!batch.length) { state.busy = false; scan(); return; }
    try {
      const response = await chrome.runtime.sendMessage({ type: 'MOZHI_ANALYZE', comments: batch.map(record => record.text) });
      if (!state.enabled || state.generation !== generation || state.video !== video) return;
      if (!response?.ok) throw new Error(response?.error || 'Local backend unavailable. Start the Mozhi backend.');
      if (!Array.isArray(response.data?.results) || response.data.results.length !== batch.length) throw new Error('Unexpected backend result count.');
      for (let i = 0; i < batch.length; i++) {
        const record = batch[i], result = response.data.results[i];
        if (!result || result.comment !== record.text || !['toxic', 'non_toxic'].includes(result.label) ||
            !Number.isFinite(result.probability) || !Number.isFinite(result.score) || typeof result.uncertain !== 'boolean')
          throw new Error('Backend results did not match the submitted comments.');
      }
      for (let i = 0; i < batch.length; i++) {
        const record = batch[i], result = response.data.results[i];
        if (state.records.get(record.element) !== record || !record.element.isConnected || textOf(record.element) !== record.text) continue;
        record.result = result; indicator(record, result);
      }
      state.error = '';
    } catch (error) {
      if (state.enabled && state.generation === generation) {
        state.error = error.message || 'Local backend unavailable.';
        batch.forEach(record => { if (!record.result && state.records.get(record.element) === record) state.pending.add(record); });
        debug(error);
      }
    } finally {
      if (state.generation === generation) { state.busy = false; summary(); schedule(); }
    }
  }
  function enable() { if (state.enabled) { state.error = ''; scan(); return; } state.enabled = true; scan(); }
  function disable() { state.enabled = false; clear(); }
  function collect() {
    if (!state.enabled) return { error: 'Turn Mozhi ON to analyze comments.' };
    const id = videoId(); if (!id) return { error: 'Open a YouTube video watch page first.' };
    const texts = comments().map(textOf).filter(text => text && text.length <= 10000);
    const eligible = texts.reduce((result, text) => {
      if (result.comments.length < 500 && result.characters + text.length <= 1_000_000) { result.comments.push(text); result.characters += text.length; }
      return result;
    }, { comments: [], characters: 0 });
    return { videoId: id, title: document.querySelector('h1.ytd-watch-metadata')?.innerText?.trim() || document.title,
      comments: eligible.comments, loaded: texts.length, truncated: eligible.comments.length < texts.length,
      scope: 'Comments currently present in the page, including loaded replies' };
  }
  chrome.runtime.onMessage.addListener((message, sender, respond) => {
    if (message?.type === 'MOZHI_ENABLE') { enable(); respond({ enabled: true, videoId: state.video }); }
    if (message?.type === 'MOZHI_DISABLE') { disable(); respond({ enabled: false }); }
    if (message?.type === 'MOZHI_COLLECT') respond(collect());
    if (message?.type === 'MOZHI_VIDEO_ID') respond({ videoId: videoId() });
    if (message?.type === 'MOZHI_PAGE_STATUS') respond({ enabled: state.enabled, videoId: state.video, ...totals(), error: state.error });
    return false;
  });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local' || !changes.mozhiEnabled) return;
    state.storageSeen = true;
    if (changes.mozhiEnabled.newValue === true) enable(); else disable();
  });
  chrome.storage.local.get('mozhiEnabled').then(value => { if (!state.storageSeen && value.mozhiEnabled === true) enable(); }).catch(debug);
  document.addEventListener('yt-navigate-finish', () => { if (state.enabled) scan(); });
  addEventListener('popstate', () => { if (state.enabled) scan(); });
  addEventListener('pageshow', () => { if (state.enabled) scan(); });
})();
