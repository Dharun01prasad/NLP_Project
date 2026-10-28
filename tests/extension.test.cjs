const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.join(__dirname, '..', 'extension');
const source = name => fs.readFileSync(path.join(root, name), 'utf8');

class Node {
  constructor(tag = 'span', text = '') {
    this.tagName = tag; this.innerText = text; this.textContent = text; this.children = [];
    this.className = ''; this.isConnected = true; this.nodeType = 1; this.attributes = {};
    this.classList = { contains: name => this.className.split(' ').includes(name),
      add: name => this.classList.toggle(name, true), remove: name => this.classList.toggle(name, false),
      toggle: (name, on) => {
      const names = new Set(this.className.split(' ').filter(Boolean));
      if (on) names.add(name); else names.delete(name);
      this.className = [...names].join(' ');
    } };
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  attachShadow() { this.shadowRoot = new Node('shadow'); return this.shadowRoot; }
  querySelector(selector) {
    if (selector.startsWith('.')) return this.children.find(node => node.className === selector.slice(1));
    return null;
  }
  insertAdjacentElement(position, node) { this.host = node; node.isConnected = true; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, callback) { (this.listeners ||= {})[name] = callback; }
  fire(name, event = {}) { return this.listeners?.[name]?.(event); }
  remove() { this.isConnected = false; }
}

function contentFixture({ texts = [], url = 'https://www.youtube.com/watch?v=videoA', enabled = false, reply } = {}) {
  const listeners = [], storageListeners = [], events = {}, observers = [], timers = new Map(), calls = [];
  let timerId = 0, queryCount = 0;
  const elements = texts.map(text => new Node('comment', text));
  const document = {
    title: 'Fixture video', body: new Node('body'), documentElement: { hasAttribute: () => false },
    createElement: tag => new Node(tag),
    querySelector: () => ({ innerText: 'Fixture video' }),
    querySelectorAll: () => { queryCount++; return elements.filter(element => element.isConnected); },
    addEventListener: (name, fn) => { events[name] = fn; },
  };
  class Observer {
    constructor(callback) { this.callback = callback; this.active = false; observers.push(this); }
    observe() { this.active = true; }
    disconnect() { this.active = false; }
    emit(nodes) { if (this.active) this.callback([{ addedNodes: nodes, removedNodes: [] }]); }
  }
  const storage = { mozhiEnabled: enabled };
  const chrome = {
    runtime: {
      onMessage: { addListener: fn => listeners.push(fn) },
      sendMessage: async request => {
        calls.push(request);
        if (reply) return reply(request);
        return { ok: true, data: { results: request.comments.map(comment =>
          ({ comment, label: 'non_toxic', score: 1, probability: .1, uncertain: false })) } };
      },
    },
    storage: {
      local: { get: async () => ({ ...storage }) },
      onChanged: { addListener: fn => storageListeners.push(fn) },
    },
  };
  const context = vm.createContext({ URL, document, chrome, location: { href: url }, MutationObserver: Observer,
    setTimeout: (fn, delay) => { assert.equal(delay, 1400); const id = ++timerId; timers.set(id, fn); return id; },
    clearTimeout: id => timers.delete(id), addEventListener: (name, fn) => { events[name] = fn; }, console });
  vm.runInContext(source('content.js'), context);
  return {
    context, elements, document, observers, calls, timers,
    get queryCount() { return queryCount; },
    async settled() { await Promise.resolve(); await Promise.resolve(); },
    async tick() { const queued = [...timers.values()]; timers.clear(); for (const callback of queued) await callback(); },
    request(message) { let value; listeners[0](message, {}, response => { value = response; }); return value; },
    toggle(on) { storage.mozhiEnabled = on; storageListeners.forEach(fn => fn({ mozhiEnabled: { newValue: on } }, 'local')); },
    mutate(nodes = [new Node('ytd-comment-renderer')]) { observers.at(-1)?.emit(nodes); },
    navigate(nextUrl, nextTexts = []) { elements.forEach(element => { element.isConnected = false; }); elements.splice(0, elements.length, ...nextTexts.map(text => new Node('comment', text))); context.location.href = nextUrl; events['yt-navigate-finish']?.(); },
  };
}

test('default OFF persists, does not scan, observe, or call API', async () => {
  const fixture = contentFixture({ texts: ['hello'] });
  await fixture.settled();
  assert.equal(fixture.queryCount, 0);
  assert.equal(fixture.observers.length, 0);
  assert.equal(fixture.calls.length, 0);
  assert.equal(fixture.request({ type: 'MOZHI_COLLECT' }).error, 'Turn Mozhi ON to analyze comments.');
  assert.equal(fixture.document.body.children.length, 0);
});

test('ON scans YouTube watch pages only', async () => {
  for (const url of ['https://www.youtube.com/', 'https://www.youtube.com/results?search_query=x',
    'https://www.youtube.com/shorts/abc', 'https://evil.example/watch?v=videoA',
    'https://www.youtube.com/watch']) {
    const fixture = contentFixture({ url, enabled: true, texts: ['hello'] });
    await fixture.settled();
    assert.equal(fixture.queryCount, 0, url);
    assert.equal(fixture.calls.length, 0, url);
    assert.equal(fixture.observers.length, 0, url);
  }
  const fixture = contentFixture({ enabled: true, texts: ['hello'] });
  await fixture.settled(); await fixture.tick();
  assert.equal(fixture.calls.length, 1);
  assert.equal(fixture.observers.length, 1);
  assert.equal(fixture.observers[0].active, true);
});

test('duplicate DOM identities are separate; rescans never repeat or inject twice', async () => {
  const fixture = contentFixture({ enabled: true, texts: ['same', 'same'] });
  await fixture.settled(); await fixture.tick();
  assert.deepEqual(Array.from(fixture.calls[0].comments), ['same', 'same']);
  const firstHost = fixture.elements[0].host;
  assert.ok(firstHost?.shadowRoot);
  fixture.mutate(); await fixture.tick();
  assert.equal(fixture.calls.length, 1);
  assert.equal(fixture.elements[0].host, firstHost);
  assert.ok(fixture.elements[1].host);
});

test('MutationObserver queues new rendered replies after debounce and batches at 100', async () => {
  const fixture = contentFixture({ enabled: true, texts: Array.from({ length: 205 }, (_, i) => `comment ${i}`) });
  await fixture.settled();
  assert.equal(fixture.calls.length, 0);
  assert.equal(fixture.timers.size, 1);
  await fixture.tick(); await fixture.tick(); await fixture.tick();
  assert.deepEqual(fixture.calls.map(call => call.comments.length), [100, 100, 5]);
  fixture.elements.push(new Node('comment', 'new reply'));
  fixture.mutate();
  assert.equal(fixture.calls.length, 3);
  await fixture.tick();
  assert.deepEqual(Array.from(fixture.calls[3].comments), ['new reply']);
});

test('SPA video navigation removes old results, disconnects observer and discards old summary', async () => {
  const fixture = contentFixture({ enabled: true, texts: ['first'] });
  await fixture.settled(); await fixture.tick();
  const previousHost = fixture.elements[0].host, previousPanel = fixture.document.body.children[0];
  fixture.navigate('https://www.youtube.com/watch?v=videoB', ['second']);
  assert.equal(previousHost.isConnected, false);
  assert.equal(previousPanel.isConnected, false);
  assert.equal(fixture.observers[0].active, false);
  assert.equal(fixture.request({ type: 'MOZHI_PAGE_STATUS' }).analyzed, 0);
  await fixture.tick();
  assert.deepEqual(Array.from(fixture.calls[1].comments), ['second']);
  fixture.navigate('https://www.youtube.com/shorts/abc');
  assert.equal(fixture.request({ type: 'MOZHI_PAGE_STATUS' }).videoId, null);
  assert.equal(fixture.observers.at(-1).active, false);
});

test('URL change before old DOM removal never reclassifies old video comments', async () => {
  const fixture = contentFixture({ enabled: true, texts: ['old video comment'] });
  await fixture.settled(); await fixture.tick();
  fixture.context.location.href = 'https://www.youtube.com/watch?v=videoB';
  fixture.mutate();
  await fixture.tick();
  assert.equal(fixture.calls.length, 1);
  assert.equal(fixture.request({ type: 'MOZHI_PAGE_STATUS' }).analyzed, 0);
  fixture.elements[0].isConnected = false;
  fixture.elements.push(new Node('comment', 'new video comment'));
  fixture.mutate(); await fixture.tick();
  assert.deepEqual(Array.from(fixture.calls[1].comments), ['new video comment']);
});

test('OFF removes indicators, summary and observer, and prevents future API calls', async () => {
  const fixture = contentFixture({ enabled: true, texts: ['hello'] });
  await fixture.settled(); await fixture.tick();
  const host = fixture.elements[0].host, panel = fixture.document.body.children[0];
  fixture.toggle(false);
  assert.equal(host.isConnected, false);
  assert.equal(panel.isConnected, false);
  assert.equal(fixture.observers[0].active, false);
  fixture.elements.push(new Node('comment', 'later'));
  fixture.mutate(); await fixture.tick();
  assert.equal(fixture.calls.length, 1);
});

test('all five emoji score ranges, accessibility, percentage and comment matching', async () => {
  const scores = [1.9, 2, 4, 6, 8];
  const fixture = contentFixture({ enabled: true, texts: scores.map((_, i) => `text ${i}`),
    reply: async request => ({ ok: true, data: { results: request.comments.map((comment, i) =>
      ({ comment, label: 'non_toxic', score: scores[i], probability: scores[i] / 10, uncertain: i === 2 })) } }) });
  await fixture.settled(); await fixture.tick();
  const icons = ['😊', '🙂', '😐', '😠', '🤬'];
  fixture.elements.forEach((comment, i) => {
    const badge = comment.host.shadowRoot.children[1].children[0];
    assert.ok(badge.textContent.startsWith(icons[i]));
    assert.match(badge.title, /Toxicity likelihood/);
    assert.match(badge.attributes['aria-label'], /out of 10/);
    assert.ok(badge.textContent.includes(`${Math.round(scores[i] * 10)}%`));
    if (i === 2) assert.equal(badge.children[0].textContent, 'Needs context');
  });
  assert.equal(fixture.request({ type: 'MOZHI_PAGE_STATUS' }).analyzed, 5);
});

test('mismatched backend result does not attach another comment result', async () => {
  const fixture = contentFixture({ enabled: true, texts: ['one', 'two'],
    reply: async () => ({ ok: true, data: { results: [{ comment: 'wrong', score: 9, probability: .9, label: 'toxic', uncertain: false },
      { comment: 'two', score: 1, probability: .1, label: 'non_toxic', uncertain: false }] } }) });
  await fixture.settled(); await fixture.tick();
  assert.equal(fixture.elements[0].host, undefined);
  assert.ok(fixture.document.body.children[0].shadowRoot.querySelector('.panel').children.some(node => node.className === 'error'));
});

test('backend-unavailable error is visible in page summary', async () => {
  const fixture = contentFixture({ enabled: true, texts: ['hello'], reply: async () => ({ ok: false, error: 'Local backend unavailable' }) });
  await fixture.settled(); await fixture.tick();
  const panel = fixture.document.body.children[0].shadowRoot.querySelector('.panel');
  assert.match(panel.children.at(-1).textContent, /backend unavailable/);
  assert.equal(fixture.elements[0].host, undefined);
});

function backgroundFixture(enabled = true, fetcher = async () => ({ ok: true, json: async () => ({ results: [] }) })) {
  let handler, tabUpdated;
  const calls = [], changes = [], injected = [];
  const sandbox = { URL, AbortController, setTimeout, clearTimeout,
    fetch: async (url, options) => { calls.push({ url, options }); return fetcher(url, options); },
    chrome: { runtime: { id: 'mozhi-test', onMessage: { addListener: fn => { handler = fn; } } },
      tabs: { onUpdated: { addListener: fn => { tabUpdated = fn; } }, sendMessage: async (tabId, message) => { injected.push(message.type); } },
      scripting: { executeScript: async () => { injected.push('inject'); } },
      storage: { local: { get: async () => ({ mozhiEnabled: enabled }) }, onChanged: { addListener: fn => changes.push(fn) } } } };
  vm.createContext(sandbox); vm.runInContext(source('background.js'), sandbox);
  return { calls, injected, updated: (url, status = 'complete') => tabUpdated(4, { status }, { url }),
    request: message => new Promise(resolve => handler(message, { id: 'mozhi-test' }, resolve)),
    foreign: message => handler(message, { id: 'another-extension' }, () => {}),
    setEnabled(value) { enabled = value; changes.forEach(fn => fn({ mozhiEnabled: { newValue: value } }, 'local')); } };
}

test('background sends requests only to local API and limits each request to 100', async () => {
  const fixture = backgroundFixture();
  assert.equal((await fixture.request({ type: 'MOZHI_ANALYZE', comments: ['hello'] })).ok, true);
  assert.equal(fixture.calls[0].url, 'http://127.0.0.1:8000/api/analyze');
  assert.deepEqual(JSON.parse(fixture.calls[0].options.body), { comments: ['hello'] });
  assert.equal((await fixture.request({ type: 'MOZHI_ANALYZE', comments: Array(101).fill('x') })).ok, false);
  assert.equal(fixture.calls.length, 1);
  assert.equal(fixture.foreign({ type: 'MOZHI_STATUS' }), false);
});

test('background blocks OFF requests and exposes backend/model failures', async () => {
  const fixture = backgroundFixture(false, async () => ({ ok: false, json: async () => ({ detail: 'No trained model' }) }));
  assert.equal((await fixture.request({ type: 'MOZHI_STATUS' })).ok, false);
  assert.equal((await fixture.request({ type: 'MOZHI_ANALYZE', comments: ['x'] })).ok, false);
  assert.equal(fixture.calls.length, 0);
  fixture.setEnabled(true);
  assert.match((await fixture.request({ type: 'MOZHI_ANALYZE', comments: ['x'] })).error, /No trained model/);
});

test('background injects only when ON and a watch page finishes loading', async () => {
  const fixture = backgroundFixture(false);
  fixture.updated('https://www.youtube.com/watch?v=videoA');
  await new Promise(setImmediate);
  assert.equal(fixture.injected.length, 0);
  fixture.setEnabled(true);
  fixture.updated('https://www.youtube.com/results?search_query=x');
  fixture.updated('https://www.youtube.com/shorts/abc');
  await new Promise(setImmediate);
  assert.equal(fixture.injected.length, 0);
  fixture.updated('https://www.youtube.com/watch?v=videoA');
  await new Promise(setImmediate);
  assert.deepEqual(fixture.injected, ['inject', 'MOZHI_ENABLE']);
});

test('switching OFF aborts an analysis already in flight', async () => {
  let started;
  const began = new Promise(resolve => { started = resolve; });
  const fixture = backgroundFixture(true, (url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(Object.assign(new Error('stopped'), { name: 'AbortError' })));
    started();
  }));
  const response = fixture.request({ type: 'MOZHI_ANALYZE', comments: ['hello'] });
  await began;
  fixture.setEnabled(false);
  assert.match((await response).error, /stopped or timed out/);
});

test('popup toggle persists OFF/ON state and manual analysis batches 205 comments', async () => {
  const storage = { mozhiEnabled: false }, calls = [], nodes = new Map();
  const document = { getElementById: id => {
    if (!nodes.has(id)) nodes.set(id, new Node('div'));
    return nodes.get(id);
  }, createElement: tag => new Node(tag) };
  const chrome = {
    storage: { local: { get: async () => ({ ...storage }), set: async value => Object.assign(storage, value) } },
    tabs: { query: async () => [{ id: 4, url: 'https://www.youtube.com/watch?v=videoA' }],
      sendMessage: async (tabId, request) => {
        if (request.type === 'MOZHI_COLLECT') return { videoId: 'videoA', title: 'Video',
          comments: Array.from({ length: 205 }, (_, i) => `comment ${i}`), truncated: false };
        if (request.type === 'MOZHI_VIDEO_ID') return { videoId: 'videoA' };
        return { enabled: request.type === 'MOZHI_ENABLE' };
      }, create: () => {} },
    scripting: { executeScript: async () => { calls.push('inject'); } },
    runtime: { sendMessage: async request => {
      calls.push(request);
      if (request.type === 'MOZHI_STATUS') return { ok: true, data: { model_ready: true } };
      return { ok: true, data: { results: request.comments.map(comment =>
        ({ comment, label: 'non_toxic', score: 1, probability: .1, uncertain: false })) } };
    } },
  };
  const context = vm.createContext({ URL, document, chrome, Blob, setTimeout, clearTimeout, console });
  vm.runInContext(source('popup.js'), context);
  await new Promise(setImmediate);
  assert.equal(storage.mozhiEnabled, false);
  assert.equal(document.getElementById('toggle-status').textContent, 'OFF · no scanning');
  assert.equal(calls.length, 0);
  await document.getElementById('enabled').fire('change', { target: { checked: true } });
  assert.equal(storage.mozhiEnabled, true);
  assert.equal(document.getElementById('toggle-status').textContent, 'ON · analyzing rendered comments');
  await document.getElementById('analyze').fire('click');
  assert.deepEqual(calls.filter(call => call.type === 'MOZHI_ANALYZE').map(call => call.comments.length), [100, 100, 5]);
  assert.equal(document.getElementById('count').textContent, 205);
  await document.getElementById('enabled').fire('change', { target: { checked: false } });
  assert.equal(storage.mozhiEnabled, false);
  assert.equal(document.getElementById('toggle-status').textContent, 'OFF · no scanning');
  assert.equal(document.getElementById('analysis').classList.contains('hidden'), true);
});

test('manifest has only activeTab, scripting, storage and fixed loopback host', () => {
  const manifest = JSON.parse(source('manifest.json'));
  assert.equal(manifest.manifest_version, 3);
  assert.deepEqual(manifest.permissions, ['activeTab', 'scripting', 'storage']);
  assert.deepEqual(manifest.host_permissions, ['http://127.0.0.1:8000/*']);
  assert.equal(manifest.background.service_worker, 'background.js');
  assert.equal(manifest.content_scripts, undefined);
  for (const file of ['popup.js', 'background.js', 'content.js', 'popup-toggle.css']) assert.ok(fs.existsSync(path.join(root, file)));
});
