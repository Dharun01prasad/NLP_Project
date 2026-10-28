const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.join(__dirname, '..', 'extension');

function collector(texts) {
  const listeners = [];
  const sandbox = {
    URL,
    location: { href: 'https://www.youtube.com/watch?v=videoA' },
    document: {
      title: 'Fixture video',
      querySelector: () => ({ innerText: 'Fixture video' }),
      querySelectorAll: () => texts.map(text => ({ innerText: text })),
    },
    chrome: { runtime: { onMessage: { addListener: fn => listeners.push(fn) } } },
  };
  vm.createContext(sandbox);
  const script = fs.readFileSync(path.join(root, 'content.js'), 'utf8');
  vm.runInContext(script, sandbox);
  return { sandbox, listeners, script, collect: () => {
    let result;
    listeners[0]({ type: 'MOZHI_COLLECT' }, {}, data => { result = data; });
    return result;
  } };
}

test('extracts loaded comment text including duplicates, preserving actual comment count', () => {
  const { collect } = collector([' nice ', 'nice', '', 'தமிழ் கருத்து', 'x'.repeat(10001)]);
  const result = collect();
  assert.equal(result.comments.length, 3);
  assert.equal(result.comments[2], 'தமிழ் கருத்து');
  assert.equal(result.videoId, 'videoA');
});

test('limits comments and records truncation', () => {
  const result = collector(Array.from({ length: 520 }, (_, i) => `comment ${i}`)).collect();
  assert.equal(result.comments.length, 500);
  assert.equal(result.loaded, 520);
  assert.equal(result.truncated, true);
  const large = collector(Array(150).fill('a'.repeat(9999))).collect();
  assert.ok(large.comments.join('').length <= 1_000_000);
  assert.ok(large.truncated);
});

test('handles YouTube navigation and avoids duplicate message listeners', () => {
  const c = collector(['hello']);
  vm.runInContext(c.script, c.sandbox);
  assert.equal(c.listeners.length, 1);
  c.sandbox.location.href = 'https://www.youtube.com/watch?v=videoB';
  assert.equal(c.collect().videoId, 'videoB');
  c.sandbox.location.href = 'https://www.youtube.com/shorts/anything';
  assert.ok(c.collect().error);
});

function background(mockFetch) {
  let handler;
  const sandbox = { AbortSignal, fetch: mockFetch, chrome: { runtime: { id: 'test-extension', onMessage: { addListener: fn => { handler = fn; } } } } };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(root, 'background.js'), 'utf8'), sandbox);
  return { handler, request: (message) => new Promise(resolve => handler(message, { id: 'test-extension' }, resolve)) };
}

test('background sends text only to the fixed local analysis endpoint', async () => {
  let request;
  const b = background(async (url, options) => { request = { url, options }; return { ok: true, json: async () => ({ results: [] }) }; });
  const response = await b.request({ type: 'MOZHI_ANALYZE', comments: ['hello'] });
  assert.equal(response.ok, true);
  assert.equal(request.url, 'http://127.0.0.1:8000/api/analyze');
  assert.deepEqual(JSON.parse(request.options.body), { comments: ['hello'] });
  assert.equal(b.handler({ type: 'MOZHI_STATUS' }, { id: 'another-extension' }, () => {}), false);
});

test('background rejects invalid batches and exposes backend failures', async () => {
  const b = background(async () => ({ ok: false, json: async () => ({ detail: 'No model trained' }) }));
  assert.equal((await b.request({ type: 'MOZHI_ANALYZE', comments: [] })).ok, false);
  assert.match((await b.request({ type: 'MOZHI_ANALYZE', comments: ['ok'] })).error, /No model/);
});

test('manifest uses local-only inference permissions and no remote scripts', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'));
  assert.equal(manifest.manifest_version, 3);
  assert.deepEqual(manifest.host_permissions, ['http://127.0.0.1:8000/*']);
  assert.deepEqual(manifest.permissions, ['activeTab', 'scripting']);
  for (const file of ['popup.js', 'background.js', 'content.js']) assert.ok(fs.existsSync(path.join(root, file)));
});
