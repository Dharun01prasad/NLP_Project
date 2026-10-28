// Optional developer check: install Playwright and Chromium separately.
const { chromium } = require(process.env.MOZHI_PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.MOZHI_CHROME || undefined, args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1060 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8000');
    await page.waitForFunction(() => document.querySelector('#model-pill').textContent.includes('ready'));
    const screenshot = async (name) => { if (process.env.MOZHI_SCREENSHOTS) await page.screenshot({ path: path.join(process.env.MOZHI_SCREENSHOTS, name), fullPage: true }); };
    await screenshot('lab_initial.png');
    await page.click('#load-examples');
    await page.click('#analyze-button');
    await page.waitForFunction(() => document.querySelector('#stat-count').textContent === '5');
    assert.equal(await page.locator('.result-row').count(), 5);
    await screenshot('lab_results.png');
    await page.click('[data-filter="toxic"]');
    await page.click('[data-filter="all"]');
    assert.equal(await page.locator('.result-row').count(), 5);
    const download = page.waitForEvent('download');
    await page.click('#export-results');
    assert.equal((await download).suggestedFilename(), 'mozhi_predictions.csv');
    await page.click('[data-page="experiments"]');
    assert.equal(await page.locator('tbody tr').count(), 3);
    await screenshot('lab_experiments.png');
    await page.click('[data-page="dataset"]');
    assert.ok((await page.locator('#dataset-name').textContent()).length);
    await screenshot('lab_dataset.png');
    await page.click('[data-page="review"]');
    await page.waitForTimeout(200);
    await screenshot('lab_review.png');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.click('[data-page="analyze"]');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await screenshot('lab_mobile.png');
    assert.deepEqual(errors, []);
    console.log('PASS: desktop analysis, filters, export, report, dataset/review views, mobile layout; no browser exceptions');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
