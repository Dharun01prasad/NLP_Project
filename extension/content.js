(() => {
  if (globalThis.__mozhiInstalled) return;
  globalThis.__mozhiInstalled = true;
  function videoId() {
    const url = new URL(location.href);
    return url.pathname === '/watch' ? url.searchParams.get('v') : null;
  }
  function collect() {
    const id = videoId();
    if (!id) return { error: 'Open a YouTube video watch page first.' };
    // YouTube changes markup. Keep selectors centralized and fail visibly if none match.
    const elements = [...document.querySelectorAll('ytd-comment-thread-renderer #content-text, ytd-comment-view-model #content-text, ytd-comment-renderer #content-text')];
    const comments = elements.map(element => element.innerText?.trim()).filter(text => text && text.length <= 10000);
    const eligible = comments.reduce((state, text) => {
      if (state.comments.length < 500 && state.characters + text.length <= 1_000_000) {
        state.comments.push(text); state.characters += text.length;
      }
      return state;
    }, { comments: [], characters: 0 });
    return { videoId: id, title: document.querySelector('h1.ytd-watch-metadata')?.innerText?.trim() || document.title,
      comments: eligible.comments, loaded: comments.length, truncated: eligible.comments.length < comments.length,
      scope: 'Comments currently present in the page, including loaded replies' };
  }
  chrome.runtime.onMessage.addListener((message, sender, respond) => {
    if (message?.type === 'MOZHI_COLLECT') respond(collect());
    if (message?.type === 'MOZHI_VIDEO_ID') respond({ videoId: videoId() });
    return false;
  });
})();
