// Client-side EN ↔ zh-Hant toggle. English in templates is the source of truth;
// this file holds the translation dictionary and walks the DOM on demand.
// Persisted in localStorage as 'lang'; pre-paint sets <html lang> in _layout.html.
(function () {
  // ── Exact-match dictionary ────────────────────────────────────────────
  const ZH = {
    // Nav + sidebar
    'Live inference': '即時推論',
    'Data recording': '資料錄製',
    'Train model': '訓練模型',
    'inference': '推論',
    'sensors': '感測器',

    // Page subtitles — Dashboard
    'never': '從未',
    'waiting…': '等待中…',
    'waiting for first 5 inferences…': '等待前 5 次推論…',

    // Page subtitles — Record
    'Existing data': '現有資料',
    'CSV files discovered in the <code>data/</code> directory.': '在 <code>data/</code> 目錄中發現了 CSV 檔案。',
    'No recordings yet — capture one below.': '尚未有錄製檔 — 在下方擷取一個。',
    'Label': '標籤',
    'labels': '標籤',
    'Samples': '樣本數',
    'Duration': '時長',
    'Size': '大小',
    'Last modified': '最後修改',
    'New recording': '新增錄製',
    'Name a label, set a duration, hit start.': '命名標籤、設定時長、點擊開始。',
    'Name': '名稱',
    'Sensor port': '感測器埠',
    'Mode': '模式',
    'Append': '附加',
    'Overwrite': '覆寫',
    'Append adds rows to an existing file. Overwrite replaces it.': '附加會將資料加入既有檔案。覆寫會取代它。',
    'Seconds': '秒數',
    'Start recording': '開始錄製',
    'Recording in progress': '錄製進行中',
    'Elapsed': '經過時間',
    'Progress': '進度',
    'Port': '埠',
    'Cancel': '取消',

    // Page subtitles — Train
    'Status': '狀態',
    'Current backbone and head artifacts.': '目前的骨幹與頭部檔案。',
    'Backbone': '骨幹',
    'Backbone mode': '骨幹模式',
    'Current head': '目前頭部',
    'missing': '缺失',
    'untrained': '未訓練',
    'Training data': '訓練資料',
    'No recordings yet — capture some via Data recording.': '尚未有錄製檔 — 透過資料錄製頁面擷取一些。',
    'Training in progress': '訓練進行中',
    'Current label': '目前標籤',
    'Labels done': '已完成標籤',
    'Eligible': '符合資格',
    'Windows': '視窗數',

    // Buttons / helpers
    'Select all eligible': '全選符合資格者',
    'Select at least 2 eligible labels.': '請選擇至少 2 個符合資格的標籤。',
    'Failed to start': '啟動失敗',
    'no sensors connected': '未連接感測器',
    'No sensors detected. Edit ALLOWED_PORTS in sensor_reader.py.': '未偵測到感測器。請編輯 sensor_reader.py 中的 ALLOWED_PORTS。',

    // Toggle button itself (rendered manually below — these are for completen'中文' : ess): 'EN',
  };

  // ── Pattern-match rules for templated strings ─────────────────────────
  const PATTERNS = [
    { re: /^(\d+) ports? available\.$/, zh: m => `共 ${m[1]} 個埠可用。` },
    { re: /^last (\d+)$/, zh: m => `最近 ${m[1]} 次` },
    { re: /^latest reading (.+) ago$/, zh: m => `最新讀取於 ${m[1]} 前` },
    { re: /^latest confidence (.+) · majority (\d+)\/(\d+)$/,
      zh: m => `最新信心 ${m[1]} · 多數 ${m[2]}/${m[3]}` },
    { re: /^Saved (.+) \((.+) samples\)$/, zh: m => `已儲存 ${m[1]} (${m[2]} 樣本)` },
    { re: /^Cancelled — kept (.+) samples in (.+)$/,
      zh: m => `已取消 — 保留 ${m[1]} 樣本於 ${m[2]}` },
    { re: /^Error: (.+)$/, zh: m => `錯誤: ${m[1]}` },
    { re: /^(\d+) labels — (.+)$/, zh: m => `${m[1]} 個標籤 — ${m[2]}` },
  ];

  const ATTRS = ['placeholder', 'title', 'aria-label'];
  const TEXT_CACHE = new WeakMap(); // textNode -> original English
  const ATTR_CACHE = new WeakMap(); // element -> { attrName: originalEn }

  function translate(en) {
    if (!en) return en;
    const trimmed = en.trim();
    if (!trimmed) return en;
    // Skip pure numbers / pure punctuation — they don't need translation.
    if (/^[\d\s.,:;\-—·•%/()\[\]]+$/.test(trimmed)) return en;
    const lead = en.match(/^\s*/)[0];
    const trail = en.match(/\s*$/)[0];
    if (ZH[trimmed]) return lead + ZH[trimmed] + trail;
    for (const { re, zh } of PATTERNS) {
      const m = trimmed.match(re);
      if (m) return lead + zh(m) + trail;
    }
    return en;
  }

  function applyToTextNode(node, lang) {
    let original = TEXT_CACHE.get(node);
    if (original === undefined) {
      original = node.nodeValue;
      TEXT_CACHE.set(node, original);
    }
    const newVal = lang === 'zh-Hant' ? translate(original) : original;
    if (node.nodeValue !== newVal) node.nodeValue = newVal;
  }

  function applyToAttribute(el, attr, lang) {
    let cache = ATTR_CACHE.get(el);
    if (!cache) { cache = {}; ATTR_CACHE.set(el, cache); }
    if (cache[attr] === undefined) cache[attr] = el.getAttribute(attr);
    const original = cache[attr];
    if (original == null) return;
    const newVal = lang === 'zh-Hant' ? translate(original) : original;
    if (el.getAttribute(attr) !== newVal) el.setAttribute(attr, newVal);
  }

  function walk(root, lang) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    let node;
    while ((node = walker.nextNode())) {
      const tag = node.parentNode && node.parentNode.nodeName;
      if (tag === 'SCRIPT' || tag === 'STYLE') continue;
      applyToTextNode(node, lang);
    }
    for (const attr of ATTRS) {
      root.querySelectorAll('[' + attr + ']').forEach(el => applyToAttribute(el, attr, lang));
    }
  }

  const currentLang = () => localStorage.getItem('lang') === 'zh-Hant' ? 'zh-Hant' : 'en';

  let observer = null;
  function withoutObserver(fn) {
    if (observer) observer.disconnect();
    try { fn(); } finally {
      if (observer) observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    }
  }
  const applyAll = () => withoutObserver(() => walk(document.body, currentLang()));

  // Expose t() so JS-built strings can opt into translation explicitly:
  //   showToast(t('Saved {p} ({n} samples)', { p: path, n: count }));
  window.t = function (s, vars) {
    let out = currentLang() === 'zh-Hant' ? translate(s) : s;
    if (vars) for (const k in vars) out = out.replace(new RegExp('\\{' + k + '\\}', 'g'), vars[k]);
    return out;
  };

  window.setLang = function (lang) {
    lang = lang === 'zh-Hant' ? 'zh-Hant' : 'en';
    localStorage.setItem('lang', lang);
    document.documentElement.lang = lang;
    applyAll();
    const btn = document.getElementById('lang-toggle');
    if (btn) btn.textContent = lang === 'zh-Hant' ? '中文' : 'EN';
  };

  function init() {
    document.documentElement.lang = currentLang();
    applyAll();
    const btn = document.getElementById('lang-toggle');
    if (btn) {
      // Toggle button label reflects what it'll switch TO, not the current state.
      btn.textContent = currentLang() === 'zh-Hant' ? '中文' : 'EN';
      btn.addEventListener('click', () => window.setLang(currentLang() === 'zh-Hant' ? 'en' : 'zh-Hant'));
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  // Catch JS-injected text — disconnect during our own writes so we don't loop.
  observer = new MutationObserver(muts => {
    if (currentLang() !== 'zh-Hant') return;
    withoutObserver(() => {
      for (const m of muts) {
        for (const n of m.addedNodes) {
          if (n.nodeType === Node.ELEMENT_NODE) walk(n, 'zh-Hant');
          else if (n.nodeType === Node.TEXT_NODE) applyToTextNode(n, 'zh-Hant');
        }
        if (m.type === 'characterData' && m.target.nodeType === Node.TEXT_NODE) {
          TEXT_CACHE.delete(m.target);
          applyToTextNode(m.target, 'zh-Hant');
        }
      }
    });
  });
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
})();
