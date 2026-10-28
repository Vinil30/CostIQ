/* CostIQ — shared UI behaviour (no dependencies) */
(function () {
  'use strict';

  const root = document.documentElement;
  root.classList.add('js');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── Theme ─────────────────────────────────────────────────────────── */
  function currentTheme() {
    const set = root.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.querySelectorAll('[data-theme-toggle]').forEach(btn => {
    btn.addEventListener('click', () => {
      const next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('costiq-theme', next); } catch (e) { /* storage unavailable */ }
    });
  });

  /* ── Mobile nav ────────────────────────────────────────────────────── */
  document.querySelectorAll('[data-nav-toggle]').forEach(btn => {
    const nav = btn.closest('.nav');
    btn.addEventListener('click', () => {
      const open = nav.classList.toggle('is-open');
      btn.setAttribute('aria-expanded', String(open));
    });
    nav.querySelectorAll('.nav-menu a').forEach(a => a.addEventListener('click', () => {
      nav.classList.remove('is-open');
      btn.setAttribute('aria-expanded', 'false');
    }));
  });

  /* ── Toasts ────────────────────────────────────────────────────────── */
  function dismiss(toast) {
    if (!toast || toast.classList.contains('is-leaving')) return;
    toast.classList.add('is-leaving');
    setTimeout(() => toast.remove(), 260);
  }
  document.querySelectorAll('.toast').forEach((t, i) => {
    t.querySelector('.toast-close')?.addEventListener('click', () => dismiss(t));
    const isError = t.classList.contains('toast-danger');
    setTimeout(() => dismiss(t), (isError ? 9000 : 5000) + i * 400);
  });

  /* ── Tabs (role="tablist") ─────────────────────────────────────────── */
  window.CostIQ = window.CostIQ || {};
  function activateTab(tab, focus) {
    const list = tab.closest('[role="tablist"]');
    list.querySelectorAll('[role="tab"]').forEach(t => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      const panel = document.getElementById(t.getAttribute('aria-controls'));
      if (panel) panel.hidden = !on;
    });
    if (focus) tab.focus();
  }
  function initTabs(scope) {
    (scope || document).querySelectorAll('[role="tablist"]').forEach(list => {
      if (list.dataset.bound) return;
      list.dataset.bound = '1';
      const tabs = () => Array.from(list.querySelectorAll('[role="tab"]'));
      list.addEventListener('click', e => {
        const tab = e.target.closest('[role="tab"]');
        if (tab) activateTab(tab);
      });
      list.addEventListener('keydown', e => {
        const all = tabs();
        const i = all.indexOf(document.activeElement);
        if (i < 0) return;
        let n = null;
        if (e.key === 'ArrowRight') n = (i + 1) % all.length;
        if (e.key === 'ArrowLeft')  n = (i - 1 + all.length) % all.length;
        if (e.key === 'Home') n = 0;
        if (e.key === 'End')  n = all.length - 1;
        if (n !== null) { e.preventDefault(); activateTab(all[n], true); }
      });
    });
  }
  window.CostIQ.initTabs = initTabs;
  window.CostIQ.activateTab = activateTab;
  initTabs();

  /* ── Relative times: <time data-relative datetime="ISO (UTC, naive)"> ─ */
  function parseUTC(iso) {
    if (!iso) return null;
    const s = /Z|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + 'Z';
    const d = new Date(s);
    return isNaN(d) ? null : d;
  }
  function relative(d) {
    const sec = Math.round((Date.now() - d.getTime()) / 1000);
    const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
    const steps = [[60, 'second'], [3600, 'minute', 60], [86400, 'hour', 3600], [604800, 'day', 86400], [2629800, 'week', 604800], [31557600, 'month', 2629800], [Infinity, 'year', 31557600]];
    for (const [limit, unit, div] of steps) {
      if (Math.abs(sec) < limit) return rtf.format(-Math.round(sec / (div || 1)), unit);
    }
  }
  function formatLocal(d, opts) {
    return d.toLocaleString(undefined, opts || { dateStyle: 'medium', timeStyle: 'short' });
  }
  function hydrateTimes(scope) {
    (scope || document).querySelectorAll('time[datetime]').forEach(el => {
      const d = parseUTC(el.getAttribute('datetime'));
      if (!d) return;
      el.title = formatLocal(d, { dateStyle: 'full', timeStyle: 'short' });
      if (el.hasAttribute('data-relative')) el.textContent = relative(d);
      else if (el.hasAttribute('data-date')) el.textContent = formatLocal(d, { dateStyle: 'medium' });
      else if (el.hasAttribute('data-time')) el.textContent = formatLocal(d, { timeStyle: 'short' });
      else el.textContent = formatLocal(d);
    });
  }
  window.CostIQ.parseUTC = parseUTC;
  window.CostIQ.relative = relative;
  window.CostIQ.formatLocal = formatLocal;
  window.CostIQ.hydrateTimes = hydrateTimes;
  hydrateTimes();

  /* ── Count-up numbers: <span data-count="1234.5" data-prefix="$" data-decimals="0"> ─ */
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    if (!isFinite(target)) return;
    const dec = parseInt(el.dataset.decimals || '0', 10);
    const pre = el.dataset.prefix || '', suf = el.dataset.suffix || '';
    const fmt = v => pre + v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suf;
    if (reduceMotion) { el.textContent = fmt(target); return; }
    const dur = 900, t0 = performance.now();
    const tick = now => {
      const p = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(target * eased);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  window.CostIQ.countUp = countUp;
  // Count when the number scrolls into view (not on page load).
  const counters = document.querySelectorAll('[data-count]');
  if ('IntersectionObserver' in window && !reduceMotion) {
    const cio = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { countUp(e.target); cio.unobserve(e.target); } }), { threshold: .4 });
    counters.forEach(el => cio.observe(el));
  } else counters.forEach(countUp);

  /* ── Split headings into words for a staggered reveal ─────────────── */
  function splitWords(el, baseDelay) {
    const walk = node => Array.from(node.childNodes).forEach(n => {
      if (n.nodeType === 3) {
        const frag = document.createDocumentFragment();
        n.textContent.split(/(\s+)/).forEach(part => {
          if (!part) return;
          if (/^\s+$/.test(part)) { frag.appendChild(document.createTextNode(part)); return; }
          const w = document.createElement('span'); w.className = 'w'; w.textContent = part; frag.appendChild(w);
        });
        n.replaceWith(frag);
      } else if (n.nodeType === 1) walk(n);
    });
    walk(el);
    el.querySelectorAll('.w').forEach((w, i) => { w.style.animationDelay = w.style.transitionDelay = (baseDelay + i * 55) + 'ms'; });
  }
  window.CostIQ.splitWords = splitWords;
  document.querySelectorAll('.split, [data-split-now]').forEach(el => splitWords(el, el.hasAttribute('data-split-now') ? 80 : 0));

  /* ── Reveal on scroll ──────────────────────────────────────────────── */
  const reveals = document.querySelectorAll('[data-reveal], [data-stagger], .split');
  if ('IntersectionObserver' in window && !reduceMotion) {
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('is-visible'); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    reveals.forEach(el => io.observe(el));
  } else {
    reveals.forEach(el => el.classList.add('is-visible'));
  }

  /* ── Expandable clamps (run on load, after page scripts render content) ─ */
  window.addEventListener('load', () => {
    document.querySelectorAll('[data-clamp-toggle]').forEach(btn => {
      const target = document.getElementById(btn.getAttribute('aria-controls'));
      if (!target) return;
      if (target.scrollHeight <= target.clientHeight + 8) { btn.hidden = true; target.classList.add('is-open'); return; }
      btn.addEventListener('click', () => {
        const open = target.classList.toggle('is-open');
        btn.setAttribute('aria-expanded', String(open));
        btn.textContent = open ? 'Show less' : 'Read more';
      });
    });
  });

  /* ── Password visibility ───────────────────────────────────────────── */
  document.querySelectorAll('[data-toggle-password]').forEach(btn => {
    const input = document.getElementById(btn.getAttribute('aria-controls'));
    btn.addEventListener('click', () => {
      const show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      btn.setAttribute('aria-pressed', String(show));
      btn.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
      btn.querySelector('.icon-eye')?.toggleAttribute('hidden', show);
      btn.querySelector('.icon-eye-off')?.toggleAttribute('hidden', !show);
    });
  });

  /* ── Interactive line chart (no dependencies) ──────────────────────────
     CostIQ.chart(el, { points: [{label, sub, id, <key>: number}], series: {key, name, fmt, tone}, onSelect })
     Hover/touch shows a crosshair + tooltip with change vs the previous point;
     arrow keys move between points, Enter/click calls onSelect; draws in when visible. */
  const fmtShort = (v, fmt) => {
    if (v == null || !isFinite(v)) return '—';
    if (fmt === 'pct') return v.toFixed(1) + '%';
    const a = Math.abs(v), s = v < 0 ? '−$' : '$';
    if (a >= 1e6) return s + (a / 1e6).toFixed(a >= 1e7 ? 1 : 2) + 'M';
    if (a >= 1e3) return s + (a / 1e3).toFixed(a >= 1e5 ? 0 : 1) + 'k';
    return s + Math.round(a);
  };
  const fmtFull = (v, fmt) => v == null || !isFinite(v) ? '—' : fmt === 'pct' ? v.toFixed(1) + '%' : (v < 0 ? '−$' : '$') + Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 });
  window.CostIQ.fmtShort = fmtShort;
  window.CostIQ.fmtFull = fmtFull;

  function chart(el, opts) {
    const NS = 'http://www.w3.org/2000/svg';
    let series = opts.series, active = -1, drawn = false;
    const pts = opts.points || [];
    el.classList.add('chart');
    el.tabIndex = 0;
    el.setAttribute('role', 'img');
    const tip = document.createElement('div'); tip.className = 'chart-tip'; tip.hidden = true;
    el.appendChild(tip);

    function path(xy) {                                       // smooth curve through points (Catmull-Rom → Bézier)
      if (xy.length < 2) return '';
      let d = `M${xy[0][0]},${xy[0][1]}`;
      for (let i = 0; i < xy.length - 1; i++) {
        const p0 = xy[i - 1] || xy[i], p1 = xy[i], p2 = xy[i + 1], p3 = xy[i + 2] || p2, t = .18;
        d += ` C${p1[0] + (p2[0] - p0[0]) * t},${p1[1] + (p2[1] - p0[1]) * t} ${p2[0] - (p3[0] - p1[0]) * t},${p2[1] - (p3[1] - p1[1]) * t} ${p2[0]},${p2[1]}`;
      }
      return d;
    }
    let geo = null;
    function render() {
      el.querySelector('svg')?.remove();
      const W = Math.max(240, el.clientWidth), H = +(el.dataset.height || 180);
      const pad = { l: 8, r: 8, t: 16, b: 26 };
      const vals = pts.map(p => p[series.key]).map(v => (v == null || !isFinite(v)) ? null : +v);
      const nums = vals.filter(v => v != null);
      el.setAttribute('aria-label', `${series.name} across ${pts.length} runs`);
      const svg = document.createElementNS(NS, 'svg');
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
      if (!nums.length) { el.dataset.empty = '1'; el.insertBefore(svg, tip); geo = null; return; }
      delete el.dataset.empty;
      let lo = Math.min(...nums), hi = Math.max(...nums);
      if (series.fmt !== 'pct') lo = Math.min(0, lo);
      if (hi === lo) hi = lo + (Math.abs(hi) || 1);
      const span = hi - lo; hi += span * .12;
      const x = i => pts.length === 1 ? W / 2 : pad.l + 18 + (i * (W - pad.l - pad.r - 36)) / (pts.length - 1);
      const y = v => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
      const tone = `var(--${series.tone || 'brand'})`;
      const gid = 'g' + Math.random().toString(36).slice(2, 8);
      let html = `<defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${tone}" stop-opacity=".28"/><stop offset="1" stop-color="${tone}" stop-opacity="0"/></linearGradient></defs>`;
      [0, .5, 1].forEach(f => { const yy = pad.t + f * (H - pad.t - pad.b); html += `<line class="chart-grid" x1="0" x2="${W}" y1="${yy}" y2="${yy}"/>`; });
      html += `<text class="chart-axis" x="${W - 4}" y="${pad.t - 4}" text-anchor="end">${fmtShort(hi - span * .12, series.fmt)}</text>`;
      const xy = vals.map((v, i) => v == null ? null : [x(i), y(v)]).filter(Boolean);
      if (xy.length > 1) {
        const line = path(xy);
        html += `<path class="chart-area" d="${line} L${xy[xy.length - 1][0]},${H - pad.b} L${xy[0][0]},${H - pad.b} Z" fill="url(#${gid})"/>`;
        html += `<path class="chart-line" d="${line}" stroke="${tone}"/>`;
      }
      html += `<line class="chart-cross" x1="0" x2="0" y1="${pad.t}" y2="${H - pad.b}" opacity="0"/>`;
      vals.forEach((v, i) => { if (v != null) html += `<circle class="chart-dot" data-i="${i}" cx="${x(i)}" cy="${y(v)}" r="3.5" fill="var(--surface)" stroke="${tone}"/>`; });
      const every = Math.max(1, Math.ceil(pts.length / Math.max(2, Math.floor(W / 70))));
      pts.forEach((p, i) => { if (i % every === 0 || i === pts.length - 1) html += `<text class="chart-axis" x="${x(i)}" y="${H - 8}" text-anchor="${i === 0 && pts.length > 1 ? 'start' : i === pts.length - 1 && pts.length > 1 ? 'end' : 'middle'}">${p.label}</text>`; });
      svg.innerHTML = html;
      el.insertBefore(svg, tip);
      geo = { x, y, vals, W, H };
      const ln = svg.querySelector('.chart-line');
      if (ln && !drawn && !reduceMotion) {
        const L = ln.getTotalLength(); ln.style.strokeDasharray = L; ln.style.strokeDashoffset = L;
        svg.querySelector('.chart-area').style.opacity = 0;
      }
      if (drawn) reveal(false);
      if (active >= 0) focus(active);
    }
    function reveal(animate = true) {
      drawn = true;
      const ln = el.querySelector('.chart-line'), ar = el.querySelector('.chart-area');
      if (ln) { ln.style.transition = animate && !reduceMotion ? 'stroke-dashoffset 1.1s cubic-bezier(.16,1,.3,1)' : 'none'; ln.style.strokeDashoffset = 0; }
      if (ar) { ar.style.transition = animate ? 'opacity .8s .4s' : 'none'; ar.style.opacity = 1; }
      el.querySelectorAll('.chart-dot').forEach((d, k) => { d.style.transitionDelay = animate ? (300 + k * 90) + 'ms' : '0ms'; d.classList.add('in'); });
    }
    function focus(i) {
      if (!geo || i < 0 || i >= pts.length || geo.vals[i] == null) return;
      active = i;
      const cx = geo.x(i), cy = geo.y(geo.vals[i]);
      const cross = el.querySelector('.chart-cross');
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('opacity', '1');
      el.querySelectorAll('.chart-dot').forEach(d => d.classList.toggle('on', +d.dataset.i === i));
      const prev = geo.vals.slice(0, i).reverse().find(v => v != null);
      let delta = '';
      if (prev != null && geo.vals[i] != null) {
        const diff = geo.vals[i] - prev, pctChg = prev ? (diff / Math.abs(prev)) * 100 : null;
        const good = series.goodUp === false ? diff < 0 : diff > 0;
        delta = `<span class="${diff === 0 ? '' : good ? 'text-success' : 'text-danger'}">${diff > 0 ? '▲' : diff < 0 ? '▼' : '•'} ${series.fmt === 'pct' ? Math.abs(diff).toFixed(1) + ' pts' : (pctChg == null ? fmtShort(Math.abs(diff)) : Math.abs(pctChg).toFixed(0) + '%')}</span> vs previous`;
      }
      tip.innerHTML = `<div class="chart-tip-k">${pts[i].sub || pts[i].label}</div><div class="chart-tip-v">${fmtFull(geo.vals[i], series.fmt)}</div>${delta ? `<div class="chart-tip-d">${delta}</div>` : ''}${opts.onSelect ? '<div class="chart-tip-h">Click to open run</div>' : ''}`;
      tip.hidden = false;
      const scale = el.clientWidth / geo.W;
      const left = Math.min(Math.max(cx * scale, 70), el.clientWidth - 70);
      tip.style.left = left + 'px'; tip.style.top = Math.max(0, cy * scale - 12) + 'px';
    }
    function blur() {
      active = -1; tip.hidden = true;
      el.querySelector('.chart-cross')?.setAttribute('opacity', '0');
      el.querySelectorAll('.chart-dot').forEach(d => d.classList.remove('on'));
    }
    function nearest(clientX) {
      if (!geo) return -1;
      const r = el.getBoundingClientRect(), px = (clientX - r.left) * (geo.W / r.width);
      let best = -1, bd = Infinity;
      geo.vals.forEach((v, i) => { if (v == null) return; const d = Math.abs(geo.x(i) - px); if (d < bd) { bd = d; best = i; } });
      return best;
    }
    el.addEventListener('pointermove', e => focus(nearest(e.clientX)));
    el.addEventListener('pointerdown', e => focus(nearest(e.clientX)));
    el.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') blur(); });
    el.addEventListener('click', e => { const i = nearest(e.clientX); if (i >= 0 && opts.onSelect) opts.onSelect(pts[i], i); });
    el.addEventListener('focus', () => focus(active >= 0 ? active : pts.length - 1));
    el.addEventListener('blur', blur);
    el.addEventListener('keydown', e => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); for (let i = active - 1; i >= 0; i--) if (geo.vals[i] != null) { focus(i); break; } }
      if (e.key === 'ArrowRight') { e.preventDefault(); for (let i = active + 1; i < pts.length; i++) if (geo.vals[i] != null) { focus(i); break; } }
      if (e.key === 'Enter' && active >= 0 && opts.onSelect) opts.onSelect(pts[active], active);
    });
    if ('ResizeObserver' in window) new ResizeObserver(() => render()).observe(el);
    render();
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { reveal(); io.disconnect(); } }), { threshold: .3 });
      io.observe(el);
    } else reveal(false);
    return { setSeries(s) { series = s; drawn = false; blur(); render(); requestAnimationFrame(() => reveal()); } };
  }
  window.CostIQ.chart = chart;

  /* Tiny inline sparkline: <svg data-spark="1,2,3" data-tone="danger"> */
  document.querySelectorAll('[data-spark]').forEach(svg => {
    const v = svg.dataset.spark.split(',').map(Number).filter(n => isFinite(n));
    if (v.length < 2) { svg.remove(); return; }
    const W = 80, H = 26, lo = Math.min(...v), hi = Math.max(...v), r = hi - lo || 1;
    const p = v.map((n, i) => [(i / (v.length - 1)) * (W - 4) + 2, H - 3 - ((n - lo) / r) * (H - 6)]);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = `<polyline points="${p.map(q => q.join(',')).join(' ')}" fill="none" stroke="var(--${svg.dataset.tone || 'brand'})" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" pathLength="1" class="spark-line"/><circle cx="${p[p.length - 1][0]}" cy="${p[p.length - 1][1]}" r="2.6" fill="var(--${svg.dataset.tone || 'brand'})"/>`;
  });

  /* Pointer-following glow on KPI tiles and .card.glow (no movement, just light) */
  document.addEventListener('pointermove', e => {
    const el = e.target.closest && e.target.closest('.kpi, .card.glow');
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--mx', (e.clientX - r.left) + 'px');
    el.style.setProperty('--my', (e.clientY - r.top) + 'px');
  }, { passive: true });

  /* ── Submit buttons: show a busy state once a form submits ────────── */
  document.querySelectorAll('form[data-busy]').forEach(form => {
    form.addEventListener('submit', e => {
      // Let page-level validation handlers run first; only go busy if the submit proceeds.
      setTimeout(() => {
        if (e.defaultPrevented) return;
        const btn = form.querySelector('[type="submit"]');
        if (!btn) return;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>' + (form.dataset.busy || 'Working…');
      }, 0);
    });
  });
})();
