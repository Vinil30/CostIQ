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
  document.querySelectorAll('[data-count]').forEach(countUp);

  /* ── Reveal on scroll ──────────────────────────────────────────────── */
  const reveals = document.querySelectorAll('[data-reveal]');
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
