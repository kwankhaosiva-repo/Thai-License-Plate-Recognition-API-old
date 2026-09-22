/**
 * static/js/viewer.js
 * Shared image viewer (lightbox) with zoom / pan for the LPR dashboard.
 * Usage:
 *   LPRViewer.open({ src, title, subtitle });
 *   LPRViewer.attachZoomPan(containerEl, imgEl, fpsEl)  -> zoom/pan for stream viewers
 * Any element with [data-lpr-viewer] attribute + data-src will open the lightbox automatically.
 */
(function () {
  'use strict';

  // Build lightbox DOM once
  let root = null;

  function build() {
    if (root) return root;
    root = document.createElement('div');
    root.id = 'lpr-lightbox';
    root.className = 'lpr-lightbox';
    root.innerHTML = `
      <div class="lpr-lightbox-header">
        <div class="lpr-lightbox-titlewrap">
          <span class="lpr-lightbox-title">-</span>
          <span class="lpr-lightbox-subtitle"></span>
        </div>
        <div class="lpr-lightbox-toolbar">
          <button type="button" data-act="zoomout" class="lpr-btn" title="ซูมออก (-)">−</button>
          <button type="button" data-act="zoomreset" class="lpr-btn lpr-zoom-level" title="รีเซ็ตซูม (0)">100%</button>
          <button type="button" data-act="zoomin" class="lpr-btn" title="ซูมเข้า (+)">+</button>
          <button type="button" data-act="fit" class="lpr-btn" title="พอดีจอ (0)">พอดีจอ</button>
          <button type="button" data-act="download" class="lpr-btn" title="บันทึกภาพ">บันทึกภาพ</button>
          <button type="button" data-act="close" class="lpr-btn lpr-btn-danger" title="ปิด (Esc)">ปิด</button>
        </div>
      </div>
      <div class="lpr-lightbox-body">
        <img class="lpr-lightbox-img" alt="Inspector" draggable="false" />
      </div>
      <div class="lpr-lightbox-hint">ลากเพื่อเลื่อน • สกรอลล์เพื่อซูม • ดับเบิลคลิกเพื่อซูม 2x • Esc เพื่อปิด</div>
    `;
    document.body.appendChild(root);

    root.querySelector('[data-act="close"]').addEventListener('click', close);
    root.querySelector('[data-act="zoomin"]').addEventListener('click', () => setZoom(zoom + 0.3));
    root.querySelector('[data-act="zoomout"]').addEventListener('click', () => setZoom(zoom - 0.3));
    root.querySelector('[data-act="zoomreset"]').addEventListener('click', () => setZoom(1));
    root.querySelector('[data-act="fit"]').addEventListener('click', () => setZoom(1));
    root.querySelector('[data-act="download"]').addEventListener('click', downloadImage);

    root.addEventListener('click', (e) => { if (e.target === root) close(); });

    const body = root.querySelector('.lpr-lightbox-body');
    const img = root.querySelector('.lpr-lightbox-img');

    body.addEventListener('wheel', (e) => {
      e.preventDefault();
      setZoom(zoom + (e.deltaY < 0 ? 0.2 : -0.2));
    }, { passive: false });

    img.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      setZoom(zoom > 1.05 ? 1 : 2);
    });

    body.addEventListener('mousedown', (e) => {
      if (zoom <= 1.02) return;
      drag = true;
      sx = e.clientX - px; sy = e.clientY - py;
      body.classList.add('dragging');
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!drag) return;
      px = e.clientX - sx; py = e.clientY - sy;
      apply();
    });
    window.addEventListener('mouseup', () => {
      drag = false;
      body.classList.remove('dragging');
    });
    return root;
  }

  let zoom = 1, px = 0, py = 0, drag = false, sx = 0, sy = 0, currentSrc = '';

  function apply() {
    if (!root) return;
    const img = root.querySelector('.lpr-lightbox-img');
    const body = root.querySelector('.lpr-lightbox-body');
    const lvl = root.querySelector('.lpr-zoom-level');
    img.style.transform = `translate(${px}px, ${py}px) scale(${zoom})`;
    lvl.textContent = `${Math.round(zoom * 100)}%`;
    body.classList.toggle('zoomed', zoom > 1.02);
  }

  function setZoom(z) {
    zoom = Math.max(0.4, Math.min(5, z));
    if (zoom <= 1.02) { px = 0; py = 0; }
    apply();
  }

  function downloadImage() {
    if (!currentSrc) return;
    const a = document.createElement('a');
    a.href = currentSrc;
    a.download = `lpr_capture_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function open(opts) {
    build();
    currentSrc = opts.src || '';
    const img = root.querySelector('.lpr-lightbox-img');
    img.src = currentSrc;
    root.querySelector('.lpr-lightbox-title').textContent = opts.title || 'ตรวจภาพ';
    root.querySelector('.lpr-lightbox-subtitle').textContent = opts.subtitle || '';
    zoom = 1; px = 0; py = 0; apply();
    root.style.display = 'flex';
  }

  function close() { if (root) root.style.display = 'none'; }

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && root && root.style.display === 'flex') close();
    if (!root || root.style.display !== 'flex') return;
    if (e.key === '+' || e.key === '=') setZoom(zoom + 0.25);
    if (e.key === '-' || e.key === '_') setZoom(zoom - 0.25);
    if (e.key === '0') setZoom(1);
  });

  // ---- Interactive zoom/pan for inline stream viewers ----
  function attachZoomPan(container, img, labelEl) {
    if (!container || !img || container.dataset.zoomPanBound) return;
    container.dataset.zoomPanBound = '1';

    let z = 1, x = 0, y = 0, dragging = false, ox = 0, oy = 0;

    const apply = () => {
      img.style.transform = `translate(${x}px, ${y}px) scale(${z})`;
      img.style.cursor = z > 1.02 ? 'grab' : 'zoom-in';
      container.classList.toggle('is-zoomed', z > 1.02);
      if (labelEl) labelEl.textContent = `${Math.round(z * 100)}%`;
    };

    const clamp = () => {
      // Limit pan so image cannot be lost off-screen
      const rect = container.getBoundingClientRect();
      const maxX = rect.width * (z - 1) / 2 + 40;
      const maxY = rect.height * (z - 1) / 2 + 40;
      x = Math.max(-maxX, Math.min(maxX, x));
      y = Math.max(-maxY, Math.min(maxY, y));
    };

    container.addEventListener('wheel', (e) => {
      if (!img.src || img.style.display === 'none') return;
      e.preventDefault();
      z = Math.max(1, Math.min(4, z + (e.deltaY < 0 ? 0.15 : -0.15)));
      if (z <= 1.02) { x = 0; y = 0; }
      clamp(); apply();
    }, { passive: false });

    container.addEventListener('mousedown', (e) => {
      if (z <= 1.02) return;
      dragging = true; ox = e.clientX - x; oy = e.clientY - y;
      img.style.cursor = 'grabbing';
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      x = e.clientX - ox; y = e.clientY - oy;
      clamp(); apply();
    });
    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      img.style.cursor = z > 1.02 ? 'grab' : 'zoom-in';
    });

    container.addEventListener('dblclick', (e) => {
      e.preventDefault();
      z = z > 1.02 ? 1 : 2;
      if (z === 1) { x = 0; y = 0; }
      clamp(); apply();
    });

    // Reset zoom when image source changes
    const resetObserver = new MutationObserver(() => { z = 1; x = 0; y = 0; apply(); });
    resetObserver.observe(img, { attributeFilter: ['src'] });

    apply();
  }

  // ---- Delegated opener for [data-lpr-viewer] elements ----
  document.addEventListener('click', (e) => {
    const t = e.target.closest('[data-lpr-viewer]');
    if (!t) return;
    const src = t.getAttribute('data-src') || (t.tagName === 'IMG' ? t.src : '');
    if (!src) return;
    e.preventDefault();
    open({
      src,
      title: t.getAttribute('data-title') || 'ตรวจภาพขยาย',
      subtitle: t.getAttribute('data-subtitle') || '',
    });
  });

  window.LPRViewer = { open, close, attachZoomPan };
})();
