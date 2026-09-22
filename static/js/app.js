/**
 * static/js/app.js
 * Client controller for the LPR Live Monitor page (Thai-first wording).
 * - Mode switching (ทดสอบภาพ/วิดีโอ vs กล้องสด RTSP)
 * - Debug (โหมดตรวจสอบละเอียด) toggle
 * - Drag & drop single/batch image & video upload
 * - 4-stage visual pipeline rendering
 * - MJPEG live stream + polling
 * - Zoom & pan for stream viewers (via LPRViewer.attachZoomPan)
 */
(function () {
  'use strict';

  const t = (key, fallback) => (window.I18N ? I18N.t(key, fallback) : fallback || key);

  // --- State ---
  const state = {
    mode: 'upload',
    isDebug: false,
    batchResults: [],
    currentIndex: 0,
    isStreaming: false,
    streamPollTimer: null,
    lastUploadedFiles: null,
    currentVideoFile: null,
    videoStreamSource: null,
    videoStreamTimer: null,
    streamConfM1: 0.80,
  };

  // --- DOM ---
  const tabUpload = document.getElementById('tabUpload');
  const tabLive = document.getElementById('tabLive');
  const uploadPanel = document.getElementById('uploadPanel');
  const livePanel = document.getElementById('livePanel');
  const debugToggle = document.getElementById('debugToggle');

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const uploadCountPill = document.getElementById('uploadCountPill');
  const batchGalleryWrapper = document.getElementById('batchGalleryWrapper');
  const batchGallery = document.getElementById('batchGallery');

  const videoStreamPlayerWrapper = document.getElementById('videoStreamPlayerWrapper');
  const videoStreamViewer = document.getElementById('videoStreamViewer');
  const videoStreamContainer = document.getElementById('videoStreamContainer');
  const videoStreamZoomLabel = document.getElementById('videoStreamZoomLabel');
  const streamFpsBadge = document.getElementById('streamFpsBadge');
  const streamMotionStatusBadge = document.getElementById('streamMotionStatusBadge');
  const btnRestartVideoStream = document.getElementById('btnRestartVideoStream');
  const btnRunBatchVideo = document.getElementById('btnRunBatchVideo');
  const btnCloseVideoStream = document.getElementById('btnCloseVideoStream');

  const rtspInput = document.getElementById('rtspInput');
  const btnConnectStream = document.getElementById('btnConnectStream');
  const rtspStreamContainer = document.getElementById('rtspStreamContainer');
  const rtspZoomLabel = document.getElementById('rtspZoomLabel');
  const streamViewer = document.getElementById('streamViewer');
  const streamPlaceholder = document.getElementById('streamPlaceholder');
  const streamStatusPill = document.getElementById('streamStatusPill');

  const totalLatencyPill = document.getElementById('totalLatencyPill');
  const timeRaw = document.getElementById('timeRaw');
  const cropRaw = document.getElementById('cropRaw');
  const cropRawPlaceholder = document.getElementById('cropRawPlaceholder');
  const metaRes = document.getElementById('metaRes');
  const metaStatus = document.getElementById('metaStatus');

  const timeM1 = document.getElementById('timeM1');
  const cropM1 = document.getElementById('cropM1');
  const cropM1Placeholder = document.getElementById('cropM1Placeholder');
  const confPlate = document.getElementById('confPlate');

  const timeM2 = document.getElementById('timeM2');
  const cropChar = document.getElementById('cropChar');
  const cropProv = document.getElementById('cropProv');
  const confCharProv = document.getElementById('confCharProv');

  const timeM3 = document.getElementById('timeM3');
  const resultPlate = document.getElementById('resultPlate');
  const resultProvince = document.getElementById('resultProvince');
  const resultBadge = document.getElementById('resultBadge');
  const patternText = document.getElementById('patternText');
  const confProvProb = document.getElementById('confProvProb');
  const altPlateContainer = document.getElementById('altPlateContainer');
  const altPlateText = document.getElementById('altPlateText');
  const altPlateReason = document.getElementById('altPlateReason');
  const dltTruckContainer = document.getElementById('dltTruckContainer');
  const dltTruckCode = document.getElementById('dltTruckCode');
  const dltTruckProvince = document.getElementById('dltTruckProvince');

  const tagModel1 = document.getElementById('tagModel1');
  const tagModel2 = document.getElementById('tagModel2');
  const tagModel3 = document.getElementById('tagModel3');

  const debugDrawer = document.getElementById('debugDrawer');
  const dbgPoly = document.getElementById('dbgPoly');
  const dbgDeskew = document.getElementById('dbgDeskew');
  const dbgComp = document.getElementById('dbgComp');
  const dbgOcr = document.getElementById('dbgOcr');
  const dbgProvBars = document.getElementById('dbgProvBars');
  const dbgCharBoxes = document.getElementById('dbgCharBoxes');
  const dbgCharBoxesTitle = document.getElementById('dbgCharBoxesTitle');
  const dbgCharBoxesNote = document.getElementById('dbgCharBoxesNote');
  const cardDbgCharBoxes = document.getElementById('cardDbgCharBoxes');

  const dbgTagM1 = document.getElementById('dbgTagM1');
  const dbgTagDeskew = document.getElementById('dbgTagDeskew');
  const dbgTagM2 = document.getElementById('dbgTagM2');
  const dbgTagOcr = document.getElementById('dbgTagOcr');
  const dbgTagProv = document.getElementById('dbgTagProv');
  const dbgTagCharBox = document.getElementById('dbgTagCharBox');

  // --- Toast (replaces alert for non-blocking feedback) ---
  function toast(message, type = 'info', duration = 3500) {
    let el = document.getElementById('lpr-toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'lpr-toast';
      el.className = 'lpr-toast';
      document.body.appendChild(el);
    }
    el.textContent = message;
    el.className = `lpr-toast show ${type === 'error' ? 'error' : type === 'success' ? 'success' : ''}`;
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.className = 'lpr-toast'; }, duration);
  }

  window.LPRToast = toast;

  // --- Init ---
  function init() {
    setupModeTabs();
    setupDebugToggle();
    setupSidebarNav();
    setupDropzone();
    setupRTSPStream();
    setupVideoStreamControls();
    setupModelsModal();
    fetchModelTags();

    // Interactive zoom & pan on both stream viewers
    if (window.LPRViewer) {
      LPRViewer.attachZoomPan(videoStreamContainer, videoStreamViewer, videoStreamZoomLabel);
      LPRViewer.attachZoomPan(rtspStreamContainer, streamViewer, rtspZoomLabel);
    }

    if (debugToggle) state.isDebug = debugToggle.checked;
  }

  // --- Sidebar nav buttons ---
  function setupSidebarNav() {
    const navModels = document.getElementById('navModelsBtn');
    if (navModels) navModels.addEventListener('click', () => {
      const pill = document.getElementById('modelsPillBtn');
      if (pill) pill.click();
    });
  }

  // --- Model tags ---
  async function fetchModelTags() {
    try {
      const resp = await fetch('/api/health');
      if (resp.ok) {
        const data = await resp.json();
        if (data.model_tags) applyModelTags(data.model_tags);
        if (data.device) {
          const devEl = document.getElementById('mDeviceTag');
          if (devEl) devEl.textContent = `${t('mm_device', 'Device:')} ${String(data.device).toUpperCase()}`;
        }
        // Models loaded count in sidebar badge + topbar
        const total = data.models_total || Object.keys(data.models || {}).length || 9;
        const loadedTxt = `${data.models_loaded || total}/${total}`;
        const badge = document.getElementById('navModelsBadge');
        const loadedText = document.getElementById('modelsLoadedText');
        if (badge) badge.textContent = loadedTxt;
        if (loadedText) loadedText.textContent = loadedTxt;
        if (typeof data.debug_mode === 'boolean' && debugToggle) {
          debugToggle.checked = data.debug_mode;
          state.isDebug = data.debug_mode;
        }
      }
    } catch (e) {
      console.warn('[Model Tags] Could not fetch /api/health:', e);
      const badge = document.getElementById('navModelsBadge');
      if (badge) { badge.textContent = '--'; badge.style.color = 'var(--text-muted)'; }
    }
  }

  function applyModelTags(tags) {
    if (!tags) return;
    if (tagModel1 && tags.model_1) tagModel1.textContent = tags.model_1;
    if (tagModel2 && tags.model_2) tagModel2.textContent = tags.model_2;
    if (tagModel3) {
      const charBox = tags.char_box ? `${tags.char_box} + ` : '';
      const charCls = tags.char_classifier || tags.char_class_thai || tags.char_classifier_thai || 'MobileNetV2';
      tagModel3.textContent = `${charBox}${charCls}`;
    }
    if (dbgTagM1 && tags.model_1) dbgTagM1.textContent = tags.model_1;
    if (dbgTagDeskew) dbgTagDeskew.textContent = 'Homography Deskew';
    if (dbgTagM2 && tags.model_2) dbgTagM2.textContent = tags.model_2;
    if (dbgTagOcr) dbgTagOcr.textContent = tags.ocr_engine || tags.ocr_ctc || 'OCR';
    if (dbgTagProv) dbgTagProv.textContent = tags.province_classifier || tags.prov_thai || tags.province_thai || 'Province Model';
    if (dbgTagCharBox) {
      const charBox = tags.char_box || 'Char Box';
      const charCls = tags.char_classifier || tags.char_class_thai || tags.char_classifier_thai || 'MobileNetV2';
      dbgTagCharBox.textContent = `${charBox} + ${charCls}`;
    }
  }

  // --- Mode switching ---
  function setupModeTabs() {
    tabUpload.addEventListener('click', () => {
      state.mode = 'upload';
      tabUpload.classList.add('active');
      tabLive.classList.remove('active');
      uploadPanel.style.display = 'block';
      livePanel.style.display = 'none';
      if (state.isStreaming) stopStream();
    });

    tabLive.addEventListener('click', () => {
      state.mode = 'live';
      tabLive.classList.add('active');
      tabUpload.classList.remove('active');
      uploadPanel.style.display = 'none';
      livePanel.style.display = 'block';
      stopVideoStreamSimulation(true);
    });
  }

  // --- Models modal ---
  function setupModelsModal() {
    const modelsPillBtn = document.getElementById('modelsPillBtn');
    const modelsModal = document.getElementById('modelsModal');
    const closeBtn = document.getElementById('closeModelsModalBtn');
    const dismissBtn = document.getElementById('modalDismissBtn');
    if (!modelsPillBtn || !modelsModal) return;

    async function openModal() {
      modelsModal.style.display = 'flex';
      try {
        const resp = await fetch('/api/health');
        if (resp.ok) {
          const data = await resp.json();
          if (data.models) {
            const m = data.models;
            const tags = data.model_tags || {};

            const parseModel = (raw) => {
              if (!raw) return { file: '-', tag: '' };
              const parts = raw.split(' (');
              return {
                file: parts[0].trim(),
                tag: parts.length > 1 ? parts[1].replace(')', '').trim() : '',
              };
            };

            const setField = (infoId, archId, rawVal, fallbackTag) => {
              const infoEl = document.getElementById(infoId);
              const archEl = document.getElementById(archId);
              const parsed = parseModel(rawVal);
              if (infoEl && parsed.file) infoEl.textContent = parsed.file;
              if (archEl) archEl.textContent = fallbackTag || parsed.tag || 'Neural Engine';
            };

            setField('mInfo1', 'mArch1', m.model_1, tags.model_1 || 'D-FINE Nano');
            setField('mInfo15', 'mArch15', m.model_1_5, tags.model_1_5 || 'MobileNetV3-Small');
            setField('mInfo2', 'mArch2', m.model_2, tags.model_2 || 'D-FINE Nano');
            setField('mInfo3a', 'mArch3a', m.model_3a_thai_char_box, tags.char_box || 'RF-DETR Base');
            setField('mInfo3aCls', 'mArch3aCls', m.model_3a_thai_char_classifier, tags.char_class_thai || 'MobileNetV2 (50 คลาส)');
            setField('mInfo3aLaoCls', 'mArch3aLaoCls', m.model_3a_lao_char_classifier, tags.char_class_lao || 'MobileNetV2 (34 คลาส)');
            setField('mInfo3aOcr', 'mArch3aOcr', m.model_3a_thai_ctc, tags.ocr_ctc || 'ResNetCRNN CTC');
            setField('mInfo3b', 'mArch3b', m.model_3b_thai, tags.prov_thai || 'ResNet18 (77 จังหวัด)');
            setField('mInfo3bLao', 'mArch3bLao', m.model_3b_lao, tags.prov_lao || 'ResNet18 (18 จังหวัด)');
          }
          if (data.device) {
            const devEl = document.getElementById('mDeviceTag');
            if (devEl) devEl.textContent = `${t('mm_device', 'Device:')} ${String(data.device).toUpperCase()}`;
          }
        }
      } catch (err) {
        console.warn('Failed to refresh models info:', err);
      }
    }

    function closeModal() { modelsModal.style.display = 'none'; }

    modelsPillBtn.addEventListener('click', openModal);
    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (dismissBtn) dismissBtn.addEventListener('click', closeModal);
    modelsModal.addEventListener('click', (e) => { if (e.target === modelsModal) closeModal(); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modelsModal.style.display === 'flex') closeModal();
    });
  }

  // --- Debug toggle ---
  function setupDebugToggle() {
    debugToggle.addEventListener('change', async (e) => {
      state.isDebug = e.target.checked;

      const current = state.batchResults[state.currentIndex];

      // Re-process with debug payload when turning ON without debug data
      if (state.isDebug && current && !current.debug && state.lastUploadedFiles && state.lastUploadedFiles.length > 0) {
        metaStatus.textContent = t('toast_reproc', 'Re-processing (debug mode)...');
        metaStatus.style.color = 'var(--accent-amber)';
        await uploadImageFiles(state.lastUploadedFiles);
        return;
      }

      if (current) renderDebugDrawer(current);
    });
  }

  // --- Drag & drop ---
  function setupDropzone() {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleSelectedFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) handleSelectedFiles(e.target.files);
    });
  }

  // --- File handling ---
  async function handleSelectedFiles(fileList) {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    const isVideo = files[0].type.startsWith('video/') || files[0].name.match(/\.(mp4|mov|avi|mkv)$/i);
    if (isVideo) {
      state.lastUploadedFiles = null;
      await uploadVideoFile(files[0]);
    } else {
      state.lastUploadedFiles = files;
      await uploadImageFiles(files);
    }
  }

  async function uploadImageFiles(files) {
    stopVideoStreamSimulation(false);
    uploadCountPill.innerHTML = files.length > 1 ? `${files.length} <span>${t('files_unit', 'files')}</span>` : `1 <span>${t('files_unit', 'files')}</span>`;
    metaStatus.textContent = t('processing_status', 'Processing...');
    metaStatus.style.color = 'var(--accent-amber)';

    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    formData.append('debug', state.isDebug ? 'true' : 'false');
    formData.append('conf_m1', '0.35');
    formData.append('conf_m2', '0.25');

    try {
      const resp = await fetch('/api/detect/image', { method: 'POST', body: formData });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      state.batchResults = data.results || [];
      state.currentIndex = 0;

      if (state.batchResults.length > 1) renderBatchGallery();
      else batchGalleryWrapper.style.display = 'none';

      if (state.batchResults.length > 0) renderPipelineResult(state.batchResults[0]);
    } catch (err) {
      console.error('[Upload Error]', err);
      metaStatus.textContent = 'เกิดข้อผิดพลาด';
      metaStatus.style.color = 'var(--accent-red)';
      toast(`${t('toast_proc_fail', 'Processing failed')}: ${err.message}`, 'error');
    }
  }

  // --- Video real-time simulation ---
  async function uploadVideoFile(videoFile) {
    state.currentVideoFile = videoFile;
    uploadCountPill.textContent = t('one_video', '1 video');
    metaStatus.textContent = t('toast_video_start', 'Starting video processing...');
    metaStatus.style.color = 'var(--accent)';

    const formData = new FormData();
    formData.append('file', videoFile);

    try {
      const resp = await fetch('/api/stream/upload_video', { method: 'POST', body: formData });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      state.videoStreamSource = data.stream_source;

      dropzone.style.display = 'none';
      batchGalleryWrapper.style.display = 'none';
      videoStreamPlayerWrapper.style.display = 'block';

      if (streamFpsBadge && data.metadata) {
        streamFpsBadge.textContent = `${data.metadata.fps} FPS • ${data.metadata.duration_sec}s`;
      }

      setVideoStreamSrc();
      metaStatus.textContent = 'กำลังประมวลผลวิดีโอแบบเรียลไทม์';
      metaStatus.style.color = 'var(--accent-green)';

      if (state.videoStreamTimer) clearInterval(state.videoStreamTimer);
      state.videoStreamTimer = setInterval(async () => {
        try {
          const detResp = await fetch('/api/stream/latest');
          if (detResp.ok) {
            const det = await detResp.json();
            if (det && det.detected) {
              renderPipelineResult(det);
              if (streamMotionStatusBadge) {
                streamMotionStatusBadge.textContent = t('vstream_detecting', 'Reading plate');
                streamMotionStatusBadge.style.color = 'var(--accent)';
                streamMotionStatusBadge.style.background = 'var(--accent-dim)';
                streamMotionStatusBadge.style.borderColor = 'rgba(56, 189, 248, 0.35)';
              }
            } else if (streamMotionStatusBadge) {
              streamMotionStatusBadge.textContent = t('vstream_idle', 'Gate idle');
              streamMotionStatusBadge.style.color = '#34d399';
              streamMotionStatusBadge.style.background = 'var(--accent-green-dim)';
              streamMotionStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            }
          }
        } catch (e) { /* transient poll error */ }
      }, 350);
    } catch (err) {
      console.error('[Video Stream Error]', err);
      metaStatus.textContent = t('err_status', 'Error');
      metaStatus.style.color = 'var(--accent-red)';
      toast(`${t('toast_video_fail', 'Failed to start video')}: ${err.message}`, 'error');
      stopVideoStreamSimulation(true);
    }
  }

  function buildStreamUrl() {
    return `/api/stream/mjpeg?source=${encodeURIComponent(state.videoStreamSource)}&loop=true&debug=${state.isDebug ? 'true' : 'false'}&conf_m1=${state.streamConfM1}&t=${Date.now()}`;
  }

  function setVideoStreamSrc() {
    if (videoStreamViewer) videoStreamViewer.src = buildStreamUrl();
  }

  async function runBatchVideoExtraction(videoFile) {
    uploadCountPill.textContent = t('one_video', '1 video');
    metaStatus.textContent = t('toast_scanning', 'Scanning all frames...');
    metaStatus.style.color = 'var(--accent-amber)';

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('debug', state.isDebug ? 'true' : 'false');
    formData.append('sample_rate', '2');
    formData.append('conf_m1', '0.80');

    try {
      const resp = await fetch('/api/detect/video', { method: 'POST', body: formData });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      state.batchResults = data.results || [];
      state.currentIndex = 0;

      if (state.batchResults.length > 1) renderBatchGallery(true);
      else batchGalleryWrapper.style.display = 'none';

      if (state.batchResults.length > 0) renderPipelineResult(state.batchResults[0]);
      else {
        metaStatus.textContent = t('not_found_status', 'No plate');
        metaStatus.style.color = 'var(--accent-red)';
        toast(t('toast_noplate_video', 'No plates found in this video'), 'error');
      }
    } catch (err) {
      console.error('[Batch Video Error]', err);
      metaStatus.textContent = t('err_status', 'Error');
      metaStatus.style.color = 'var(--accent-red)';
      toast(`${t('toast_scan_fail', 'Video scan failed')}: ${err.message}`, 'error');
    }
  }

  function stopVideoStreamSimulation(resetDropzone = true) {
    if (state.videoStreamTimer) { clearInterval(state.videoStreamTimer); state.videoStreamTimer = null; }
    if (videoStreamViewer) videoStreamViewer.src = '';
    if (videoStreamPlayerWrapper) videoStreamPlayerWrapper.style.display = 'none';
    if (resetDropzone && dropzone) {
      dropzone.style.display = 'flex';
      uploadCountPill.innerHTML = `0 <span>${t('files_unit', 'files')}</span>`;
      metaStatus.textContent = t('ready_status', 'Ready');
      metaStatus.style.color = 'var(--accent-green)';
    }
    state.videoStreamSource = null;
  }

  function setupVideoStreamControls() {
    if (btnRestartVideoStream) {
      btnRestartVideoStream.addEventListener('click', () => {
        if (!state.videoStreamSource) return;
        setVideoStreamSrc();
      });
    }

    const streamConfSlider = document.getElementById('streamConfSlider');
    const streamConfVal = document.getElementById('streamConfVal');
    if (streamConfSlider && streamConfVal) {
      streamConfSlider.addEventListener('input', (e) => {
        streamConfVal.textContent = `${e.target.value}%`;
        state.streamConfM1 = (parseFloat(e.target.value) / 100).toFixed(2);
      });
      streamConfSlider.addEventListener('change', () => {
        if (!state.videoStreamSource) return;
        setVideoStreamSrc();
      });
    }

    if (btnRunBatchVideo) {
      btnRunBatchVideo.addEventListener('click', () => {
        if (state.currentVideoFile) runBatchVideoExtraction(state.currentVideoFile);
      });
    }

    if (btnCloseVideoStream) {
      btnCloseVideoStream.addEventListener('click', () => stopVideoStreamSimulation(true));
    }
  }

  // --- Batch gallery ---
  function renderBatchGallery(isVideo = false) {
    batchGalleryWrapper.style.display = 'flex';
    batchGallery.innerHTML = '';

    state.batchResults.forEach((res, idx) => {
      const thumb = document.createElement('img');
      thumb.className = `batch-thumb ${idx === state.currentIndex ? 'active' : ''}`;
      thumb.src = (res.crops && (res.crops.plate_rectified || res.crops.raw)) || '';
      thumb.title = isVideo
        ? `${res.timestamp_sec}s • ${res.plate_text || t('no_plate', 'NO PLATE')}`
        : `${res.filename || `#${idx + 1}`} • ${res.plate_text || t('no_plate', 'NO PLATE')}`;

      thumb.addEventListener('click', () => {
        state.currentIndex = idx;
        document.querySelectorAll('.batch-thumb').forEach((el) => el.classList.remove('active'));
        thumb.classList.add('active');
        renderPipelineResult(res);
      });

      batchGallery.appendChild(thumb);
    });
  }

  // --- Pipeline rendering ---
  function renderPipelineResult(res) {
    if (!res) return;

    const totalMs = res.timing ? res.timing.total_ms : '--';
    totalLatencyPill.innerHTML = `เวลาประมวลผล: <span class="mono">${totalMs} ms</span>`;

    // Stage 0
    if (res.crops && res.crops.raw) {
      cropRaw.src = res.crops.raw;
      cropRaw.style.display = 'block';
      cropRaw.style.cursor = 'zoom-in';
      cropRawPlaceholder.style.display = 'none';
    } else {
      cropRaw.style.display = 'none';
      cropRawPlaceholder.style.display = 'flex';
    }
    timeRaw.textContent = `${totalMs} ms`;
    metaStatus.textContent = res.detected ? t('plate_found', 'Plate detected') : t('not_found_status', 'No plate');
    metaStatus.style.color = res.detected ? 'var(--accent-green)' : 'var(--accent-red)';
    metaRes.textContent = res.detected ? '320x160' : '--';

    if (!res.detected) {
      if (cropM1) cropM1.style.display = 'none';
      if (cropM1Placeholder) cropM1Placeholder.style.display = 'flex';
      if (cropChar) cropChar.style.display = 'none';
      if (cropProv) cropProv.style.display = 'none';
      if (resultPlate) resultPlate.textContent = t('no_plate', 'NO PLATE');
      if (resultProvince) resultProvince.textContent = '—';
      if (resultBadge) {
        resultBadge.textContent = t('no_plate', 'NO PLATE');
        resultBadge.style.color = '#f87171';
        resultBadge.style.background = 'var(--accent-red-dim)';
        resultBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      }
      if (patternText) patternText.textContent = '--';
      if (confPlate) confPlate.textContent = '--';
      if (confCharProv) confCharProv.textContent = '--';
      if (confProvProb) confProvProb.textContent = '--';
      if (altPlateContainer) altPlateContainer.style.display = 'none';
      if (debugDrawer) debugDrawer.classList.remove('active');
      return;
    }

    if (res.model_tags) applyModelTags(res.model_tags);

    const badgeCountry = document.getElementById('badgeCountry');
    const countryMeta = document.getElementById('countryMeta');
    const layoutMeta = document.getElementById('layoutMeta');
    const labelCharBox = document.getElementById('labelCharBox');
    const labelProvBox = document.getElementById('labelProvBox');

    if (res.country && badgeCountry) {
      const isThai = res.country === 'Thai';
      badgeCountry.className = `badge-country ${isThai ? 'thai' : 'laos'}`;
      badgeCountry.textContent = `${res.country_flag || ''} ${isThai ? t('country_th', 'Thai') : t('country_lao', 'Lao')}`;

      const cConf = res.country_confidence ? (res.country_confidence * 100).toFixed(1) + '%' : '99.9%';
      if (countryMeta) {
        countryMeta.textContent = `${res.country_flag || ''} ${isThai ? t('country_th', 'Thai') : t('country_lao', 'Lao')} (${cConf})`;
        countryMeta.style.color = isThai ? 'var(--accent)' : '#f87171';
      }
      if (layoutMeta) {
        layoutMeta.textContent = isThai ? `${t('layout_std', 'Standard')}` : `${t('layout_inv', 'Inverted')}`;
        layoutMeta.style.color = isThai ? 'var(--accent)' : 'var(--accent-amber)';
      }
      if (labelCharBox && labelProvBox) {
        labelCharBox.textContent = isThai ? t('lbl_char_top', 'Chars (Top)') : t('lbl_char_bot', 'Chars (Bottom)');
        labelProvBox.textContent = isThai ? t('lbl_prov_bot', 'Province (Bottom)') : t('lbl_prov_top', 'Province (Top)');
      }
    }

    if (confCharProv && res.confidence) {
      const cC = (res.confidence.char_detection * 100).toFixed(1);
      const pC = (res.confidence.prov_detection * 100).toFixed(1);
      confCharProv.textContent = `${cC}% / ${pC}%`;
    }

    // Stage 1
    if (res.crops && res.crops.plate_rectified) {
      cropM1.src = res.crops.plate_rectified;
      cropM1.style.display = 'block';
      cropM1.style.cursor = 'zoom-in';
      cropM1Placeholder.style.display = 'none';
    }
    timeM1.textContent = `${res.timing.m1_ms} ms`;
    confPlate.textContent = `${(res.confidence.plate_detection * 100).toFixed(1)}%`;

    // Stage 2
    if (res.crops && res.crops.char_crop) { cropChar.src = res.crops.char_crop; cropChar.style.display = 'block'; }
    if (res.crops && res.crops.prov_crop) { cropProv.src = res.crops.prov_crop; cropProv.style.display = 'block'; }
    timeM2.textContent = `${res.timing.m2_ms} ms`;

    // Stage 3
    timeM3.textContent = `${res.timing.m3_ms} ms`;
    const plateDisplay = (res.plate_text && res.plate_text.trim()) ? res.plate_text : '--';
    resultPlate.textContent = plateDisplay;
    resultProvince.textContent = res.province || '--';

    if (res.is_valid) {
      resultBadge.textContent = t('res_valid', 'VALID FORMAT');
      resultBadge.style.color = '#34d399';
      resultBadge.style.background = 'var(--accent-green-dim)';
      resultBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else if (res.low_confidence) {
      resultBadge.textContent = t('res_lowconf', 'LOW CONFIDENCE — SAVED FOR REVIEW');
      resultBadge.style.color = '#fbbf24';
      resultBadge.style.background = 'var(--accent-amber-dim)';
      resultBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    } else {
      resultBadge.textContent = t('res_invalid', 'NON-STANDARD');
      resultBadge.style.color = '#fbbf24';
      resultBadge.style.background = 'var(--accent-amber-dim)';
      resultBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    }

    patternText.textContent = res.pattern_name || '--';
    confProvProb.textContent = `${(res.confidence.province_classification * 100).toFixed(1)}%`;

    // Alt candidate
    if (altPlateContainer) {
      if (res.is_ambiguous && res.alternative_plate_text) {
        altPlateContainer.style.display = 'flex';
        if (altPlateText) altPlateText.textContent = res.alternative_plate_text;
        if (altPlateReason) {
          const c0 = (res.alternative_candidates && res.alternative_candidates.length > 0) ? res.alternative_candidates[0] : null;
          altPlateReason.textContent = c0
            ? `อ่านเป็น '${c0.primary}' แทน '${c0.alternative}' (ต่างกัน ${c0.margin_pct}%)`
            : 'มีตัวเลือกอื่นที่ใกล้เคียงกัน';
        }
      } else {
        altPlateContainer.style.display = 'none';
      }
    }

    // DLT truck
    if (dltTruckContainer) {
      if (res.dlt_truck_code && res.dlt_truck_province) {
        dltTruckContainer.style.display = 'flex';
        if (dltTruckCode) dltTruckCode.textContent = res.dlt_truck_code;
        if (dltTruckProvince) dltTruckProvince.textContent = res.dlt_truck_province;
      } else {
        dltTruckContainer.style.display = 'none';
      }
    }

    renderDebugDrawer(res);
  }

  // --- Debug drawer ---
  function renderDebugDrawer(res) {
    if (!state.isDebug || !res.debug) {
      debugDrawer.classList.remove('active');
      return;
    }

    debugDrawer.classList.add('active');
    const d = res.debug;

    dbgPoly.src = d.poly_overlay || '';
    dbgDeskew.src = d.deskewed || d.raw_warp || '';
    dbgComp.src = d.comp_overlay || '';
    dbgOcr.src = d.char_enhanced || '';

    if (dbgCharBoxes && d.char_boxes_overlay) {
      dbgCharBoxes.src = d.char_boxes_overlay;
      if (cardDbgCharBoxes) cardDbgCharBoxes.style.display = 'block';
      if (dbgCharBoxesTitle && d.char_box_text) {
        dbgCharBoxesTitle.textContent = `${t('dbg_3c', 'Per-character boxes')}: ${d.char_box_text}`;
      }
      if (dbgCharBoxesNote) {
        if (d.char_box_note) {
          dbgCharBoxesNote.textContent = d.char_box_note;
          dbgCharBoxesNote.className = `debug-status-note status-${d.char_box_status || 'complete'}`;
          dbgCharBoxesNote.style.display = 'block';
        } else {
          dbgCharBoxesNote.style.display = 'none';
        }
      }
    } else if (cardDbgCharBoxes) {
      cardDbgCharBoxes.style.display = 'none';
    }

    dbgProvBars.innerHTML = '';
    if (d.prov_top5 && d.prov_top5.length > 0) {
      d.prov_top5.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'prob-item';
        row.innerHTML = `
          <div class="prob-text"><span>${item.name}</span><span>${item.prob}%</span></div>
          <div class="prob-track"><div class="prob-fill" style="width: ${Math.min(100, Math.max(0, item.prob))}%;"></div></div>
        `;
        dbgProvBars.appendChild(row);
      });
    } else {
      dbgProvBars.innerHTML = '<div class="empty-state">ยังไม่มีข้อมูล</div>';
    }
  }

  // --- RTSP live stream ---
  function setupRTSPStream() {
    btnConnectStream.addEventListener('click', () => {
      if (state.isStreaming) stopStream();
      else startStream();
    });
  }

  function startStream() {
    const source = rtspInput.value.trim() || '0';
    state.isStreaming = true;
    btnConnectStream.innerHTML = `<span>${t('rtsp_disconnect', 'Disconnect')}</span>`;
    btnConnectStream.classList.add('btn-danger');
    btnConnectStream.classList.remove('btn-primary');

    streamStatusPill.textContent = t('rtsp_streaming', 'Live streaming');
    streamStatusPill.style.color = 'var(--accent-green)';

    streamPlaceholder.style.display = 'none';
    streamViewer.style.display = 'block';
    streamViewer.src = `/api/stream/mjpeg?source=${encodeURIComponent(source)}&debug=${state.isDebug ? 'true' : 'false'}`;

    if (state.streamPollTimer) clearInterval(state.streamPollTimer);
    state.streamPollTimer = setInterval(async () => {
      try {
        const resp = await fetch('/api/stream/latest');
        if (resp.ok) {
          const data = await resp.json();
          if (data && data.detected) renderPipelineResult(data);
        }
      } catch (e) { /* polling error ignored */ }
    }, 600);
  }

  function stopStream() {
    state.isStreaming = false;
    btnConnectStream.innerHTML = `<span>${t('rtsp_connect', 'Connect')}</span>`;
    btnConnectStream.classList.remove('btn-danger');
    btnConnectStream.classList.add('btn-primary');

    streamStatusPill.textContent = t('rtsp_standby', 'Standby');
    streamStatusPill.style.color = 'var(--text-secondary)';

    streamViewer.src = '';
    streamViewer.style.display = 'none';
    streamPlaceholder.style.display = 'flex';

    if (state.streamPollTimer) { clearInterval(state.streamPollTimer); state.streamPollTimer = null; }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
