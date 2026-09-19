/**
 * static/js/app.js
 * Client application controller for Thai License Plate Recognition Dashboard.
 * Handles:
 *  - Mode switching (Upload vs Live RTSP Stream)
 *  - Debug mode toggling (ON/OFF)
 *  - Drag & drop single/batch image and video upload
 *  - Batch carousel navigation
 *  - 3-stage visual pipeline breakdown rendering
 *  - Debug breakdown drawer with province probability distribution
 *  - Live MJPEG stream connection and real-time detection polling
 */

(function () {
  'use strict';

  // --- State Management ---
  const state = {
    mode: 'upload', // 'upload' | 'live'
    isDebug: false,
    batchResults: [],
    currentIndex: 0,
    isStreaming: false,
    streamPollTimer: null,
    lastUploadedFiles: null, // retained for debug re-upload when toggle is turned ON after upload
    currentVideoFile: null,
    videoStreamSource: null,
    videoStreamTimer: null,
    streamConfM1: 0.80,
  };


  // --- DOM Elements ---
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

  // Video Stream Simulation Elements
  const videoStreamPlayerWrapper = document.getElementById('videoStreamPlayerWrapper');
  const videoStreamViewer = document.getElementById('videoStreamViewer');
  const streamFpsBadge = document.getElementById('streamFpsBadge');
  const streamMotionStatusBadge = document.getElementById('streamMotionStatusBadge');
  const btnRestartVideoStream = document.getElementById('btnRestartVideoStream');
  const btnRunBatchVideo = document.getElementById('btnRunBatchVideo');
  const btnCloseVideoStream = document.getElementById('btnCloseVideoStream');

  // RTSP Elements
  const rtspInput = document.getElementById('rtspInput');
  const btnConnectStream = document.getElementById('btnConnectStream');
  const streamViewer = document.getElementById('streamViewer');
  const streamPlaceholder = document.getElementById('streamPlaceholder');
  const streamStatusPill = document.getElementById('streamStatusPill');

  // Pipeline Breakdown Elements
  const totalLatencyPill = document.getElementById('totalLatencyPill');
  
  // Stage 0: Raw
  const timeRaw = document.getElementById('timeRaw');
  const cropRaw = document.getElementById('cropRaw');
  const cropRawPlaceholder = document.getElementById('cropRawPlaceholder');
  const metaRes = document.getElementById('metaRes');
  const metaStatus = document.getElementById('metaStatus');

  // Stage 1: Model 1
  const timeM1 = document.getElementById('timeM1');
  const cropM1 = document.getElementById('cropM1');
  const cropM1Placeholder = document.getElementById('cropM1Placeholder');
  const confPlate = document.getElementById('confPlate');

  // Stage 2: Model 2
  const timeM2 = document.getElementById('timeM2');
  const cropChar = document.getElementById('cropChar');
  const cropProv = document.getElementById('cropProv');
  const confCharProv = document.getElementById('confCharProv');

  // Stage 3: Model 3
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

  // Stage Model Tags
  const tagModel1 = document.getElementById('tagModel1');
  const tagModel2 = document.getElementById('tagModel2');
  const tagModel3 = document.getElementById('tagModel3');

  // Debug Drawer
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

  // Debug Model Tags
  const dbgTagM1 = document.getElementById('dbgTagM1');
  const dbgTagDeskew = document.getElementById('dbgTagDeskew');
  const dbgTagM2 = document.getElementById('dbgTagM2');
  const dbgTagOcr = document.getElementById('dbgTagOcr');
  const dbgTagProv = document.getElementById('dbgTagProv');
  const dbgTagCharBox = document.getElementById('dbgTagCharBox');

  // --- Initialization ---
  function init() {
    setupModeTabs();
    setupDebugToggle();
    setupDropzone();
    setupRTSPStream();
    setupVideoStreamControls();
    setupModelsModal();
    fetchModelTags();
    // Sync debug state from checkbox on page load (in case browser restores checked state)
    if (debugToggle) {
      state.isDebug = debugToggle.checked;
    }
  }

  // --- Dynamic Model Tags Management ---
  async function fetchModelTags() {
    try {
      const resp = await fetch('/api/health');
      if (resp.ok) {
        const data = await resp.json();
        if (data.model_tags) {
          applyModelTags(data.model_tags);
        }
        // Sync debug toggle with server cfg.DEBUG_MODE on page load
        if (debugToggle && typeof data.debug_mode === 'boolean') {
          debugToggle.checked = data.debug_mode;
          state.isDebug = data.debug_mode;
        }
      }
    } catch (e) {
      console.warn('[Model Tags] Could not fetch initial model tags from /api/health:', e);
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
    if (dbgTagOcr) {
      dbgTagOcr.textContent = tags.ocr_engine || tags.ocr_ctc || 'OCR Engine';
    }
    if (dbgTagProv) {
      dbgTagProv.textContent = tags.province_classifier || tags.prov_thai || tags.province_thai || 'Province Model';
    }
    if (dbgTagCharBox) {
      const charBox = tags.char_box || 'YOLO11-Box';
      const charCls = tags.char_classifier || tags.char_class_thai || tags.char_classifier_thai || 'MobileNetV2';
      dbgTagCharBox.textContent = `${charBox} + ${charCls}`;
    }
  }

  // --- Mode Switching ---
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

  // --- Models Architecture Modal ---
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
              const file = parts[0].trim();
              const tag = parts.length > 1 ? parts[1].replace(')', '').trim() : '';
              return { file, tag };
            };

            const setField = (infoId, archId, rawVal, fallbackTag) => {
              const infoEl = document.getElementById(infoId);
              const archEl = document.getElementById(archId);
              const parsed = parseModel(rawVal);
              if (infoEl && parsed.file) infoEl.textContent = parsed.file;
              if (archEl) archEl.textContent = fallbackTag || parsed.tag || 'Neural Engine';
            };

            setField('mInfo1', 'mArch1', m.model_1, tags.model_1 || 'D-FINE Nano (MIT)');
            setField('mInfo15', 'mArch15', m.model_1_5, tags.model_1_5 || 'MobileNetV3-Small');
            setField('mInfo2', 'mArch2', m.model_2, tags.model_2 || 'D-FINE Nano (MIT)');
            setField('mInfo3a', 'mArch3a', m.model_3a_thai_char_box, tags.char_box || 'RF-DETR Base');
            setField('mInfo3aCls', 'mArch3aCls', m.model_3a_thai_char_classifier, tags.char_class_thai || 'MobileNetV2 (50 Classes)');
            setField('mInfo3aLaoCls', 'mArch3aLaoCls', m.model_3a_lao_char_classifier, tags.char_class_lao || 'MobileNetV2 (34 Classes)');
            setField('mInfo3aOcr', 'mArch3aOcr', m.model_3a_thai_ctc, tags.ocr_ctc || 'ResNetCRNN CTC');
            setField('mInfo3b', 'mArch3b', m.model_3b_thai, tags.prov_thai || 'ResNet18-Grayscale (77 Provinces)');
            setField('mInfo3bLao', 'mArch3bLao', m.model_3b_lao, tags.prov_lao || 'ResNet18-Grayscale (18 Provinces)');
          }
          if (data.device) {
            const devEl = document.getElementById('mDeviceTag');
            if (devEl) devEl.textContent = `⚡ Execution Device: ${data.device.toUpperCase()}`;
          }
        }
      } catch (err) {
        console.warn('Failed to refresh models info:', err);
      }
    }

    function closeModal() {
      modelsModal.style.display = 'none';
    }

    modelsPillBtn.addEventListener('click', openModal);
    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (dismissBtn) dismissBtn.addEventListener('click', closeModal);

    modelsModal.addEventListener('click', (e) => {
      if (e.target === modelsModal) {
        closeModal();
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modelsModal.style.display === 'flex') {
        closeModal();
      }
    });
  }

  // --- Debug Mode Toggle ---
  function setupDebugToggle() {
    debugToggle.addEventListener('change', async (e) => {
      state.isDebug = e.target.checked;
      console.log(`[Debug Mode] Toggled: ${state.isDebug ? 'ON' : 'OFF'}`);

      const current = state.batchResults[state.currentIndex];

      // If debug just turned ON and current result has no debug payload, re-process with debug=true
      if (state.isDebug && current && !current.debug && state.lastUploadedFiles && state.lastUploadedFiles.length > 0) {
        console.log('[Debug Mode] Re-uploading with debug=true to generate overlays...');
        metaStatus.textContent = 'Re-processing (Debug ON)...';
        metaStatus.style.color = 'var(--accent-amber)';
        await uploadImageFiles(state.lastUploadedFiles);
        return;
      }

      // Otherwise just re-render drawer with available data
      if (current) {
        renderDebugDrawer(current);
      }
    });
  }


  // --- Drag & Drop Setup ---
  function setupDropzone() {
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleSelectedFiles(e.dataTransfer.files);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleSelectedFiles(e.target.files);
      }
    });

    // Sample Plates Quick Test Buttons
    document.querySelectorAll('.btn-sample').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const url = btn.dataset.sample;
        const name = btn.dataset.name;
        try {
          const resp = await fetch(url);
          const blob = await resp.blob();
          const file = new File([blob], name, { type: 'image/jpeg' });
          await uploadImageFiles([file]);
        } catch (err) {
          console.error('Failed to load sample image:', err);
        }
      });
    });
  }

  // --- Handle Files (Single/Batch Images or Video) ---
  async function handleSelectedFiles(fileList) {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    // Check if uploaded file is a video
    const isVideo = files[0].type.startsWith('video/') || files[0].name.match(/\.(mp4|mov|avi|mkv)$/i);

    if (isVideo) {
      state.lastUploadedFiles = null; // videos don't support debug re-upload
      await uploadVideoFile(files[0]);
    } else {
      state.lastUploadedFiles = files; // save for debug re-upload
      await uploadImageFiles(files);
    }
  }


  // Upload Images (Batch or Single)
  async function uploadImageFiles(files) {
    stopVideoStreamSimulation(false);
    uploadCountPill.textContent = `${files.length} file${files.length > 1 ? 's' : ''}`;
    metaStatus.textContent = 'Processing...';
    metaStatus.style.color = 'var(--accent-amber)';

    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    formData.append('debug', state.isDebug ? 'true' : 'false');
    formData.append('conf_m1', '0.35');
    formData.append('conf_m2', '0.25');

    try {
      const resp = await fetch('/api/detect/image', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) throw new Error(`Server returned HTTP ${resp.status}`);
      const data = await resp.json();

      state.batchResults = data.results || [];
      state.currentIndex = 0;

      if (state.batchResults.length > 1) {
        renderBatchGallery();
      } else {
        batchGalleryWrapper.style.display = 'none';
      }

      if (state.batchResults.length > 0) {
        renderPipelineResult(state.batchResults[0]);
      }
    } catch (err) {
      console.error('[Upload Error]', err);
      metaStatus.textContent = 'Inference Error';
      metaStatus.style.color = 'var(--accent-red)';
      alert(`Inference failed: ${err.message}`);
    }
  }

  // --- Video Real-Time RTSP Stream Simulation ---
  async function uploadVideoFile(videoFile) {
    state.currentVideoFile = videoFile;
    uploadCountPill.textContent = '1 video';
    metaStatus.textContent = 'Initializing Real-Time Stream...';
    metaStatus.style.color = 'var(--accent-cyan)';

    const formData = new FormData();
    formData.append('file', videoFile);

    try {
      const resp = await fetch('/api/stream/upload_video', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) throw new Error(`Server returned HTTP ${resp.status}`);
      const data = await resp.json();

      state.videoStreamSource = data.stream_source;

      // Switch Dropzone to Live Video Stream Player
      if (dropzone) dropzone.style.display = 'none';
      if (batchGalleryWrapper) batchGalleryWrapper.style.display = 'none';
      if (videoStreamPlayerWrapper) videoStreamPlayerWrapper.style.display = 'block';

      if (streamFpsBadge && data.metadata) {
        streamFpsBadge.textContent = `${data.metadata.fps} FPS (${data.metadata.duration_sec}s)`;
      }

      if (videoStreamViewer) {
        videoStreamViewer.src = `/api/stream/mjpeg?source=${encodeURIComponent(data.stream_source)}&loop=true&debug=${state.isDebug ? 'true' : 'false'}&conf_m1=${state.streamConfM1}&t=${Date.now()}`;
      }

      metaStatus.textContent = 'RTSP Stream Mock Active';
      metaStatus.style.color = 'var(--accent-green)';

      // High-frequency polling to continuously update the 4-Stage visual breakdown cards on the right
      if (state.videoStreamTimer) clearInterval(state.videoStreamTimer);
      state.videoStreamTimer = setInterval(async () => {
        try {
          const detResp = await fetch('/api/stream/latest');
          if (detResp.ok) {
            const det = await detResp.json();
            if (det && det.detected) {
              renderPipelineResult(det);
              if (streamMotionStatusBadge) {
                streamMotionStatusBadge.textContent = '⚡ Car Active (Inferencing)';
                streamMotionStatusBadge.style.color = 'var(--accent-cyan)';
                streamMotionStatusBadge.style.background = 'rgba(0, 240, 255, 0.15)';
                streamMotionStatusBadge.style.borderColor = 'rgba(0, 240, 255, 0.35)';
              }
            } else if (streamMotionStatusBadge) {
              streamMotionStatusBadge.textContent = '🟢 Gate Idle (0ms)';
              streamMotionStatusBadge.style.color = '#34d399';
              streamMotionStatusBadge.style.background = 'rgba(16, 185, 129, 0.15)';
              streamMotionStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            }
          }
        } catch (e) {
          // ignore transient poll errors
        }
      }, 350);

    } catch (err) {
      console.error('[Video Stream Error]', err);
      metaStatus.textContent = 'Stream Init Failed';
      metaStatus.style.color = 'var(--accent-red)';
      alert(`Could not start real-time video simulation: ${err.message}`);
      stopVideoStreamSimulation(true);
    }
  }

  // Fallback: Batch Scan All Video Frames (Static extraction)
  async function runBatchVideoExtraction(videoFile) {
    uploadCountPill.textContent = '1 video';
    metaStatus.textContent = 'Scanning All Frames...';
    metaStatus.style.color = 'var(--accent-amber)';

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('debug', state.isDebug ? 'true' : 'false');
    formData.append('sample_rate', '2');
    formData.append('conf_m1', '0.80');

    try {
      const resp = await fetch('/api/detect/video', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) throw new Error(`Server returned HTTP ${resp.status}`);
      const data = await resp.json();

      state.batchResults = data.results || [];
      state.currentIndex = 0;

      if (state.batchResults.length > 1) {
        renderBatchGallery(true);
      } else {
        batchGalleryWrapper.style.display = 'none';
      }

      if (state.batchResults.length > 0) {
        renderPipelineResult(state.batchResults[0]);
      } else {
        metaStatus.textContent = 'No plates detected in batch scan';
        metaStatus.style.color = 'var(--accent-red)';
      }
    } catch (err) {
      console.error('[Batch Video Error]', err);
      metaStatus.textContent = 'Batch Scan Error';
      metaStatus.style.color = 'var(--accent-red)';
      alert(`Batch video extraction failed: ${err.message}`);
    }
  }

  function stopVideoStreamSimulation(resetDropzone = true) {
    if (state.videoStreamTimer) {
      clearInterval(state.videoStreamTimer);
      state.videoStreamTimer = null;
    }
    if (videoStreamViewer) {
      videoStreamViewer.src = '';
    }
    if (videoStreamPlayerWrapper) {
      videoStreamPlayerWrapper.style.display = 'none';
    }
    if (resetDropzone && dropzone) {
      dropzone.style.display = 'block';
      uploadCountPill.textContent = '0 files';
      metaStatus.textContent = 'Ready';
      metaStatus.style.color = 'var(--accent-green)';
    }
    state.videoStreamSource = null;
  }

  function setupVideoStreamControls() {
    if (btnRestartVideoStream) {
      btnRestartVideoStream.addEventListener('click', () => {
        if (!state.videoStreamSource || !videoStreamViewer) return;
        videoStreamViewer.src = `/api/stream/mjpeg?source=${encodeURIComponent(state.videoStreamSource)}&loop=true&debug=${state.isDebug ? 'true' : 'false'}&conf_m1=${state.streamConfM1}&t=${Date.now()}`;
      });
    }

    const streamConfSlider = document.getElementById('streamConfSlider');
    const streamConfVal = document.getElementById('streamConfVal');
    if (streamConfSlider && streamConfVal) {
      streamConfSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        streamConfVal.textContent = `${val}%`;
        state.streamConfM1 = (parseFloat(val) / 100).toFixed(2);
      });
      streamConfSlider.addEventListener('change', () => {
        if (!state.videoStreamSource || !videoStreamViewer) return;
        videoStreamViewer.src = `/api/stream/mjpeg?source=${encodeURIComponent(state.videoStreamSource)}&loop=true&debug=${state.isDebug ? 'true' : 'false'}&conf_m1=${state.streamConfM1}&t=${Date.now()}`;
      });
    }

    if (btnRunBatchVideo) {
      btnRunBatchVideo.addEventListener('click', () => {
        if (!state.currentVideoFile) return;
        runBatchVideoExtraction(state.currentVideoFile);
      });
    }

    if (btnCloseVideoStream) {
      btnCloseVideoStream.addEventListener('click', () => {
        stopVideoStreamSimulation(true);
      });
    }
  }

  // --- Render Batch Carousel ---
  function renderBatchGallery(isVideo = false) {
    batchGalleryWrapper.style.display = 'flex';
    batchGallery.innerHTML = '';

    state.batchResults.forEach((res, idx) => {
      const thumb = document.createElement('img');
      thumb.className = `batch-thumb ${idx === state.currentIndex ? 'active' : ''}`;
      thumb.src = (res.crops && (res.crops.plate_rectified || res.crops.raw)) || '';
      thumb.title = isVideo
        ? `Time: ${res.timestamp_sec}s | ${res.plate_text || 'No plate'}`
        : `${res.filename || 'Image ' + (idx + 1)} | ${res.plate_text || 'No plate'}`;

      thumb.addEventListener('click', () => {
        state.currentIndex = idx;
        document.querySelectorAll('.batch-thumb').forEach((el) => el.classList.remove('active'));
        thumb.classList.add('active');
        renderPipelineResult(res);
      });

      batchGallery.appendChild(thumb);
    });
  }

  // --- Render 3-Stage Pipeline Breakdown ---
  function renderPipelineResult(res) {
    if (!res) return;

    // Total Latency
    const totalMs = res.timing ? res.timing.total_ms : '--';
    totalLatencyPill.textContent = `Latency: ${totalMs} ms`;

    // Stage 0: Raw
    if (res.crops && res.crops.raw) {
      cropRaw.src = res.crops.raw;
      cropRaw.style.display = 'block';
      cropRawPlaceholder.style.display = 'none';
    } else {
      cropRaw.style.display = 'none';
      cropRawPlaceholder.style.display = 'flex';
    }
    timeRaw.textContent = `${totalMs} ms`;
    metaStatus.textContent = res.detected ? 'Plate Detected' : 'No Target';
    metaStatus.style.color = res.detected ? 'var(--accent-green)' : 'var(--accent-red)';
    metaRes.textContent = res.detected ? '320x160 Warp' : '--';

    // If detection failed
    if (!res.detected) {
      if (cropM1) cropM1.style.display = 'none';
      if (cropM1Placeholder) cropM1Placeholder.style.display = 'flex';
      if (cropChar) cropChar.style.display = 'none';
      if (cropProv) cropProv.style.display = 'none';
      if (resultPlate) resultPlate.textContent = 'NO PLATE';
      if (resultProvince) resultProvince.textContent = 'None';
      if (resultBadge) {
        resultBadge.textContent = 'NOT DETECTED';
        resultBadge.style.color = 'var(--accent-red)';
        resultBadge.style.background = 'rgba(239, 68, 68, 0.15)';
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

    // Dynamic Model Tags for this detection result (Thai vs Lao models)
    if (res.model_tags) {
      applyModelTags(res.model_tags);
    }

    // Country Classifier (Model 1.5)
    const badgeCountry = document.getElementById('badgeCountry');
    const countryMeta = document.getElementById('countryMeta');
    const layoutMeta = document.getElementById('layoutMeta');
    const labelCharBox = document.getElementById('labelCharBox');
    const labelProvBox = document.getElementById('labelProvBox');

    if (res.country && badgeCountry) {
      const isThai = res.country === 'Thai';
      badgeCountry.className = `badge-country ${isThai ? 'thai' : 'laos'}`;
      badgeCountry.textContent = `${res.country_flag || ''} ${res.country.toUpperCase()}`;
      
      const cConf = res.country_confidence ? (res.country_confidence * 100).toFixed(1) + '%' : '99.9%';
      if (countryMeta) {
        countryMeta.textContent = `${res.country_flag || ''} ${res.country} (${cConf})`;
        countryMeta.style.color = isThai ? 'var(--accent-cyan)' : '#f87171';
      }
      
      if (layoutMeta) {
        layoutMeta.textContent = isThai ? 'Standard (Top Char)' : 'Inverted (Top Prov)';
        layoutMeta.style.color = isThai ? 'var(--accent-cyan)' : 'var(--accent-amber)';
      }

      if (labelCharBox && labelProvBox) {
        labelCharBox.textContent = isThai ? 'plate_char (Top)' : 'plate_char (Bottom)';
        labelProvBox.textContent = isThai ? 'province (Bottom)' : 'province (Top)';
      }
    }

    if (confCharProv && res.confidence) {
      const cC = (res.confidence.char_detection * 100).toFixed(1);
      const pC = (res.confidence.prov_detection * 100).toFixed(1);
      confCharProv.textContent = `${cC}% / ${pC}%`;
    }

    // Stage 1: Model 1
    if (res.crops && res.crops.plate_rectified) {
      cropM1.src = res.crops.plate_rectified;
      cropM1.style.display = 'block';
      cropM1Placeholder.style.display = 'none';
    }
    timeM1.textContent = `${res.timing.m1_ms} ms`;
    confPlate.textContent = `${(res.confidence.plate_detection * 100).toFixed(1)}%`;

    // Stage 2: Model 2
    if (res.crops && res.crops.char_crop) {
      cropChar.src = res.crops.char_crop;
      cropChar.style.display = 'block';
    }
    if (res.crops && res.crops.prov_crop) {
      cropProv.src = res.crops.prov_crop;
      cropProv.style.display = 'block';
    }
    timeM2.textContent = `${res.timing.m2_ms} ms`;

    // Stage 3: Model 3
    timeM3.textContent = `${res.timing.m3_ms} ms`;
    // Show '--' when Lao plate text wasn't resolved (empty string from server)
    const plateDisplay = (res.plate_text && res.plate_text.trim()) ? res.plate_text : '--';
    resultPlate.textContent = plateDisplay;
    resultProvince.textContent = res.province || '--';

    if (res.is_valid) {
      resultBadge.textContent = 'VALID FORMAT';
      resultBadge.style.color = 'var(--accent-green)';
      resultBadge.style.background = 'var(--accent-green-dim)';
      resultBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      resultBadge.textContent = 'UNSTANDARDIZED';
      resultBadge.style.color = 'var(--accent-amber)';
      resultBadge.style.background = 'var(--accent-amber-dim)';
      resultBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    }

    patternText.textContent = res.pattern_name || '--';
    confProvProb.textContent = `${(res.confidence.province_classification * 100).toFixed(1)}%`;

    // Render Alternative Candidate Pill
    if (altPlateContainer) {
      if (res.is_ambiguous && res.alternative_plate_text) {
        altPlateContainer.style.display = 'flex';
        if (altPlateText) altPlateText.textContent = res.alternative_plate_text;
        if (altPlateReason) {
          const c0 = (res.alternative_candidates && res.alternative_candidates.length > 0) ? res.alternative_candidates[0] : null;
          if (c0) {
            altPlateReason.textContent = `⚠️ Noise disambiguated: '${c0.primary}' over '${c0.alternative}' (${c0.margin_pct}% margin)`;
          } else {
            altPlateReason.textContent = '⚠️ Close-margin alternative candidate';
          }
        }
      } else {
        altPlateContainer.style.display = 'none';
      }
    }

    // Render DLT Truck Code Badge
    if (dltTruckContainer) {
      if (res.dlt_truck_code && res.dlt_truck_province) {
        dltTruckContainer.style.display = 'flex';
        if (dltTruckCode) dltTruckCode.textContent = res.dlt_truck_code;
        if (dltTruckProvince) dltTruckProvince.textContent = res.dlt_truck_province;
      } else {
        dltTruckContainer.style.display = 'none';
      }
    }

    // Render Debug Drawer
    renderDebugDrawer(res);
  }

  // --- Render Debug Inspection Drawer ---
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

    // Render Character Boxes Overlay & Text
    if (dbgCharBoxes && d.char_boxes_overlay) {
      dbgCharBoxes.src = d.char_boxes_overlay;
      if (cardDbgCharBoxes) cardDbgCharBoxes.style.display = 'block';
      if (dbgCharBoxesTitle && d.char_box_text) {
        dbgCharBoxesTitle.textContent = `Boxes: ${d.char_box_text}`;
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

    // Render Province Probabilities Bar Chart
    dbgProvBars.innerHTML = '';
    if (d.prov_top5 && d.prov_top5.length > 0) {
      d.prov_top5.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'prob-item';
        row.innerHTML = `
          <div class="prob-text">
            <span>${item.name}</span>
            <span>${item.prob}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${Math.min(100, Math.max(0, item.prob))}%;"></div>
          </div>
        `;
        dbgProvBars.appendChild(row);
      });
    } else {
      dbgProvBars.innerHTML = '<div class="empty-state">No distribution data</div>';
    }
  }

  // --- RTSP / Live Stream Controller ---
  function setupRTSPStream() {
    btnConnectStream.addEventListener('click', () => {
      if (state.isStreaming) {
        stopStream();
      } else {
        startStream();
      }
    });
  }

  function startStream() {
    const source = rtspInput.value.trim() || '0';
    console.log(`[RTSP Stream] Connecting to source: ${source}`);

    state.isStreaming = true;
    btnConnectStream.innerHTML = '<span>Disconnect</span>';
    btnConnectStream.classList.add('btn-danger');
    btnConnectStream.classList.remove('btn-primary');

    streamStatusPill.textContent = 'Live Streaming';
    streamStatusPill.style.color = 'var(--accent-green)';

    streamPlaceholder.style.display = 'none';
    streamViewer.style.display = 'block';

    // Set stream src to MJPEG endpoint with debug parameter
    const streamUrl = `/api/stream/mjpeg?source=${encodeURIComponent(source)}&debug=${state.isDebug ? 'true' : 'false'}`;
    streamViewer.src = streamUrl;

    // Start polling latest detection every 600ms to update pipeline breakdown cards
    if (state.streamPollTimer) clearInterval(state.streamPollTimer);
    state.streamPollTimer = setInterval(async () => {
      try {
        const resp = await fetch('/api/stream/latest');
        if (resp.ok) {
          const data = await resp.json();
          if (data && data.detected) {
            renderPipelineResult(data);
          }
        }
      } catch (e) {
        // Stream polling error ignored
      }
    }, 600);
  }

  function stopStream() {
    console.log('[RTSP Stream] Disconnecting');
    state.isStreaming = false;

    btnConnectStream.innerHTML = '<span>Connect</span>';
    btnConnectStream.classList.remove('btn-danger');
    btnConnectStream.classList.add('btn-primary');

    streamStatusPill.textContent = 'Standby';
    streamStatusPill.style.color = 'var(--text-secondary)';

    streamViewer.src = '';
    streamViewer.style.display = 'none';
    streamPlaceholder.style.display = 'flex';

    if (state.streamPollTimer) {
      clearInterval(state.streamPollTimer);
      state.streamPollTimer = null;
    }
  }

  // Initialize on DOM load
  document.addEventListener('DOMContentLoaded', init);
})();
