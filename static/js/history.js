/**
 * static/js/history.js
 * Frontend controller for Thai & Laos LPR Recognition History & Analytics Dashboard.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentPage = 1;
  const pageSize = 25;
  let sortBy = "timestamp";
  let sortOrder = "desc";
  let autoRefreshTimer = null;
  let isLive = true;

  // DOM Elements
  const statTotal = document.getElementById("statTotal");
  const statThai = document.getElementById("statThai");
  const statLao = document.getElementById("statLao");
  const statLatency = document.getElementById("statLatency");
  const statValidPct = document.getElementById("statValidPct");

  const filterDateQuick = document.getElementById("filterDateQuick");
  const customDateWrap = document.getElementById("customDateWrap");
  const filterDateFrom = document.getElementById("filterDateFrom");
  const filterDateTo = document.getElementById("filterDateTo");
  const filterCountry = document.getElementById("filterCountry");
  const filterStatus = document.getElementById("filterStatus");
  const filterSearch = document.getElementById("filterSearch");
  const btnApplyFilter = document.getElementById("btnApplyFilter");
  const btnToggleRefresh = document.getElementById("btnToggleRefresh");
  const lblLive = document.getElementById("lblLive");

  const historyTableBody = document.getElementById("historyTableBody");
  const lblShowing = document.getElementById("lblShowing");
  const lblTotal = document.getElementById("lblTotal");
  const lblCurrentPage = document.getElementById("lblCurrentPage");
  const lblTotalPages = document.getElementById("lblTotalPages");
  const btnPrevPage = document.getElementById("btnPrevPage");
  const btnNextPage = document.getElementById("btnNextPage");

  const btnExportCsv = document.getElementById("btnExportCsv");
  const btnClearHistory = document.getElementById("btnClearHistory");

  // Modal Elements
  const histModal = document.getElementById("histModal");
  const btnCloseModal = document.getElementById("btnCloseModal");
  const modalPlateText = document.getElementById("modalPlateText");
  const modalCountryBadge = document.getElementById("modalCountryBadge");
  const modalProvinceText = document.getElementById("modalProvinceText");
  const modalValidBadge = document.getElementById("modalValidBadge");
  const modalImgBox = document.getElementById("modalImgBox");
  const modalPlateImg = document.getElementById("modalPlateImg");
  const tabCropPlate = document.getElementById("tabCropPlate");
  const tabCropScene = document.getElementById("tabCropScene");
  const modalPattern = document.getElementById("modalPattern");
  const modalProvProb = document.getElementById("modalProvProb");
  const modalBoxText = document.getElementById("modalBoxText");
  const modalCtcText = document.getElementById("modalCtcText");
  const modalTotalLatency = document.getElementById("modalTotalLatency");
  const modalLatencyBars = document.getElementById("modalLatencyBars");
  const modalLatencyChips = document.getElementById("modalLatencyChips");

  // Lightbox Zoom Modal Elements
  const lightboxModal = document.getElementById("lightboxModal");
  const lightboxTitle = document.getElementById("lightboxTitle");
  const lightboxSubtitle = document.getElementById("lightboxSubtitle");
  const lightboxBody = document.getElementById("lightboxBody");
  const lightboxImg = document.getElementById("lightboxImg");
  const btnZoomIn = document.getElementById("btnZoomIn");
  const btnZoomOut = document.getElementById("btnZoomOut");
  const btnZoomReset = document.getElementById("btnZoomReset");
  const btnZoomFit = document.getElementById("btnZoomFit");
  const btnCloseLightbox = document.getElementById("btnCloseLightbox");
  const lblZoomLevel = document.getElementById("lblZoomLevel");

  // Lightbox Pan & Zoom State
  let zoomLevel = 1.0;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;

  // Active record in inspection modal
  let activeModalRecord = null;
  let activeModalImageType = "plate"; // "plate" | "scene"

  // Cache for records currently loaded
  let currentRecordsMap = {};

  // Compute date range from quick filter
  function getDateRange() {
    const quick = filterDateQuick.value;
    const now = new Date();

    const formatDate = (d) => {
      const year = d.getFullYear();
      const month = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    if (quick === "today") {
      const s = formatDate(now);
      return { from: s, to: s };
    } else if (quick === "yesterday") {
      const y = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      const s = formatDate(y);
      return { from: s, to: s };
    } else if (quick === "7d") {
      const past = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      return { from: formatDate(past), to: formatDate(now) };
    } else if (quick === "30d") {
      const past = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      return { from: formatDate(past), to: formatDate(now) };
    } else if (quick === "custom") {
      return { from: filterDateFrom.value, to: filterDateTo.value };
    }
    return { from: "", to: "" };
  }

  // Load KPI Stats
  async function loadStats() {
    try {
      const days = filterDateQuick.value === "30d" ? 30 : 7;
      const res = await fetch(`/api/history/stats?days=${days}`);
      if (!res.ok) return;
      const data = await res.json();

      statTotal.textContent = Number(data.total_detections || 0).toLocaleString();
      statThai.textContent = Number(data.thai_count || 0).toLocaleString();
      statLao.textContent = Number(data.lao_count || 0).toLocaleString();
      statLatency.textContent = `${data.avg_latency_ms || 0} ms`;
      statValidPct.textContent = `${data.valid_pct || 100}%`;
    } catch (err) {
      console.warn("Failed to load history stats:", err);
    }
  }

  // Load Paginated History Table
  async function loadHistory() {
    try {
      const dateRange = getDateRange();
      const params = new URLSearchParams({
        page: currentPage,
        page_size: pageSize,
        date_from: dateRange.from,
        date_to: dateRange.to,
        country: filterCountry.value,
        status: filterStatus.value,
        search: filterSearch.value.trim(),
        sort_by: sortBy,
        sort_order: sortOrder,
      });

      const res = await fetch(`/api/history?${params.toString()}`);
      if (!res.ok) throw new Error("API returned " + res.status);
      const data = await res.json();

      renderTable(data.records || []);

      lblShowing.textContent = data.records?.length || 0;
      lblTotal.textContent = data.total || 0;
      lblCurrentPage.textContent = data.page || 1;
      lblTotalPages.textContent = data.total_pages || 1;

      btnPrevPage.disabled = (data.page || 1) <= 1;
      btnNextPage.disabled = (data.page || 1) >= (data.total_pages || 1);
    } catch (err) {
      historyTableBody.innerHTML = `
        <tr>
          <td colspan="9" style="text-align: center; padding: 40px; color: var(--accent-red);">
            Failed to load recognition history: ${err.message}
          </td>
        </tr>
      `;
    }
  }

  // Render Table Rows
  function renderTable(records) {
    currentRecordsMap = {};
    if (!records || records.length === 0) {
      historyTableBody.innerHTML = `
        <tr>
          <td colspan="9" style="text-align: center; padding: 48px; color: var(--text-muted);">
            No license plate recognition logs found matching current filters.
          </td>
        </tr>
      `;
      return;
    }

    const rowsHtml = records.map((r) => {
      currentRecordsMap[r.record_id] = r;
      const isThai = (r.country || "Thai") === "Thai";
      const countryBadge = isThai
        ? `<span class="badge-country thai">🇹🇭 Thai</span>`
        : `<span class="badge-country laos">🇱🇦 Laos</span>`;

      const validBadge = r.is_valid
        ? `<span class="badge-valid">VALID</span>`
        : `<span class="badge-invalid">INVALID</span>`;

      const thumbImg = r.thumbnail
        ? `<img src="${r.thumbnail}" alt="Crop" class="hist-thumb btn-thumb-zoom" data-record-id="${r.record_id}" title="🔍 Click to zoom / expand image">`
        : `<div class="hist-thumb" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:10px;">NO CROP</div>`;

      return `
        <tr data-record-id="${r.record_id}">
          <td>${thumbImg}</td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">${r.timestamp}</td>
          <td>${countryBadge}</td>
          <td><span class="hist-plate-badge">${escapeHtml(r.plate_text)}</span></td>
          <td style="font-weight: 600; color: #fff;">${escapeHtml(r.province || "-")}</td>
          <td style="font-size: 0.8rem; color: var(--text-secondary);">${escapeHtml(r.pattern || "-")}</td>
          <td>${validBadge}</td>
          <td style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-cyan);">${r.total_latency_ms || 0} ms</td>
          <td style="text-align: center;">
            <button type="button" class="btn-hist btn-inspect" data-record-id="${r.record_id}" style="padding: 4px 10px; font-size: 0.74rem; background: rgba(0, 240, 255, 0.12); color: var(--accent-cyan); border-color: rgba(0, 240, 255, 0.3);">
              Inspect
            </button>
          </td>
        </tr>
      `;
    }).join("");

    historyTableBody.innerHTML = rowsHtml;

    // Attach click listener for thumbnail zoom
    document.querySelectorAll(".btn-thumb-zoom").forEach((img) => {
      img.addEventListener("click", (e) => {
        e.stopPropagation();
        const rid = img.getAttribute("data-record-id");
        const rec = currentRecordsMap[rid];
        if (rec) {
          const title = `${rec.plate_text || "Plate"} - ${rec.province || ""}`;
          const subtitle = `${rec.timestamp} • ${rec.country || "Thai"} • ${rec.pattern || ""}`;
          // Prefer full raw image if available, else thumbnail
          const imgSrc = rec.raw_image || rec.thumbnail;
          openLightbox(imgSrc, title, subtitle);
        }
      });
    });

    // Attach click listener for Inspect buttons
    document.querySelectorAll(".btn-inspect").forEach((btn) => {
      btn.addEventListener("click", () => {
        const rid = btn.getAttribute("data-record-id");
        if (currentRecordsMap[rid]) {
          openInspectionModal(currentRecordsMap[rid]);
        }
      });
    });
  }

  // Open Detailed Modal
  function openInspectionModal(record) {
    activeModalRecord = record;
    activeModalImageType = "plate";

    modalPlateText.textContent = record.plate_text || "UNKNOWN";
    const isThai = (record.country || "Thai") === "Thai";

    modalCountryBadge.className = isThai ? "badge-country thai" : "badge-country laos";
    modalCountryBadge.textContent = isThai ? "🇹🇭 Thailand" : "🇱🇦 Laos";

    modalProvinceText.textContent = record.province || "-";
    modalValidBadge.className = record.is_valid ? "badge-valid" : "badge-invalid";
    modalValidBadge.textContent = record.is_valid ? "VALID FORMAT" : "INVALID FORMAT";

    modalPattern.textContent = record.pattern || "Custom / Unstandardized";
    modalProvProb.textContent = `${record.province_prob || 0}%`;
    modalBoxText.textContent = record.char_box_text || "(None)";
    modalCtcText.textContent = record.ctc_text || "(None)";
    modalTotalLatency.textContent = `${record.total_latency_ms || 0} ms`;

    // Handle Image Switcher Tabs
    tabCropPlate.classList.add("active");
    tabCropScene.classList.remove("active");

    if (record.raw_image) {
      tabCropScene.style.display = "inline-block";
    } else {
      tabCropScene.style.display = "none";
    }

    if (record.thumbnail) {
      modalPlateImg.src = record.thumbnail;
      modalPlateImg.style.display = "block";
    } else if (record.raw_image) {
      modalPlateImg.src = record.raw_image;
      modalPlateImg.style.display = "block";
    } else {
      modalPlateImg.style.display = "none";
    }

    // Render Latency Waterfall Blocks & Chips
    const lat = record.model_latencies || {};
    const total = Math.max(1, record.total_latency_ms || 1);
    const m1 = lat.m1_ms || 0;
    const countryMs = lat.country_ms || 0;
    const m2 = lat.m2_ms || 0;
    const m3 = lat.m3_ms || (lat.m3_ocr_ms || 0) + (lat.m3_prov_ms || 0);

    modalLatencyBars.innerHTML = `
      <div style="width: ${(m1 / total) * 100}%; background: #38bdf8;" title="Model 1 (Plate Det): ${m1}ms"></div>
      <div style="width: ${(countryMs / total) * 100}%; background: #06b6d4;" title="Country Cls: ${countryMs}ms"></div>
      <div style="width: ${(m2 / total) * 100}%; background: #f59e0b;" title="Model 2 (Warp/Layout): ${m2}ms"></div>
      <div style="width: ${(m3 / total) * 100}%; background: #a855f7;" title="Model 3 (OCR & Province): ${m3}ms"></div>
    `;

    if (modalLatencyChips) {
      modalLatencyChips.innerHTML = `
        <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 4px; padding: 2px 8px;">M1 Det: <b>${m1}ms</b></span>
        <span style="background: rgba(6, 182, 212, 0.15); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 4px; padding: 2px 8px;">Country: <b>${countryMs}ms</b></span>
        <span style="background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 4px; padding: 2px 8px;">M2 Warp: <b>${m2}ms</b></span>
        <span style="background: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 4px; padding: 2px 8px;">M3 Recog: <b>${m3}ms</b></span>
      `;
    }

    histModal.style.display = "flex";
  }

  // Modal Image Tabs click handlers
  tabCropPlate.addEventListener("click", () => {
    tabCropPlate.classList.add("active");
    tabCropScene.classList.remove("active");
    activeModalImageType = "plate";
    if (activeModalRecord && activeModalRecord.thumbnail) {
      modalPlateImg.src = activeModalRecord.thumbnail;
    }
  });

  tabCropScene.addEventListener("click", () => {
    tabCropScene.classList.add("active");
    tabCropPlate.classList.remove("active");
    activeModalImageType = "scene";
    if (activeModalRecord && activeModalRecord.raw_image) {
      modalPlateImg.src = activeModalRecord.raw_image;
    }
  });

  // Clicking image in inspection modal opens Lightbox Zoom!
  modalImgBox.addEventListener("click", () => {
    if (!activeModalRecord) return;
    const isPlate = activeModalImageType === "plate";
    const src = isPlate ? (activeModalRecord.thumbnail || activeModalRecord.raw_image) : (activeModalRecord.raw_image || activeModalRecord.thumbnail);
    const title = `${activeModalRecord.plate_text} (${isPlate ? "Plate Crop" : "Full Vehicle Scene"})`;
    const subtitle = `${activeModalRecord.timestamp} • ${activeModalRecord.country} • ${activeModalRecord.province || ""}`;
    openLightbox(src, title, subtitle);
  });

  function closeModal() {
    histModal.style.display = "none";
  }

  btnCloseModal.addEventListener("click", closeModal);
  histModal.addEventListener("click", (e) => {
    if (e.target === histModal) closeModal();
  });

  // --- Lightbox Pan & Zoom Controller ---
  function updateLightboxTransform() {
    lightboxImg.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
    lblZoomLevel.textContent = `${Math.round(zoomLevel * 100)}%`;
    if (zoomLevel > 1.05) {
      lightboxBody.classList.add("zoomed");
    } else {
      lightboxBody.classList.remove("zoomed");
    }
  }

  function openLightbox(src, title, subtitle) {
    if (!src) return;
    lightboxImg.src = src;
    lightboxTitle.textContent = title || "Image Inspection";
    lightboxSubtitle.textContent = subtitle || "";
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;
    updateLightboxTransform();
    lightboxModal.style.display = "flex";
  }

  function closeLightbox() {
    lightboxModal.style.display = "none";
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;
    updateLightboxTransform();
  }

  btnCloseLightbox.addEventListener("click", closeLightbox);
  lightboxModal.addEventListener("click", (e) => {
    if (e.target === lightboxModal) closeLightbox();
  });

  btnZoomIn.addEventListener("click", () => {
    zoomLevel = Math.min(4.0, zoomLevel + 0.3);
    updateLightboxTransform();
  });

  btnZoomOut.addEventListener("click", () => {
    zoomLevel = Math.max(0.4, zoomLevel - 0.3);
    if (zoomLevel <= 1.0) {
      panX = 0;
      panY = 0;
    }
    updateLightboxTransform();
  });

  btnZoomReset.addEventListener("click", () => {
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;
    updateLightboxTransform();
  });

  btnZoomFit.addEventListener("click", () => {
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;
    updateLightboxTransform();
  });

  // Mouse wheel zoom inside lightbox
  lightboxBody.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.2 : -0.2;
    zoomLevel = Math.max(0.4, Math.min(4.5, zoomLevel + delta));
    if (zoomLevel <= 1.0) {
      panX = 0;
      panY = 0;
    }
    updateLightboxTransform();
  }, { passive: false });

  // Click image to toggle zoom (1x <-> 2.2x)
  lightboxImg.addEventListener("click", (e) => {
    e.stopPropagation();
    if (zoomLevel > 1.05) {
      zoomLevel = 1.0;
      panX = 0;
      panY = 0;
    } else {
      zoomLevel = 2.2;
    }
    updateLightboxTransform();
  });

  // Pan dragging when zoomed
  lightboxBody.addEventListener("mousedown", (e) => {
    if (zoomLevel <= 1.05) return;
    isDragging = true;
    dragStartX = e.clientX - panX;
    dragStartY = e.clientY - panY;
    lightboxBody.classList.add("grabbing");
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    panX = e.clientX - dragStartX;
    panY = e.clientY - dragStartY;
    updateLightboxTransform();
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      lightboxBody.classList.remove("grabbing");
    }
  });

  // Global ESC key listener
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (lightboxModal.style.display === "flex") {
        closeLightbox();
      } else if (histModal.style.display === "flex") {
        closeModal();
      }
    }
  });

  // Filter Event Handlers
  filterDateQuick.addEventListener("change", () => {
    if (filterDateQuick.value === "custom") {
      customDateWrap.style.display = "inline-flex";
    } else {
      customDateWrap.style.display = "none";
      currentPage = 1;
      loadStats();
      loadHistory();
    }
  });

  btnApplyFilter.addEventListener("click", () => {
    currentPage = 1;
    loadStats();
    loadHistory();
  });

  filterSearch.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      currentPage = 1;
      loadHistory();
    }
  });

  // Sorting
  document.querySelectorAll(".hist-table th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const field = th.getAttribute("data-sort");
      if (sortBy === field) {
        sortOrder = sortOrder === "desc" ? "asc" : "desc";
      } else {
        sortBy = field;
        sortOrder = "desc";
      }
      updateSortIcons();
      loadHistory();
    });
  });

  function updateSortIcons() {
    ["Timestamp", "Plate", "Province", "Latency"].forEach((name) => {
      const el = document.getElementById(`sortIcon${name}`);
      if (el) el.textContent = "";
    });
    const activeMap = {
      timestamp: "sortIconTimestamp",
      plate_text: "sortIconPlate",
      province: "sortIconProvince",
      total_latency_ms: "sortIconLatency",
    };
    const activeIcon = document.getElementById(activeMap[sortBy]);
    if (activeIcon) {
      activeIcon.textContent = sortOrder === "desc" ? "▼" : "▲";
    }
  }

  // Pagination Controls
  btnPrevPage.addEventListener("click", () => {
    if (currentPage > 1) {
      currentPage -= 1;
      loadHistory();
    }
  });

  btnNextPage.addEventListener("click", () => {
    currentPage += 1;
    loadHistory();
  });

  // Live Auto-Refresh Toggle
  btnToggleRefresh.addEventListener("click", () => {
    isLive = !isLive;
    if (isLive) {
      btnToggleRefresh.className = "btn-hist btn-hist-live active";
      lblLive.textContent = "Live: ON (10s)";
      startAutoRefresh();
    } else {
      btnToggleRefresh.className = "btn-hist btn-hist-live";
      lblLive.textContent = "Live: OFF";
      stopAutoRefresh();
    }
  });

  function startAutoRefresh() {
    stopAutoRefresh();
    autoRefreshTimer = setInterval(() => {
      loadStats();
      loadHistory();
    }, 10000);
  }

  function stopAutoRefresh() {
    if (autoRefreshTimer) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
  }

  // Export CSV
  btnExportCsv.addEventListener("click", () => {
    const dateRange = getDateRange();
    const params = new URLSearchParams({
      date_from: dateRange.from,
      date_to: dateRange.to,
      country: filterCountry.value,
      search: filterSearch.value.trim(),
    });
    window.location.href = `/api/history/export?${params.toString()}`;
  });

  // Clear History
  btnClearHistory.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear historical recognition records? This cannot be undone.")) {
      return;
    }
    try {
      const res = await fetch("/api/history/clear", { method: "POST" });
      const data = await res.json();
      alert(`Cleared ${data.deleted_records || 0} historical records.`);
      currentPage = 1;
      loadStats();
      loadHistory();
    } catch (err) {
      alert("Failed to clear history: " + err.message);
    }
  });

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial Load
  loadStats();
  loadHistory();
  startAutoRefresh();
});
