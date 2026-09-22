/**
 * static/js/history.js
 * Frontend controller for ประวัติการอ่านป้ายทะเบียน (Recognition History).
 * Uses shared LPRViewer (viewer.js) for full-screen zoom/pan inspection.
 */
document.addEventListener("DOMContentLoaded", () => {
  const t = (key, fallback) => (window.I18N ? I18N.t(key, fallback) : fallback || key);

  // State
  let currentPage = 1;
  const pageSize = 25;
  let sortBy = "timestamp";
  let sortOrder = "desc";
  let autoRefreshTimer = null;
  let isLive = true;

  // DOM
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
  const historyTableHead = document.getElementById("historyTableHead");
  const lblShowing = document.getElementById("lblShowing");
  const lblTotal = document.getElementById("lblTotal");
  const lblCurrentPage = document.getElementById("lblCurrentPage");
  const lblTotalPages = document.getElementById("lblTotalPages");
  const btnPrevPage = document.getElementById("btnPrevPage");
  const btnNextPage = document.getElementById("btnNextPage");

  const btnExportCsv = document.getElementById("btnExportCsv");
  const btnClearHistory = document.getElementById("btnClearHistory");

  const histSourceSwitcher = document.getElementById("histSourceSwitcher");
  const btnSourceLocal = document.getElementById("btnSourceLocal");
  const btnSourceBigQuery = document.getElementById("btnSourceBigQuery");
  let currentSource = "local";

  // Modal
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

  let activeModalRecord = null;
  let activeModalImageType = "plate";
  let currentRecordsMap = {};

  // --- Helpers ---
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function toast(message, type = "info", duration = 3500) {
    let el = document.getElementById("lpr-toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "lpr-toast";
      el.className = "lpr-toast";
      document.body.appendChild(el);
    }
    el.textContent = message;
    el.className = `lpr-toast show ${type === "error" ? "error" : type === "success" ? "success" : ""}`;
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.className = "lpr-toast"; }, duration);
  }

  // Thai date quick filter
  function getDateRange() {
    const quick = filterDateQuick.value;
    const now = new Date();
    const formatDate = (d) => {
      const year = d.getFullYear();
      const month = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    if (quick === "today") return { from: formatDate(now), to: formatDate(now) };
    if (quick === "yesterday") {
      const y = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      const s = formatDate(y);
      return { from: s, to: s };
    }
    if (quick === "7d") {
      const past = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      return { from: formatDate(past), to: formatDate(now) };
    }
    if (quick === "30d") {
      const past = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      return { from: formatDate(past), to: formatDate(now) };
    }
    if (quick === "custom") return { from: filterDateFrom.value, to: filterDateTo.value };
    return { from: "", to: "" };
  }

  // --- Data loading ---
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

  async function loadHistory() {
    try {
      if (currentSource === "cloud") {
        historyTableBody.innerHTML = `
          <tr><td colspan="10" style="text-align: center; padding: 40px; color: var(--accent);">
            ${t("cloud_loading", "Loading cloud data...")}
          </td></tr>`;
        const res = await fetch(`/api/cloud/firestore/records?limit=50`);
        if (!res.ok) throw new Error("API returned " + res.status);
        const data = await res.json();
        renderCloudTable(data.records || []);
        lblShowing.textContent = data.records?.length || 0;
        lblTotal.textContent = data.count || 0;
        lblCurrentPage.textContent = 1;
        lblTotalPages.textContent = 1;
        btnPrevPage.disabled = true;
        btnNextPage.disabled = true;
        return;
      }

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
        <tr><td colspan="10" style="text-align: center; padding: 40px; color: var(--accent-red);">
          ${t("load_fail", "Failed to load history")}: ${escapeHtml(err.message)}
        </td></tr>`;
    }
  }

  // --- Cloud table ---
  function renderCloudTable(records) {
    if (historyTableHead) {
      historyTableHead.innerHTML = `
        <tr>
          <th style="width: 110px;">${t("col_crop", "Crop")}</th>
          <th style="width: 170px;">${t("col_time_utc7", "Time (UTC+7)")}</th>
          <th style="width: 100px;">${t("col_country", "Country")}</th>
          <th style="width: 170px;">${t("col_plate", "Plate No.")}</th>
          <th style="width: 160px;">${t("col_province", "Province")}</th>
          <th style="width: 160px;">${t("col_pattern", "Pattern")}</th>
          <th style="width: 90px;">${t("col_validity", "Validity")}</th>
          <th style="width: 150px;">${t("col_by", "Read by")}</th>
          <th style="width: 90px;">${t("col_latency", "Latency")}</th>
        </tr>`;
    }

    if (!records || records.length === 0) {
      historyTableBody.innerHTML = `
        <tr><td colspan="10" style="text-align: center; padding: 48px; color: var(--text-muted);">
          ${t("cloud_empty", "No records in cloud")}
        </td></tr>`;
      return;
    }

    historyTableBody.innerHTML = records.map((r) => {
      const isThai = (r.country || "Thai") === "Thai";
      const countryBadge = isThai
        ? `<span class="badge-country thai">🇹🇭 ${t("country_th", "Thai")}</span>`
        : `<span class="badge-country laos">🇱🇦 ${t("country_lao", "Lao")}</span>`;
      const validBadge = r.is_valid
        ? `<span class="badge-valid">${t("valid_short", "VALID")}</span>`
        : `<span class="badge-invalid">${t("invalid_short", "INVALID")}</span>`;
      const userTag = r.user_name || r.user_email || t("guest", "Guest");
      const latency = r.total_latency_ms || r.latency_ms || 0;

      const cropRaw = r.thumbnail || r.plate_crop_base64 || "";
      const cropSrc = cropRaw
        ? cropRaw.startsWith("data:") ? cropRaw : `data:image/jpeg;base64,${cropRaw}`
        : null;
      const cropHtml = cropSrc
        ? `<img src="${cropSrc}" style="width:90px; height:38px; object-fit:cover; border-radius:5px; border:1px solid rgba(56,189,248,0.25);" data-lpr-viewer data-title="${escapeHtml(r.plate_text || "-")}" data-subtitle="${escapeHtml(r.timestamp || "")}">`
        : `<span style="font-size:0.7rem; color:var(--text-muted);">${t("no_image", "No image")}</span>`;

      return `
        <tr>
          <td>${cropHtml}</td>
          <td class="hist-mono" style="font-size: 0.8rem; color: var(--accent);">${escapeHtml(r.timestamp || "-")}</td>
          <td>${countryBadge}</td>
          <td><span class="hist-plate-badge" style="font-size: 0.92rem;">${escapeHtml(r.plate_text || r.plate_number || "-")}</span></td>
          <td style="font-weight: 600; color: var(--text-primary);">${escapeHtml(r.province || "-")}</td>
          <td style="font-size: 0.8rem; color: var(--text-secondary);">${escapeHtml(r.pattern || "-")}</td>
          <td>${validBadge}</td>
          <td><span class="hist-user-tag">${escapeHtml(userTag)}</span></td>
          <td class="hist-mono" style="font-weight: 600; color: var(--accent); font-size: 0.8rem;">${latency} ms</td>
        </tr>`;
    }).join("");
  }

  // --- Local table ---
  function renderTable(records) {
    if (historyTableHead) {
      historyTableHead.innerHTML = `
        <tr>
          <th style="width: 110px;">${t("col_crop", "Crop")}</th>
          <th class="sortable" data-sort="timestamp" style="width: 170px;">${t("col_time", "Time")} <span id="sortIconTimestamp">▼</span></th>
          <th style="width: 110px;">${t("col_country", "Country")}</th>
          <th class="sortable" data-sort="plate_text" style="width: 170px;">${t("col_plate", "Plate No.")} <span id="sortIconPlate"></span></th>
          <th class="sortable" data-sort="province" style="width: 150px;">${t("col_province", "Province")} <span id="sortIconProvince"></span></th>
          <th style="width: 150px;">${t("col_pattern", "Pattern")}</th>
          <th style="width: 100px;">${t("col_validity", "Validity")}</th>
          <th style="width: 130px;">${t("col_by", "Read by")}</th>
          <th class="sortable" data-sort="total_latency_ms" style="width: 100px;">${t("col_latency", "Latency")} <span id="sortIconLatency"></span></th>
          <th style="width: 90px; text-align: center;">${t("col_detail", "Details")}</th>
        </tr>`;
    }

    currentRecordsMap = {};
    if (!records || records.length === 0) {
      historyTableBody.innerHTML = `
        <tr><td colspan="10" style="text-align: center; padding: 48px; color: var(--text-muted);">
          ${t("no_records", "No records match the current filters")}
        </td></tr>`;
      return;
    }

    historyTableBody.innerHTML = records.map((r) => {
      currentRecordsMap[r.record_id] = r;
      const isThai = (r.country || "Thai") === "Thai";
      const countryBadge = isThai
        ? `<span class="badge-country thai">🇹🇭 ${t("country_th", "Thai")}</span>`
        : `<span class="badge-country laos">🇱🇦 ${t("country_lao", "Lao")}</span>`;
      const validBadge = r.is_valid
        ? `<span class="badge-valid">${t("valid_short", "VALID")}</span>`
        : `<span class="badge-invalid">${t("invalid_short", "INVALID")}</span>`;
      const thumbImg = r.thumbnail
        ? `<img src="${r.thumbnail}" alt="crop" class="hist-thumb btn-thumb-zoom" data-record-id="${r.record_id}" title="${t("click_expand", "Click to expand")}">`
        : `<div class="hist-thumb" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:10px;">${t("no_image", "No image")}</div>`;
      const userTag = r.user_email || r.user_id || t("guest", "Guest");

      return `
        <tr data-record-id="${r.record_id}">
          <td>${thumbImg}</td>
          <td class="hist-mono" style="font-size: 0.8rem; color: var(--text-secondary);">${escapeHtml(r.timestamp)}</td>
          <td>${countryBadge}</td>
          <td><span class="hist-plate-badge">${escapeHtml(r.plate_text)}</span></td>
          <td style="font-weight: 600; color: var(--text-primary);">${escapeHtml(r.province || "-")}</td>
          <td style="font-size: 0.8rem; color: var(--text-secondary);">${escapeHtml(r.pattern || "-")}</td>
          <td>${validBadge}</td>
          <td><span class="hist-user-tag" title="${escapeHtml(userTag)}">${escapeHtml(userTag)}</span></td>
          <td class="hist-mono" style="font-weight: 600; color: var(--accent);">${r.total_latency_ms || 0} ms</td>
          <td style="text-align: center;">
            <button type="button" class="btn-hist btn-inspect" data-record-id="${r.record_id}">${t("btn_inspect", "Inspect")}</button>
          </td>
        </tr>`;
    }).join("");

    bindRowActions();
  }

  function bindRowActions() {
    document.querySelectorAll(".btn-thumb-zoom").forEach((img) => {
      img.addEventListener("click", (e) => {
        e.stopPropagation();
        const rec = currentRecordsMap[img.getAttribute("data-record-id")];
        if (rec && window.LPRViewer) {
          LPRViewer.open({
            src: rec.raw_image || rec.thumbnail,
            title: `${rec.plate_text || "-"} • ${rec.province || ""}`,
            subtitle: `${rec.timestamp} • ${rec.country === "Laos" ? t("country_lao", "Lao") : t("country_th", "Thai")} • ${rec.pattern || ""}`,
          });
        }
      });
    });

    document.querySelectorAll(".btn-inspect").forEach((btn) => {
      btn.addEventListener("click", () => {
        const rec = currentRecordsMap[btn.getAttribute("data-record-id")];
        if (rec) openInspectionModal(rec);
      });
    });
  }

  // --- Inspection modal ---
  function openInspectionModal(record) {
    activeModalRecord = record;
    activeModalImageType = "plate";

    modalPlateText.textContent = record.plate_text || t("no_unknown", "UNKNOWN");
    const isThai = (record.country || "Thai") === "Thai";

    modalCountryBadge.className = isThai ? "badge-country thai" : "badge-country laos";
    modalCountryBadge.textContent = isThai ? `🇹🇭 ${t("country_th", "Thai")}` : `🇱🇦 ${t("country_lao", "Lao")}`;
    modalProvinceText.textContent = record.province || "-";
    modalValidBadge.className = record.is_valid ? "badge-valid" : "badge-invalid";
    modalValidBadge.textContent = record.is_valid ? t("res_valid", "VALID FORMAT") : t("res_invalid", "NON-STANDARD");

    modalPattern.textContent = record.pattern || t("pattern_nonstd", "Non-standard pattern");
    modalProvProb.textContent = `${record.province_prob || 0}%`;
    modalBoxText.textContent = record.char_box_text || `(${t("none", "None")})`;
    modalCtcText.textContent = record.ctc_text || `(${t("none", "None")})`;
    modalTotalLatency.textContent = `${record.total_latency_ms || 0} ms`;

    tabCropPlate.classList.add("active");
    tabCropScene.classList.remove("active");
    tabCropScene.style.display = record.raw_image ? "inline-block" : "none";

    if (record.thumbnail) {
      modalPlateImg.src = record.thumbnail;
      modalPlateImg.style.display = "block";
    } else if (record.raw_image) {
      modalPlateImg.src = record.raw_image;
      modalPlateImg.style.display = "block";
    } else {
      modalPlateImg.style.display = "none";
    }

    // Latency waterfall
    const lat = record.model_latencies || {};
    const total = Math.max(1, record.total_latency_ms || 1);
    const m1 = lat.m1_ms || 0;
    const countryMs = lat.country_ms || 0;
    const m2 = lat.m2_ms || 0;
    const m3 = lat.m3_ms || (lat.m3_ocr_ms || 0) + (lat.m3_prov_ms || 0);

    modalLatencyBars.innerHTML = `
      <div style="width: ${(m1 / total) * 100}%; background: #38bdf8;" title="${t("lat_m1", "Plate det")}: ${m1}ms"></div>
      <div style="width: ${(countryMs / total) * 100}%; background: #06b6d4;" title="${t("lat_country", "Country")}: ${countryMs}ms"></div>
      <div style="width: ${(m2 / total) * 100}%; background: #f59e0b;" title="${t("lat_m2", "Zone split")}: ${m2}ms"></div>
      <div style="width: ${(m3 / total) * 100}%; background: #a78bfa;" title="${t("lat_m3", "Recognition")}: ${m3}ms"></div>`;

    modalLatencyChips.innerHTML = `
      <span class="lat-chip" style="--c: 56, 189, 248;">${t("lat_m1", "Plate det")} <b>${m1}ms</b></span>
      <span class="lat-chip" style="--c: 6, 182, 212;">${t("lat_country", "Country")} <b>${countryMs}ms</b></span>
      <span class="lat-chip" style="--c: 245, 158, 11;">${t("lat_m2", "Zone split")} <b>${m2}ms</b></span>
      <span class="lat-chip" style="--c: 167, 139, 250;">${t("lat_m3", "Recognition")} <b>${m3}ms</b></span>`;

    histModal.style.display = "flex";
  }

  tabCropPlate.addEventListener("click", () => {
    tabCropPlate.classList.add("active");
    tabCropScene.classList.remove("active");
    activeModalImageType = "plate";
    if (activeModalRecord && activeModalRecord.thumbnail) modalPlateImg.src = activeModalRecord.thumbnail;
  });

  tabCropScene.addEventListener("click", () => {
    tabCropScene.classList.add("active");
    tabCropPlate.classList.remove("active");
    activeModalImageType = "scene";
    if (activeModalRecord && activeModalRecord.raw_image) modalPlateImg.src = activeModalRecord.raw_image;
  });

  modalImgBox.addEventListener("click", () => {
    if (!activeModalRecord || !window.LPRViewer) return;
    const isPlate = activeModalImageType === "plate";
    const src = isPlate
      ? (activeModalRecord.thumbnail || activeModalRecord.raw_image)
      : (activeModalRecord.raw_image || activeModalRecord.thumbnail);
    LPRViewer.open({
      src,
      title: `${activeModalRecord.plate_text} (${isPlate ? t("tab_plate_img", "Plate image") : t("tab_scene_img", "Full scene")})`,
      subtitle: `${activeModalRecord.timestamp} • ${activeModalRecord.country === "Laos" ? t("country_lao", "Lao") : t("country_th", "Thai")} • ${activeModalRecord.province || ""}`,
    });
  });

  function closeModal() { histModal.style.display = "none"; }
  btnCloseModal.addEventListener("click", closeModal);
  histModal.addEventListener("click", (e) => { if (e.target === histModal) closeModal(); });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && histModal.style.display === "flex") closeModal();
  });

  // --- Filters ---
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

  btnApplyFilter.addEventListener("click", () => { currentPage = 1; loadStats(); loadHistory(); });
  filterSearch.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { currentPage = 1; loadHistory(); }
  });

  // Sorting
  document.addEventListener("click", (e) => {
    const th = e.target.closest("th.sortable");
    if (!th) return;
    const field = th.getAttribute("data-sort");
    if (sortBy === field) sortOrder = sortOrder === "desc" ? "asc" : "desc";
    else { sortBy = field; sortOrder = "desc"; }
    updateSortIcons();
    loadHistory();
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
    if (activeIcon) activeIcon.textContent = sortOrder === "desc" ? "▼" : "▲";
  }

  // Pagination
  btnPrevPage.addEventListener("click", () => { if (currentPage > 1) { currentPage -= 1; loadHistory(); } });
  btnNextPage.addEventListener("click", () => { currentPage += 1; loadHistory(); });

  // Auto refresh
  btnToggleRefresh.addEventListener("click", () => {
    isLive = !isLive;
    if (isLive) {
      btnToggleRefresh.className = "btn-hist btn-hist-live active";
      lblLive.textContent = t("on", "ON");
      startAutoRefresh();
    } else {
      btnToggleRefresh.className = "btn-hist btn-hist-live";
      lblLive.textContent = t("off", "OFF");
      stopAutoRefresh();
    }
  });

  function startAutoRefresh() {
    stopAutoRefresh();
    autoRefreshTimer = setInterval(() => { loadStats(); loadHistory(); }, 10000);
  }
  function stopAutoRefresh() {
    if (autoRefreshTimer) { clearInterval(autoRefreshTimer); autoRefreshTimer = null; }
  }

  // Export
  btnExportCsv.addEventListener("click", () => {
    const dateRange = getDateRange();
    const params = new URLSearchParams({
      date_from: dateRange.from,
      date_to: dateRange.to,
      country: filterCountry.value,
      search: filterSearch.value.trim(),
    });
    window.location.href = `/api/history/export?${params.toString()}`;
    toast(t("exporting", "Preparing export..."), "success");
  });

  // Clear
  btnClearHistory.addEventListener("click", async () => {
    if (!confirm(t("confirm_clear", "Delete all history?"))) return;
    try {
      const res = await fetch("/api/history/clear", { method: "POST" });
      const data = await res.json();
      toast(t("cleared", "Deleted {n} records").replace("{n}", data.deleted_records || 0), "success");
      currentPage = 1;
      loadStats();
      loadHistory();
    } catch (err) {
      toast(t("clear_fail", "Failed to clear history") + ": " + err.message, "error");
    }
  });

  // Source switcher
  async function setupSourceSwitcher() {
    if (btnSourceLocal && btnSourceBigQuery) {
      btnSourceLocal.addEventListener("click", () => {
        if (currentSource === "local") return;
        currentSource = "local";
        btnSourceLocal.style.background = "var(--accent-strong)";
        btnSourceLocal.style.color = "#fff";
        btnSourceLocal.style.fontWeight = "600";
        btnSourceBigQuery.style.background = "transparent";
        btnSourceBigQuery.style.color = "var(--text-secondary)";
        btnSourceBigQuery.style.fontWeight = "500";
        loadHistory();
      });

      btnSourceBigQuery.addEventListener("click", () => {
        if (currentSource === "cloud") return;
        currentSource = "cloud";
        btnSourceBigQuery.style.background = "var(--accent-strong)";
        btnSourceBigQuery.style.color = "#fff";
        btnSourceBigQuery.style.fontWeight = "600";
        btnSourceLocal.style.background = "transparent";
        btnSourceLocal.style.color = "var(--text-secondary)";
        btnSourceLocal.style.fontWeight = "500";
        loadHistory();
      });
    }

    try {
      const resp = await fetch("/api/cloud/status");
      if (resp.ok) {
        const st = await resp.json();
        if (st.is_cloud_run) {
          if (histSourceSwitcher) histSourceSwitcher.style.display = "none";
          if (currentSource !== "cloud") { currentSource = "cloud"; loadHistory(); }
        }
      }
    } catch (e) { /* not on cloud */ }
  }

  // Help (sidebar) — simple onboarding dialog for first-time users
  const navHelpBtn = document.getElementById("navHelpBtn");
  if (navHelpBtn) {
    navHelpBtn.addEventListener("click", () => {
      alert(t("help_text", "Quick guide"));
    });
  }

  setupSourceSwitcher();
  loadStats();
  loadHistory();
  startAutoRefresh();

  // Re-render dynamic content when the user switches language
  window.addEventListener("lpr:lang-changed", () => {
    updateSortIcons();
    loadHistory();
  });
});
