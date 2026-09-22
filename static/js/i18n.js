/**
 * static/js/i18n.js
 * Bilingual UI layer — English (default) / Thai, switchable from the topbar.
 * Usage:
 *   Static markup:  <span data-i18n="key">fallback</span>
 *   Dynamic JS:     I18N.t('key', 'fallback text')
 *   Persisted in localStorage('lpr_lang'). Applied automatically on DOM ready.
 */
(function () {
  'use strict';

  // [th, en] — index resolved by current language
  const DICT = {
    // ---- App shell ----
    app_name: ['ระบบอ่านป้ายทะเบียน', 'License Plate Recognition'],
    app_sub: ['Thai & Lao LPR', 'Thai & Lao LPR'],
    sec_overview: ['ภาพรวม', 'Overview'],
    sec_system: ['ระบบ', 'System'],
    sec_user: ['ผู้ใช้งาน', 'User'],
    nav_live: ['จอตรวจจับสด', 'Live Monitor'],
    nav_history: ['ประวัติการอ่านป้าย', 'Recognition History'],
    nav_models: ['สถานะโมเดล AI', 'AI Models'],
    nav_debug: ['โหมดตรวจสอบละเอียด', 'Debug Inspection'],
    nav_debug_title: ['เปิดเพื่อดูภาพขั้นตอนการประมวลผลของ AI (ใช้ทรัพยากรมากขึ้น)', 'Show intermediate AI pipeline images (uses more resources)'],
    nav_settings: ['ตั้งค่าของฉัน', 'My Settings'],
    nav_logout: ['ออกจากระบบ', 'Sign Out'],
    nav_login: ['เข้าสู่ระบบ', 'Sign In'],
    models_ready: ['โมเดลพร้อมใช้งาน', 'Models ready'],
    nav_collapse: ['ย่อเมนู', 'Collapse'],
    nav_expand: ['ขยายเมนู', 'Expand'],

    // ---- Live page ----
    live_title: ['จอตรวจจับสด', 'Live Monitor'],
    live_sub: ['อัปโหลดภาพ/วิดีโอ หรือเชื่อมต่อกล้อง RTSP เพื่ออ่านป้ายทะเบียนแบบเรียลไทม์', 'Upload images/videos or connect an RTSP camera for real-time plate recognition'],
    tab_upload: ['ทดสอบภาพ / วิดีโอ', 'Image / Video'],
    tab_live: ['กล้องสด (RTSP)', 'Live Camera (RTSP)'],
    upload_title: ['อัปโหลดไฟล์ทดสอบ', 'Upload Files'],
    files_unit: ['ไฟล์', 'files'],
    drop_title: ['ลากไฟล์มาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์', 'Drag & drop files here, or click to browse'],
    drop_sub: ['รองรับภาพ .jpg .png .webp และวิดีโอ .mp4 (หลายไฟล์พร้อมกัน)', 'Supports .jpg .png .webp images and .mp4 videos (multiple files)'],
    one_video: ['1 วิดีโอ', '1 video'],

    // Upload video player
    vstream_title: ['กำลังประมวลผลวิดีโอแบบเรียลไทม์', 'Real-time video processing'],
    vstream_mode: ['ทดสอบจากไฟล์', 'File Test'],
    vstream_replay: ['เล่นซ้ำ', 'Replay'],
    vstream_scan: ['สแกนทุกเฟรม', 'Scan All Frames'],
    vstream_close: ['ปิดวิดีโอ', 'Close'],
    vstream_idle: ['สถานะ: รอการเคลื่อนไหว', 'Gate idle'],
    vstream_detecting: ['กำลังอ่านป้าย', 'Reading plate'],
    conf_min: ['ความมั่นใจขั้นต่ำ:', 'Min. confidence:'],
    conf_hint: ['สูงขึ้น = เข้มงวดขึ้น ลดการอ่านเจอแต่สิ่งไม่ใช่ป้าย', 'Higher = stricter, fewer false reads'],
    zoom_hint: ['ซูม/เลื่อนได้', 'Zoom & pan'],

    // RTSP panel
    rtsp_title: ['กล้องสด (RTSP)', 'Live Camera (RTSP)'],
    rtsp_standby: ['รอเชื่อมต่อ', 'Standby'],
    rtsp_connect: ['เชื่อมต่อ', 'Connect'],
    rtsp_disconnect: ['ตัดการเชื่อมต่อ', 'Disconnect'],
    rtsp_placeholder: ['ใส่ที่อยู่กล้อง RTSP หรือหมายเลขกล้อง แล้วกดเชื่อมต่อ', 'Enter an RTSP address or camera index, then connect'],
    rtsp_streaming: ['กำลังถ่ายทอดสด', 'Live streaming'],

    // Pipeline
    pipe_title: ['ผลการประมวลผลรายสเต็ป', 'Pipeline Breakdown'],
    pipe_latency: ['เวลาประมวลผล:', 'Latency:'],
    th_raw: ['ภาพต้นฉบับ', 'Raw Input'],
    th_m1: ['ค้นหาป้าย', 'Plate Detection'],
    th_m2: ['แยกโซนข้อความ', 'Text Zone Split'],
    th_m3: ['อ่านตัวหนังสือ', 'Recognition'],
    waiting_file: ['รอไฟล์...', 'Waiting for file...'],
    no_plate_yet: ['ยังไม่พบป้าย', 'No plate yet'],
    no_plate: ['ไม่พบป้าย', 'NO PLATE'],
    plate_found: ['พบป้ายทะเบียน', 'Plate detected'],
    not_found_status: ['ไม่พบป้าย', 'No plate'],
    ready_status: ['พร้อมทำงาน', 'Ready'],
    processing_status: ['กำลังประมวลผล...', 'Processing...'],
    err_status: ['เกิดข้อผิดพลาด', 'Error'],
    res_wait: ['รอผลลัพธ์', 'Awaiting result'],
    res_valid: ['รูปแบบถูกต้อง', 'VALID FORMAT'],
    res_invalid: ['รูปแบบไม่ตรงมาตรฐาน', 'NON-STANDARD'],
    res_lowconf: ['ความมั่นใจต่ำ — เก็บภาพไว้ตรวจสอบ', 'LOW CONFIDENCE — SAVED FOR REVIEW'],
    meta_resolution: ['ความละเอียด', 'Resolution'],
    meta_status: ['สถานะ', 'Status'],
    meta_country: ['ประเทศ', 'Country'],
    meta_confidence: ['ความมั่นใจ', 'Confidence'],
    meta_layout: ['เลย์เอาต์', 'Layout'],
    meta_pattern: ['รูปแบบป้าย', 'Plate pattern'],
    meta_provconf: ['ความมั่นใจจังหวัด', 'Province confidence'],
    lbl_char: ['ตัวอักษร', 'Characters'],
    lbl_prov: ['จังหวัด', 'Province'],
    lbl_char_top: ['ตัวอักษร (บน)', 'Chars (Top)'],
    lbl_char_bot: ['ตัวอักษร (ล่าง)', 'Chars (Bottom)'],
    lbl_prov_top: ['จังหวัด (บน)', 'Province (Top)'],
    lbl_prov_bot: ['จังหวัด (ล่าง)', 'Province (Bottom)'],
    layout_std: ['มาตรฐาน', 'Standard'],
    layout_inv: ['กลับด้าน', 'Inverted'],
    country_th: ['ไทย', 'Thai'],
    country_lao: ['ลาว', 'Lao'],
    dlt_label: ['รหัสรถบรรทุก ขส.10ป:', 'DLT truck code:'],
    alt_badge: ['ตัวเลือกที่เป็นไปได้', 'ALTERNATIVE CANDIDATE'],

    // Debug drawer
    dbg_title: ['ภาพขั้นตอนการประมวลผล (โหมดตรวจสอบ)', 'Pipeline Inspection (Debug Mode)'],
    dbg_fordev: ['สำหรับนักพัฒนา', 'Developer tool'],
    dbg_1: ['ขั้นที่ 1: กรอบป้าย & มุม 4 จุด', 'Stage 1: Plate box & 4 corners'],
    dbg_1b: ['ขั้นที่ 1: ป้ายหลังปรับมุม', 'Stage 1: Deskewed plate'],
    dbg_2: ['ขั้นที่ 2: โซนตัวอักษร/จังหวัด', 'Stage 2: Char/province zones'],
    dbg_3: ['ขั้นที่ 3: ภาพก่อนอ่านตัวหนังสือ', 'Stage 3: Pre-OCR image'],
    dbg_3b: ['ขั้นที่ 3: จังหวัดที่น่าจะเป็นอันดับต้น', 'Stage 3: Top province candidates'],
    dbg_3c: ['ขั้นที่ 3: กรอบตัวอักษรแต่ละตัว', 'Stage 3: Per-character boxes'],

    // Models modal
    mm_title: ['โมเดล AI ที่กำลังทำงาน', 'Active AI Models'],
    mm_sub: ['ระบบอ่านป้ายทะเบียนไทยและลาว แบ่งการทำงานเป็น 4 ขั้นตอน', 'Thai & Lao plate recognition in 4 pipeline stages'],
    mm_active: ['ทำงานอยู่', 'ACTIVE'],
    mm_close: ['ปิด', 'Close'],
    mm_device: ['อุปกรณ์ประมวลผล:', 'Device:'],

    // Models modal extra
    mm_model: ['โมเดล:', 'Model:'],
    mm_s1_t: ['ค้นหาตำแหน่งป้ายทะเบียน', 'License Plate Detection & Rectification'],
    mm_s1_s: ['ระบุกรอบป้ายและปรับมุมภาพให้ตรง', 'Quad perspective warp & rotated box localization'],
    mm_s15_t: ['จำแนกประเทศ', 'Country Classifier'],
    mm_s15_s: ['แยกป้ายไทยและลาว เพื่อเลือกวิธีอ่านที่เหมาะสม', 'Routes Thai vs Lao plates to the right pipeline'],
    mm_s2_t: ['แยกโซนตัวอักษรและจังหวัด', 'Layout Component Localization'],
    mm_s2_s: ['ระบุตำแหน่งแถวตัวหนังสือและแถวจังหวัด', 'Splits character row and province banner'],
    mm_s3_t: ['อ่านตัวหนังสือบนป้าย', 'Character Recognition (Dual-Engine)'],
    mm_s3_s: ['อ่านทีละตัวอักษรพร้อมตรวจซ้ำด้วยระบบสำรอง', 'Per-character classification + CTC fallback'],
    mm_s4_t: ['ระบุจังหวัด', 'Province Classification'],
    mm_s4_s: ['ครอบคลุม 77 จังหวัดไทย และ 18 จังหวัดลาว', '77 Thai + 18 Lao provinces'],
    mm_charbox: ['กรอบตัวอักษร:', 'Char BBox:'],
    mm_thai_cls: ['ตัวอักษรไทย:', 'Thai Cls:'],
    mm_lao_cls: ['ตัวอักษรลาว:', 'Lao Cls:'],
    mm_ctc: ['ระบบสำรอง:', 'CTC Engine:'],
    mm_thai_prov: ['จังหวัดไทย:', 'Thai:'],
    mm_lao_prov: ['จังหวัดลาว:', 'Lao:'],
    batch_title: ['ไฟล์ที่อัปโหลด (คลิกเพื่อดูผลลัพธ์)', 'Uploaded files (click to view result)'],

    // History extras
    nav_help: ['วิธีใช้งาน', 'How to use'],
    tab_plate_img: ['ภาพป้าย', 'Plate image'],
    tab_scene_img: ['ภาพรถเต็มคัน', 'Full scene'],
    click_expand: ['คลิกเพื่อขยาย', 'Click to expand'],
    ctc_output: ['ผลจากระบบอ่านสำรอง', 'CTC OCR output'],
    latency_breakdown: ['เวลาประมวลผลแต่ละขั้น', 'Per-stage latency'],
    lat_m1: ['ค้นหาป้าย', 'Plate det'],
    lat_country: ['จำแนกประเทศ', 'Country'],
    lat_m2: ['แยกโซน', 'Zone split'],
    lat_m3: ['อ่านตัวหนังสือ', 'Recognition'],
    no_image: ['ไม่มีภาพ', 'No image'],
    no_records: ['ไม่พบประวัติการอ่านป้ายตามเงื่อนไขที่เลือก', 'No records match the current filters'],
    load_fail: ['โหลดประวัติไม่สำเร็จ', 'Failed to load history'],
    cloud_loading: ['กำลังโหลดข้อมูลจากคลาวด์...', 'Loading cloud data...'],
    cloud_empty: ['ไม่พบข้อมูลในคลาวด์', 'No records in cloud'],
    no_unknown: ['ไม่ทราบเลขป้าย', 'UNKNOWN'],
    none: ['ไม่มี', 'None'],
    pattern_nonstd: ['ไม่ตรงรูปแบบมาตรฐาน', 'Non-standard pattern'],
    exporting: ['กำลังเตรียมไฟล์ส่งออก...', 'Preparing export...'],
    cleared: ['ลบ {n} รายการเรียบร้อยแล้ว', 'Deleted {n} records'],
    clear_fail: ['ลบประวัติไม่สำเร็จ', 'Failed to clear history'],
    confirm_clear: ['ต้องการลบประวัติการอ่านป้ายทั้งหมดใช่หรือไม่? การกระทำนี้ไม่สามารถย้อนกลับได้', 'Delete all recognition history? This cannot be undone.'],
    help_text: ['วิธีใช้งานอย่างง่าย:\n\n1. เลือกช่วงเวลา ประเทศ และรูปแบบที่ต้องการดู\n2. พิมพ์เลขทะเบียนในช่องค้นหา แล้วกดค้นหา\n3. คลิกที่ภาพป้ายเพื่อขยายและซูม\n4. กดปุ่มรายละเอียดเพื่อดูข้อมูลครบทุกขั้นตอน\n5. กดส่งออกไฟล์เพื่อบันทึกเป็น Excel (CSV)', 'Quick guide:\n\n1. Pick a time range, country and format\n2. Type a plate number and press Search\n3. Click a crop image to zoom & pan\n4. Press Inspect for the full pipeline breakdown\n5. Press Export CSV to download an Excel file'],

    // Toasts
    toast_proc_fail: ['ประมวลผลไม่สำเร็จ', 'Processing failed'],
    toast_sample_fail: ['โหลดภาพตัวอย่างไม่สำเร็จ', 'Failed to load sample'],
    toast_video_fail: ['เริ่มประมวลผลวิดีโอไม่สำเร็จ', 'Failed to start video'],
    toast_noplate_video: ['ไม่พบป้ายทะเบียนในวิดีโอนี้', 'No plates found in this video'],
    toast_reproc: ['กำลังประมวลผลซ้ำ (โหมดตรวจสอบ)...', 'Re-processing (debug mode)...'],
    toast_video_start: ['กำลังเริ่มประมวลผลวิดีโอ...', 'Starting video processing...'],
    toast_scanning: ['กำลังสแกนทุกเฟรม...', 'Scanning all frames...'],
    toast_scan_fail: ['สแกนวิดีโอไม่สำเร็จ', 'Video scan failed'],

    // ---- History page ----
    hist_title: ['ประวัติการอ่านป้ายทะเบียน', 'Recognition History'],
    hist_sub: ['ค้นหา ตรวจสอบ และส่งออกข้อมูลการอ่านป้ายย้อนหลังทั้งหมด', 'Search, inspect and export all past plate reads'],
    kpi_total: ['อ่านป้ายทั้งหมด', 'Total Detections'],
    kpi_thai: ['ป้ายไทย', 'Thai Plates'],
    kpi_lao: ['ป้ายลาว', 'Lao Plates'],
    kpi_latency: ['เวลาประมวลผลเฉลี่ย', 'Avg. Latency'],
    kpi_valid: ['รูปแบบถูกต้อง', 'Valid Format'],
    f_7d: ['7 วันล่าสุด', 'Last 7 days'],
    f_today: ['วันนี้', 'Today'],
    f_yesterday: ['เมื่อวาน', 'Yesterday'],
    f_30d: ['30 วันล่าสุด', 'Last 30 days'],
    f_all: ['ทั้งหมด', 'All time'],
    f_custom: ['เลือกช่วงเอง...', 'Custom range...'],
    f_to: ['ถึง', 'to'],
    f_allcountry: ['ทุกประเทศ', 'All countries'],
    thailand: ['ไทย', 'Thailand'],
    laos: ['ลาว', 'Laos'],
    f_allformat: ['ทุกรูปแบบ', 'All formats'],
    f_valid: ['รูปแบบถูกต้อง', 'Valid only'],
    f_invalid: ['รูปแบบไม่ตรงมาตรฐาน', 'Non-standard only'],
    search_ph: ['ค้นหาเลขป้าย, จังหวัด, รูปแบบ...', 'Search plate, province, pattern...'],
    btn_search: ['ค้นหา', 'Search'],
    src_local: ['ฐานข้อมูลเครื่องนี้', 'Local DB'],
    src_cloud: ['ข้อมูลคลาวด์', 'Cloud Data'],
    autorefresh: ['อัปเดตอัตโนมัติ:', 'Auto-refresh:'],
    on: ['เปิด', 'ON'],
    off: ['ปิด', 'OFF'],
    btn_export: ['ส่งออกไฟล์', 'Export CSV'],
    btn_clear: ['ลบประวัติ', 'Clear History'],
    col_crop: ['ภาพป้าย', 'Crop'],
    col_time: ['เวลา', 'Time'],
    col_time_utc7: ['เวลา (UTC+7)', 'Time (UTC+7)'],
    col_country: ['ประเทศ', 'Country'],
    col_plate: ['เลขทะเบียน', 'Plate No.'],
    col_province: ['จังหวัด', 'Province'],
    col_pattern: ['รูปแบบ', 'Pattern'],
    col_validity: ['ความถูกต้อง', 'Validity'],
    col_by: ['อ่านโดย', 'Read by'],
    col_latency: ['เวลา', 'Latency'],
    col_detail: ['รายละเอียด', 'Details'],
    valid_short: ['ถูกต้อง', 'VALID'],
    invalid_short: ['ไม่ตรงมาตรฐาน', 'INVALID'],
    loading_hist: ['กำลังโหลดประวัติการอ่านป้าย...', 'Loading history...'],
    showing: ['แสดง', 'Showing'],
    of_records: ['จาก', 'of'],
    records_unit: ['รายการ', 'records'],
    page_word: ['หน้า', 'Page'],
    prev: ['ก่อนหน้า', 'Prev'],
    next: ['ถัดไป', 'Next'],
    btn_inspect: ['รายละเอียด', 'Inspect'],
    guest: ['ผู้เยี่ยมชม', 'Guest'],

    // Auth
    signin_tab: ['เข้าสู่ระบบ', 'Sign In'],
    register_tab: ['สมัครบัญชีใหม่', 'Register'],
    auth_title_login: ['เข้าสู่ระบบแดชบอร์ดอ่านป้ายทะเบียน', 'Sign in to the LPR dashboard'],
    auth_title_register: ['สมัครบัญชีผู้ใช้งานระบบอ่านป้ายทะเบียน', 'Create an LPR operator account'],
    lbl_name: ['ชื่อ-นามสกุล', 'Full name'],
    lbl_role: ['สิทธิ์การใช้งาน', 'User role'],
    role_admin: ['ผู้ดูแลระบบ (จัดการได้ทุกอย่าง)', 'Administrator (full access)'],
    role_operator: ['ผู้ปฏิบัติงาน (ตรวจสอบและบันทึก)', 'Operator (monitor & review)'],
    role_viewer: ['ผู้ชม (ดูข้อมูลเท่านั้น)', 'Viewer (read only)'],
    lbl_email: ['อีเมล หรือ ชื่อผู้ใช้', 'Email or username'],
    lbl_password: ['รหัสผ่าน', 'Password'],
    btn_signin: ['เข้าสู่ระบบ', 'Sign In'],
    btn_register: ['สมัครบัญชีใหม่', 'Create Account'],
    role_admin_short: ['ผู้ดูแล', 'Admin'],
    role_operator_short: ['ผู้ปฏิบัติงาน', 'Operator'],
    role_viewer_short: ['ผู้ชม', 'Viewer'],
    settings_title_1: ['ตั้งค่า', 'User'],
    settings_title_2: ['ผู้ใช้', 'Settings'],
    set_saved_for: ['บันทึกไว้ในบัญชี:', 'Saved to account:'],
    set_conf: ['ความมั่นใจขั้นต่ำเริ่มต้น (0.1 - 0.99)', 'Default confidence threshold (0.1 - 0.99)'],
    set_conf_hint: ['ค่าสูง = ระบบเข้มงวดขึ้น มั่นใจก่อนแจ้งผล', 'Higher = stricter before reporting'],
    set_refresh: ['โหลดประวัติใหม่อัตโนมัติ', 'Auto-refresh history'],
    set_refresh_on: ['เปิด (อัปเดตแบบเรียลไทม์)', 'Enabled (real-time)'],
    set_refresh_off: ['ปิด (กดโหลดเอง)', 'Disabled (manual)'],
    set_debug: ['โหมดตรวจสอบละเอียดเริ่มต้น', 'Default debug inspection'],
    set_debug_off: ['ปิด (ประหยัดทรัพยากร — แนะนำ)', 'Off (saves resources — recommended)'],
    set_debug_on: ['เปิด (ดูภาพทุกขั้นตอนของ AI)', 'On (inspect all AI steps)'],
    set_save: ['บันทึกการตั้งค่า', 'Save Settings'],
    toast_set_saved: ['บันทึกการตั้งค่าเรียบร้อยแล้ว', 'Settings saved'],
    toast_set_fail: ['บันทึกการตั้งค่าไม่สำเร็จ', 'Failed to save settings'],
    auth_bad_creds: ['อีเมลหรือรหัสผ่านไม่ถูกต้อง', 'Invalid email/username or password'],
    auth_net_err: ['เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ', 'Network error'],
    auth_reg_fail: ['สมัครบัญชีไม่สำเร็จ', 'Registration failed'],
  };

  const DIRECTION_FALLBACK = 'en';
  let current = (localStorage.getItem('lpr_lang') || 'en').toLowerCase() === 'th' ? 'th' : 'en';

  function t(key, fallback) {
    const entry = DICT[key];
    if (!entry) return fallback || key;
    return entry[current === 'th' ? 0 : 1] || entry[1] || fallback || key;
  }

  function apply(root) {
    const scope = root || document;
    scope.querySelectorAll('[data-i18n]').forEach((el) => {
      const key = el.getAttribute('data-i18n');
      const val = t(key, null);
      if (val) el.textContent = val;
    });
    scope.querySelectorAll('[data-i18n-ph]').forEach((el) => {
      const key = el.getAttribute('data-i18n-ph');
      const val = t(key, null);
      if (val) el.placeholder = val;
    });
    scope.querySelectorAll('[data-i18n-title]').forEach((el) => {
      const key = el.getAttribute('data-i18n-title');
      const val = t(key, null);
      if (val) el.title = val;
    });
    document.documentElement.lang = current === 'th' ? 'th' : 'en';
    window.dispatchEvent(new CustomEvent('lpr:lang-changed', { detail: { lang: current } }));
  }

  function setLang(lang) {
    current = lang === 'th' ? 'th' : 'en';
    localStorage.setItem('lpr_lang', current);
    apply();
  }

  function getLang() { return current; }

  // Language toggle button injected into any [data-lpr-lang] placeholder
  function renderToggle() {
    document.querySelectorAll('[data-lpr-lang]').forEach((host) => {
      host.innerHTML = `
        <button type="button" id="lpr-lang-btn" class="lang-toggle" title="Switch language / สลับภาษา">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7m-3.5-3.5L18 21m-3-6l-3 6m0-11l-1.5 3M14 3l4 6h-8l4-6z"/></svg>
          <span>${current === 'th' ? 'EN' : 'ไทย'}</span>
        </button>`;
      host.querySelector('#lpr-lang-btn').addEventListener('click', () => {
        setLang(current === 'th' ? 'en' : 'th');
        renderToggle();
      });
    });
  }

  window.I18N = { t, apply, setLang, getLang, renderToggle };
  document.addEventListener('DOMContentLoaded', () => { apply(); renderToggle(); initSidebarCollapse(); });

  // Sidebar collapse/expand — persisted per browser, shared by both pages
  function initSidebarCollapse() {
    const btn = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.sidebar');
    const shell = document.querySelector('.app-shell');
    if (!btn || !sidebar || !shell) return;

    const applyState = (collapsed) => {
      sidebar.classList.toggle('sidebar-collapsed', collapsed);
      shell.classList.toggle('sidebar-is-collapsed', collapsed);
      const label = btn.querySelector('.toggle-label');
      if (label) label.setAttribute('data-i18n', collapsed ? 'nav_expand' : 'nav_collapse');
    };

    let collapsed = localStorage.getItem('lpr_sidebar_collapsed') === '1';
    applyState(collapsed);
    apply();

    btn.addEventListener('click', () => {
      collapsed = !collapsed;
      localStorage.setItem('lpr_sidebar_collapsed', collapsed ? '1' : '0');
      applyState(collapsed);
      apply();
    });
  }
})();
