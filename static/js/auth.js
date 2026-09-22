/**
 * Thai License Plate Recognition (Thai LPR) - Authentication & User Session Manager
 * Adapted and enhanced from StaffLenz AI pattern.
 * 
 * Capabilities:
 * - Email / User ID & Password Sign In
 * - User Registration with SQLite / Firestore Dual-Storage
 * - Session JWT Storage & Authorization Header Injection
 * - Fast Dev Admin Mode on Localhost
 * - User Settings & Preferences Sync (Confidence, Refresh Rate, Theme)
 * - Custom Lifecycle Events: thailpr:auth-success, thailpr:logout
 */

class ThaiLPRAuthManager {
    constructor() {
        this.t = (key, fallback) => (window.I18N ? I18N.t(key, fallback) : fallback || key);
        this.currentUser = null;
        this.token = null;
        this.authConfig = null;

        try {
            const savedToken = sessionStorage.getItem("thailpr_token");
            const savedUser = sessionStorage.getItem("thailpr_user");
            if (savedToken && savedUser) {
                this.token = savedToken;
                this.currentUser = JSON.parse(savedUser);
            }
        } catch (e) {
            console.warn("[AUTH] Could not restore session:", e);
        }

        this.patchFetch();
    }

    /**
     * Intercepts window.fetch to automatically include Authorization headers on /api/ calls.
     */
    patchFetch() {
        const originalFetch = window.fetch;
        const self = this;
        window.fetch = async function (resource, options = {}) {
            if (typeof resource === "string" && resource.startsWith("/api/") && !resource.startsWith("/api/auth/login") && !resource.startsWith("/api/auth/register")) {
                options = options || {};
                options.headers = options.headers || {};

                if (options.headers instanceof Headers) {
                    if (self.token && !options.headers.has("Authorization")) {
                        options.headers.set("Authorization", `Bearer ${self.token}`);
                    }
                } else if (Array.isArray(options.headers)) {
                    if (self.token) {
                        options.headers.push(["Authorization", `Bearer ${self.token}`]);
                    }
                } else {
                    if (self.token && !options.headers["Authorization"]) {
                        options.headers["Authorization"] = `Bearer ${self.token}`;
                    }
                }
            }
            return originalFetch.call(this, resource, options);
        };
    }

    async init() {
        try {
            const res = await fetch("/api/config/auth");
            if (res.ok) {
                this.authConfig = await res.json();
            }
        } catch (e) {
            console.warn("[AUTH] Could not fetch auth config:", e);
        }

        // Validate session if token is present
        if (this.token) {
            try {
                const meRes = await fetch("/api/auth/me", {
                    headers: this.getAuthHeaders(),
                });
                if (meRes.ok) {
                    const meData = await meRes.json();
                    this.currentUser = meData.user || this.currentUser;
                    sessionStorage.setItem("thailpr_user", JSON.stringify(this.currentUser));
                    this.hideLoginModal();
                    this.renderAuthUI();
                    window.dispatchEvent(new CustomEvent("thailpr:auth-success", { detail: { user: this.currentUser, token: this.token } }));
                    return;
                } else {
                    this.currentUser = null;
                    this.token = null;
                    try { sessionStorage.removeItem("thailpr_token"); sessionStorage.removeItem("thailpr_user"); } catch (e) { }
                }
            } catch (e) {
                // If offline, continue with cached user
                this.hideLoginModal();
                this.renderAuthUI();
                window.dispatchEvent(new CustomEvent("thailpr:auth-success", { detail: { user: this.currentUser, token: this.token } }));
                return;
            }
        }

        this.renderAuthUI();

        const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
        const isAuthRequired = this.authConfig?.auth_required !== false;
        const devAdminAllowed = this.authConfig?.allow_dev_admin === true;

        if (!this.currentUser && isLocalhost && devAdminAllowed) {
            await this.loginWithDevAdmin();
        } else if (!this.currentUser && isAuthRequired) {
            this.showLoginModal();
        }
    }

    getAuthHeaders() {
        const headers = { "Content-Type": "application/json" };
        if (this.token) {
            headers["Authorization"] = `Bearer ${this.token}`;
        }
        return headers;
    }

    async loginWithPassword(emailOrId, password) {
        const errorBox = document.getElementById("auth-modal-error");
        if (errorBox) errorBox.style.display = "none";

        try {
            const res = await fetch("/api/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email: emailOrId, password: password }),
            });
            const data = await res.json();
            if (res.ok && data.status === "success") {
                this.currentUser = data.user;
                this.token = data.token;
                try {
                    sessionStorage.setItem("thailpr_token", this.token);
                    sessionStorage.setItem("thailpr_user", JSON.stringify(this.currentUser));
                } catch (e) { }
                this.hideLoginModal();
                this.renderAuthUI();
                window.dispatchEvent(new CustomEvent("thailpr:auth-success", { detail: { user: this.currentUser, token: this.token } }));
                return { success: true };
            } else {
                const msg = data.detail || data.message || this.t("auth_bad_creds", "Invalid email/username or password");
                this.showAuthError(msg);
                return { success: false, message: msg };
            }
        } catch (err) {
            const msg = this.t("auth_net_err", "Network error") + ": " + err.message;
            this.showAuthError(msg);
            return { success: false, message: msg };
        }
    }

    async registerUser(emailOrId, password, name, role = "admin") {
        const errorBox = document.getElementById("auth-modal-error");
        if (errorBox) errorBox.style.display = "none";

        try {
            const res = await fetch("/api/auth/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email: emailOrId, password: password, name: name, role: role }),
            });
            const data = await res.json();
            if (res.ok && data.status === "success") {
                this.currentUser = data.user;
                this.token = data.token;
                try {
                    sessionStorage.setItem("thailpr_token", this.token);
                    sessionStorage.setItem("thailpr_user", JSON.stringify(this.currentUser));
                } catch (e) { }
                this.hideLoginModal();
                this.renderAuthUI();
                window.dispatchEvent(new CustomEvent("thailpr:auth-success", { detail: { user: this.currentUser, token: this.token } }));
                return { success: true };
            } else {
                const msg = data.detail || data.message || this.t("auth_reg_fail", "Registration failed");
                this.showAuthError(msg);
                return { success: false, message: msg };
            }
        } catch (err) {
            const msg = this.t("auth_net_err", "Network error") + ": " + err.message;
            this.showAuthError(msg);
            return { success: false, message: msg };
        }
    }

    async loginWithDevAdmin() {
        try {
            const res = await fetch("/api/auth/verify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id_token: "dev_admin_token" }),
            });
            if (res.ok) {
                const data = await res.json();
                this.currentUser = data.user;
                this.token = data.token || "dev_admin_token";
                try {
                    sessionStorage.setItem("thailpr_token", this.token);
                    sessionStorage.setItem("thailpr_user", JSON.stringify(this.currentUser));
                } catch (e) { }
                this.hideLoginModal();
                this.renderAuthUI();
                window.dispatchEvent(new CustomEvent("thailpr:auth-success", { detail: { user: this.currentUser, token: this.token } }));
            }
        } catch (e) {
            console.warn("[AUTH] Dev admin quick login error:", e);
        }
    }

    async logout() {
        try {
            await fetch("/api/auth/logout", { method: "POST" });
        } catch (e) { }
        this.currentUser = null;
        this.token = null;
        try {
            sessionStorage.removeItem("thailpr_token");
            sessionStorage.removeItem("thailpr_user");
            document.cookie = "auth_token=; Max-Age=0; path=/;";
        } catch (e) { }
        window.dispatchEvent(new CustomEvent("thailpr:logout"));
        this.renderAuthUI();
        this.showLoginModal();
    }

    showLoginModal(initialTab = "login") {
        let modal = document.getElementById("auth-login-modal");
        if (!modal) {
            this.createLoginModal();
            modal = document.getElementById("auth-login-modal");
        }
        if (modal) {
            modal.style.display = "flex";
            this.switchModalTab(initialTab);
            if (window.I18N) I18N.apply(modal);
        }
    }

    hideLoginModal() {
        const modal = document.getElementById("auth-login-modal");
        if (modal) modal.style.display = "none";
    }

    switchModalTab(tab) {
        const tabLogin = document.getElementById("tab-auth-login");
        const tabRegister = document.getElementById("tab-auth-register");
        const groupName = document.getElementById("group-auth-name");
        const groupRole = document.getElementById("group-auth-role");
        const btnSubmit = document.getElementById("btn-auth-submit");
        const modalTitle = document.getElementById("auth-modal-title");
        const errorBox = document.getElementById("auth-modal-error");

        if (errorBox) errorBox.style.display = "none";

        if (tab === "register") {
            if (tabLogin) { tabLogin.style.borderBottom = "2px solid transparent"; tabLogin.style.color = ""; tabLogin.classList.remove("active"); }
            if (tabRegister) { tabRegister.classList.add("active"); }
            if (groupName) groupName.style.display = "block";
            if (groupRole) groupRole.style.display = "block";
            if (btnSubmit) btnSubmit.textContent = this.t("btn_register", "Create Account");
            if (modalTitle) modalTitle.textContent = this.t("auth_title_register", "Create an LPR operator account");
            btnSubmit.dataset.mode = "register";
        } else {
            if (tabRegister) { tabRegister.style.borderBottom = "2px solid transparent"; tabRegister.style.color = ""; tabRegister.classList.remove("active"); }
            if (tabLogin) { tabLogin.classList.add("active"); }
            if (groupName) groupName.style.display = "none";
            if (groupRole) groupRole.style.display = "none";
            if (btnSubmit) btnSubmit.textContent = this.t("btn_signin", "Sign In");
            if (modalTitle) modalTitle.textContent = this.t("auth_title_login", "Sign in to the LPR dashboard");
            btnSubmit.dataset.mode = "login";
        }
    }

    createLoginModal() {
        const html = `
        <div id="auth-login-modal" class="auth-modal-overlay" style="display:none;">
            <div class="auth-modal-card">
                <button type="button" id="btn-auth-close-modal" class="auth-modal-close" aria-label="ปิด">&times;</button>

                <div class="auth-modal-header">
                    <div class="auth-brand-badge">
                        <span class="auth-brand-dot"></span>
                        <h2 class="auth-brand-title">LPR</h2>
                    </div>
                    <p id="auth-modal-title" class="auth-modal-subtitle">Sign in to the LPR dashboard</p>
                </div>

                <div class="auth-modal-tabs">
                    <button type="button" id="tab-auth-login" class="auth-tab-btn active" data-i18n="signin_tab">Sign In</button>
                    <button type="button" id="tab-auth-register" class="auth-tab-btn" data-i18n="register_tab">Register</button>
                </div>

                <div id="auth-modal-error" class="auth-modal-error" style="display:none;"></div>

                <div class="auth-form-fields">
                    <div id="group-auth-name" style="display:none;">
                        <label class="auth-label" data-i18n="lbl_name">Full name</label>
                        <input type="text" id="auth-name-input" class="auth-input" autocomplete="name">
                    </div>

                    <div id="group-auth-role" style="display:none;">
                        <label class="auth-label" data-i18n="lbl_role">User role</label>
                        <select id="auth-role-input" class="auth-input">
                            <option value="admin" data-i18n="role_admin">Administrator (full access)</option>
                            <option value="operator" data-i18n="role_operator">Operator (monitor &amp; review)</option>
                            <option value="viewer" data-i18n="role_viewer">Viewer (read only)</option>
                        </select>
                    </div>

                    <div>
                        <label class="auth-label" data-i18n="lbl_email">Email or username</label>
                        <input type="text" id="auth-email-input" class="auth-input" placeholder="admin@thailpr.local" autocomplete="username">
                    </div>

                    <div>
                        <label class="auth-label" data-i18n="lbl_password">Password</label>
                        <input type="password" id="auth-password-input" class="auth-input" placeholder="••••••••" autocomplete="current-password">
                    </div>

                    <button type="button" id="btn-auth-submit" class="auth-btn-primary" data-mode="login" data-i18n="btn_signin">
                        Sign In
                    </button>
                </div>
            </div>
        </div>
        `;
        document.body.insertAdjacentHTML("beforeend", html);

        document.getElementById("tab-auth-login")?.addEventListener("click", () => this.switchModalTab("login"));
        document.getElementById("tab-auth-register")?.addEventListener("click", () => this.switchModalTab("register"));
        document.getElementById("btn-auth-close-modal")?.addEventListener("click", () => this.hideLoginModal());

        document.getElementById("btn-auth-submit")?.addEventListener("click", async () => {
            const mode = document.getElementById("btn-auth-submit").dataset.mode || "login";
            const emailOrId = document.getElementById("auth-email-input").value.trim();
            const password = document.getElementById("auth-password-input").value;
            const name = document.getElementById("auth-name-input")?.value.trim() || "";
            const role = document.getElementById("auth-role-input")?.value || "admin";

            if (!emailOrId) { this.showAuthError(this.t("lbl_email", "Email or username")); return; }
            if (!password) { this.showAuthError(this.t("lbl_password", "Password")); return; }

            if (mode === "register") await this.registerUser(emailOrId, password, name, role);
            else await this.loginWithPassword(emailOrId, password);
        });

        const handleEnterKey = (e) => {
            if (e.key === "Enter") document.getElementById("btn-auth-submit")?.click();
        };
        document.getElementById("auth-email-input")?.addEventListener("keydown", handleEnterKey);
        document.getElementById("auth-password-input")?.addEventListener("keydown", handleEnterKey);
        document.getElementById("auth-name-input")?.addEventListener("keydown", handleEnterKey);
    }

    showAuthError(msg) {
        const errorBox = document.getElementById("auth-modal-error");
        if (errorBox) { errorBox.textContent = msg; errorBox.style.display = "block"; }
        else toast(msg, "error");
    }

    renderAuthUI() {
        const userContainer = document.getElementById("auth-user-container");
        if (!userContainer) return;

        const gearSvg = `<svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>`;
        const keySvg = `<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/></svg>`;
        const plusSvg = `<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"/></svg>`;

        if (this.currentUser) {
            const rawName = this.currentUser.name || this.currentUser.email || this.t("guest", "Guest");
            const initial = rawName.charAt(0).toUpperCase();
            const role = (this.currentUser.role || "admin").toUpperCase();
            const roleLabel = { ADMIN: this.t("role_admin_short", "Admin"), OPERATOR: this.t("role_operator_short", "Operator"), VIEWER: this.t("role_viewer_short", "Viewer") }[role] || role;

            userContainer.innerHTML = `
                <div class="auth-profile-badge" title="${rawName} (${roleLabel})">
                    <div class="auth-avatar-circle">${initial}</div>
                    <div class="auth-user-info">
                        <span class="auth-name-text">${rawName}</span>
                        <span class="auth-role-pill ${role.toLowerCase()}">${roleLabel}</span>
                    </div>
                    <button id="btn-auth-settings" class="auth-icon-btn" data-i18n-title="nav_settings" title="My Settings">${gearSvg}</button>
                    <button id="btn-auth-logout" class="auth-logout-btn" data-i18n="nav_logout" title="Sign Out">Sign Out</button>
                </div>
            `;

            document.getElementById("btn-auth-logout")?.addEventListener("click", () => this.logout());
            document.getElementById("btn-auth-settings")?.addEventListener("click", () => this.showUserSettingsModal());
        } else {
            userContainer.innerHTML = `
                <div class="auth-guest-actions">
                    <button id="btn-auth-open-login" class="auth-nav-btn login-btn">${keySvg} <span data-i18n="nav_login">Sign In</span></button>
                    <button id="btn-auth-open-register" class="auth-nav-btn register-btn">${plusSvg} <span data-i18n="register_tab">Register</span></button>
                </div>
            `;
            document.getElementById("btn-auth-open-login")?.addEventListener("click", () => this.showLoginModal("login"));
            document.getElementById("btn-auth-open-register")?.addEventListener("click", () => this.showLoginModal("register"));
        }

        // Sidebar user menu
        const sidebarUser = document.getElementById("sidebar-nav-user");
        if (sidebarUser) {
            if (this.currentUser) {
                sidebarUser.innerHTML = `
                    <button type="button" class="nav-item" id="navSettingsBtn">
                        ${gearSvg}
                        <span class="nav-label" data-i18n="nav_settings">My Settings</span>
                    </button>
                    <button type="button" class="nav-item" id="navLogoutBtn">
                        <svg width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/></svg>
                        <span class="nav-label" data-i18n="nav_logout">Sign Out</span>
                    </button>
                `;
                document.getElementById("navSettingsBtn")?.addEventListener("click", () => this.showUserSettingsModal());
                document.getElementById("navLogoutBtn")?.addEventListener("click", () => this.logout());
            } else {
                sidebarUser.innerHTML = `
                    <button type="button" class="nav-item" id="navLoginBtn">
                        ${keySvg}
                        <span class="nav-label" data-i18n="nav_login">Sign In</span>
                    </button>
                `;
                document.getElementById("navLoginBtn")?.addEventListener("click", () => this.showLoginModal("login"));
            }
        }
        if (window.I18N) I18N.apply();
    }

    showUserSettingsModal() {
        let modal = document.getElementById("user-settings-modal");
        if (!modal) {
            const html = `
            <div id="user-settings-modal" class="auth-modal-overlay" style="display:none;">
                <div class="auth-modal-card" style="max-width:440px;">
                    <button type="button" id="btn-settings-close" class="auth-modal-close">&times;</button>
                    <div class="auth-modal-header">
                        <h2 class="auth-brand-title"><span data-i18n="settings_title_1">User</span> <span style="color:var(--accent);" data-i18n="settings_title_2">Settings</span></h2>
                        <p class="auth-modal-subtitle"><span data-i18n="set_saved_for">Saved to account:</span> <strong>${this.currentUser?.email || ""}</strong></p>
                    </div>

                    <div class="auth-form-fields" style="margin-top:1rem;">
                        <div>
                            <label class="auth-label" data-i18n="set_conf">Default confidence threshold (0.1 - 0.99)</label>
                            <input type="number" id="setting-confidence" class="auth-input" min="0.1" max="0.99" step="0.05" value="${this.currentUser?.settings?.confidence_threshold || 0.50}">
                            <div style="font-size:0.72rem; color:var(--text-muted); margin-top:4px;" data-i18n="set_conf_hint">Higher = stricter</div>
                        </div>

                        <div>
                            <label class="auth-label" data-i18n="set_refresh">Auto-refresh history</label>
                            <select id="setting-auto-refresh" class="auth-input">
                                <option value="true" ${this.currentUser?.settings?.auto_refresh_history !== false ? "selected" : ""} data-i18n="set_refresh_on">Enabled (real-time)</option>
                                <option value="false" ${this.currentUser?.settings?.auto_refresh_history === false ? "selected" : ""} data-i18n="set_refresh_off">Disabled (manual)</option>
                            </select>
                        </div>

                        <div>
                            <label class="auth-label" data-i18n="set_debug">Default debug inspection</label>
                            <select id="setting-debug-view" class="auth-input">
                                <option value="false" ${!this.currentUser?.settings?.debug_view_default ? "selected" : ""} data-i18n="set_debug_off">Off (recommended)</option>
                                <option value="true" ${this.currentUser?.settings?.debug_view_default ? "selected" : ""} data-i18n="set_debug_on">On (inspect all AI steps)</option>
                            </select>
                        </div>

                        <button type="button" id="btn-save-settings" class="auth-btn-primary" style="margin-top:0.5rem;" data-i18n="set_save">
                            Save Settings
                        </button>
                    </div>
                </div>
            </div>
            `;
            document.body.insertAdjacentHTML("beforeend", html);
            modal = document.getElementById("user-settings-modal");

            document.getElementById("btn-settings-close")?.addEventListener("click", () => {
                modal.style.display = "none";
            });

            document.getElementById("btn-save-settings")?.addEventListener("click", async () => {
                const conf = parseFloat(document.getElementById("setting-confidence").value) || 0.50;
                const autoRefresh = document.getElementById("setting-auto-refresh").value === "true";
                const debugDefault = document.getElementById("setting-debug-view").value === "true";

                try {
                    const res = await fetch("/api/user/settings", {
                        method: "POST",
                        headers: this.getAuthHeaders(),
                        body: JSON.stringify({
                            confidence_threshold: conf,
                            auto_refresh_history: autoRefresh,
                            debug_view_default: debugDefault,
                        }),
                    });
                    if (res.ok) {
                        const data = await res.json();
                        if (this.currentUser) {
                            this.currentUser.settings = data.settings;
                            sessionStorage.setItem("thailpr_user", JSON.stringify(this.currentUser));
                        }
                        modal.style.display = "none";
                        if (window.LPRToast) window.LPRToast(this.t("toast_set_saved", "Settings saved"), "success");
                    } else {
                        if (window.LPRToast) window.LPRToast(this.t("toast_set_fail", "Failed to save settings"), "error");
                    }
                } catch (e) {
                    alert(this.t("toast_set_fail", "Failed to save settings") + ": " + e.message);
                }
            });
        }
        modal.style.display = "flex";
    }
}

window.authManager = new ThaiLPRAuthManager();
document.addEventListener("DOMContentLoaded", () => {
    window.authManager.init();
});
