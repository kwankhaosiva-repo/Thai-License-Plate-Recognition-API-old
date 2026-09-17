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
                const msg = data.detail || data.message || "Invalid Email/ID or Password";
                if (errorBox) {
                    errorBox.textContent = msg;
                    errorBox.style.display = "block";
                } else {
                    alert(msg);
                }
                return { success: false, message: msg };
            }
        } catch (err) {
            const msg = "Network error: " + err.message;
            if (errorBox) {
                errorBox.textContent = msg;
                errorBox.style.display = "block";
            }
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
                alert(`Account successfully created for ${data.user.name || data.user.email}!`);
                this.hideLoginModal();
                this.renderAuthUI();
                window.dispatchEvent(new CustomEvent("thailpr:auth-success", { detail: { user: this.currentUser, token: this.token } }));
                return { success: true };
            } else {
                const msg = data.detail || data.message || "Registration failed";
                if (errorBox) {
                    errorBox.textContent = msg;
                    errorBox.style.display = "block";
                } else {
                    alert(msg);
                }
                return { success: false, message: msg };
            }
        } catch (err) {
            const msg = "Network error: " + err.message;
            if (errorBox) {
                errorBox.textContent = msg;
                errorBox.style.display = "block";
            }
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
            if (tabLogin) { tabLogin.style.borderBottom = "2px solid transparent"; tabLogin.style.color = "#94a3b8"; }
            if (tabRegister) { tabRegister.style.borderBottom = "2px solid #06b6d4"; tabRegister.style.color = "#fff"; }
            if (groupName) groupName.style.display = "block";
            if (groupRole) groupRole.style.display = "block";
            if (btnSubmit) btnSubmit.textContent = "Create Account";
            if (modalTitle) modalTitle.textContent = "Create New LPR Operator Account";
            btnSubmit.dataset.mode = "register";
        } else {
            if (tabRegister) { tabRegister.style.borderBottom = "2px solid transparent"; tabRegister.style.color = "#94a3b8"; }
            if (tabLogin) { tabLogin.style.borderBottom = "2px solid #06b6d4"; tabLogin.style.color = "#fff"; }
            if (groupName) groupName.style.display = "none";
            if (groupRole) groupRole.style.display = "none";
            if (btnSubmit) btnSubmit.textContent = "Sign In to LPR System";
            if (modalTitle) modalTitle.textContent = "Sign In to Thai LPR Dashboard";
            btnSubmit.dataset.mode = "login";
        }
    }

    createLoginModal() {
        const dbType = this.authConfig?.db_provider === "firestore" ? "Cloud Firestore" : "SQLite Database";
        const html = `
        <div id="auth-login-modal" class="auth-modal-overlay" style="display:none;">
            <div class="auth-modal-card">
                <!-- Close Button -->
                <button type="button" id="btn-auth-close-modal" class="auth-modal-close" aria-label="Close">&times;</button>

                <!-- Header Branding -->
                <div class="auth-modal-header">
                    <div class="auth-brand-badge">
                        <span class="auth-brand-dot"></span>
                        <h2 class="auth-brand-title">Thai <span style="color:var(--accent-cyan, #06b6d4);">LPR</span></h2>
                    </div>
                    <p id="auth-modal-title" class="auth-modal-subtitle">Sign In to Thai LPR Dashboard</p>
                    <div class="auth-db-tag">
                        <span>Database:</span> <strong>${dbType}</strong>
                    </div>
                </div>

                <!-- Tabs: Sign In vs Register -->
                <div class="auth-modal-tabs">
                    <button type="button" id="tab-auth-login" class="auth-tab-btn active">Sign In</button>
                    <button type="button" id="tab-auth-register" class="auth-tab-btn">Create Account</button>
                </div>

                <!-- Error Container -->
                <div id="auth-modal-error" class="auth-modal-error" style="display:none;"></div>

                <!-- Form Inputs -->
                <div class="auth-form-fields">
                    <div id="group-auth-name" style="display:none;">
                        <label class="auth-label">Full Name</label>
                        <input type="text" id="auth-name-input" class="auth-input" placeholder="e.g. Kwankhao Sivasomboon" autocomplete="name">
                    </div>

                    <div id="group-auth-role" style="display:none;">
                        <label class="auth-label">User Role</label>
                        <select id="auth-role-input" class="auth-input">
                            <option value="admin">Administrator (Full Access)</option>
                            <option value="operator">Operator (Monitor & Review)</option>
                            <option value="viewer">Viewer (Read Only)</option>
                        </select>
                    </div>

                    <div>
                        <label class="auth-label">Email or User ID</label>
                        <input type="text" id="auth-email-input" class="auth-input" placeholder="admin@thailpr.local or username" autocomplete="username">
                    </div>

                    <div>
                        <label class="auth-label">Password</label>
                        <input type="password" id="auth-password-input" class="auth-input" placeholder="••••••••" autocomplete="current-password">
                    </div>

                    <button type="button" id="btn-auth-submit" class="auth-btn-primary" data-mode="login">
                        Sign In to LPR System
                    </button>

                    <div class="auth-divider">
                        <span>OR</span>
                    </div>

                    <button type="button" id="btn-auth-dev-admin" class="auth-btn-secondary">
                        ⚡ Quick Login (Localhost Admin)
                    </button>
                </div>
            </div>
        </div>
        `;
        document.body.insertAdjacentHTML("beforeend", html);

        // Bind events
        document.getElementById("tab-auth-login")?.addEventListener("click", () => this.switchModalTab("login"));
        document.getElementById("tab-auth-register")?.addEventListener("click", () => this.switchModalTab("register"));
        document.getElementById("btn-auth-dev-admin")?.addEventListener("click", () => this.loginWithDevAdmin());
        document.getElementById("btn-auth-close-modal")?.addEventListener("click", () => this.hideLoginModal());

        document.getElementById("btn-auth-submit")?.addEventListener("click", async () => {
            const mode = document.getElementById("btn-auth-submit").dataset.mode || "login";
            const emailOrId = document.getElementById("auth-email-input").value.trim();
            const password = document.getElementById("auth-password-input").value;
            const name = document.getElementById("auth-name-input")?.value.trim() || "";
            const role = document.getElementById("auth-role-input")?.value || "admin";

            if (!emailOrId) {
                alert("Please enter your Email or User ID");
                return;
            }
            if (!password) {
                alert("Please enter your Password");
                return;
            }

            if (mode === "register") {
                await this.registerUser(emailOrId, password, name, role);
            } else {
                await this.loginWithPassword(emailOrId, password);
            }
        });

        const handleEnterKey = (e) => {
            if (e.key === "Enter") {
                document.getElementById("btn-auth-submit")?.click();
            }
        };
        document.getElementById("auth-email-input")?.addEventListener("keydown", handleEnterKey);
        document.getElementById("auth-password-input")?.addEventListener("keydown", handleEnterKey);
        document.getElementById("auth-name-input")?.addEventListener("keydown", handleEnterKey);
    }

    renderAuthUI() {
        const userContainer = document.getElementById("auth-user-container");
        if (!userContainer) return;

        if (this.currentUser) {
            const rawName = this.currentUser.name || this.currentUser.email || "Operator";
            const initial = rawName.charAt(0).toUpperCase();
            const role = (this.currentUser.role || "admin").toUpperCase();

            userContainer.innerHTML = `
                <div class="auth-profile-badge" title="Logged in as ${rawName} (${role})">
                    <div class="auth-avatar-circle">${initial}</div>
                    <div class="auth-user-info">
                        <span class="auth-name-text">${rawName}</span>
                        <span class="auth-role-pill ${role.toLowerCase()}">${role}</span>
                    </div>
                    <button id="btn-auth-settings" class="auth-icon-btn" title="User Settings & Preferences">⚙️</button>
                    <button id="btn-auth-logout" class="auth-logout-btn" title="Sign Out">Logout</button>
                </div>
            `;

            document.getElementById("btn-auth-logout")?.addEventListener("click", () => this.logout());
            document.getElementById("btn-auth-settings")?.addEventListener("click", () => this.showUserSettingsModal());
        } else {
            userContainer.innerHTML = `
                <div class="auth-guest-actions">
                    <button id="btn-auth-open-login" class="auth-nav-btn login-btn">
                        🔑 Sign In
                    </button>
                    <button id="btn-auth-open-register" class="auth-nav-btn register-btn">
                        + Register
                    </button>
                </div>
            `;
            document.getElementById("btn-auth-open-login")?.addEventListener("click", () => this.showLoginModal("login"));
            document.getElementById("btn-auth-open-register")?.addEventListener("click", () => this.showLoginModal("register"));
        }
    }

    showUserSettingsModal() {
        let modal = document.getElementById("user-settings-modal");
        if (!modal) {
            const html = `
            <div id="user-settings-modal" class="auth-modal-overlay" style="display:none;">
                <div class="auth-modal-card" style="max-width:440px;">
                    <button type="button" id="btn-settings-close" class="auth-modal-close">&times;</button>
                    <div class="auth-modal-header">
                        <h2 class="auth-brand-title">User <span style="color:var(--accent-cyan, #06b6d4);">Settings</span></h2>
                        <p class="auth-modal-subtitle">Saved in database for account: <strong>${this.currentUser?.email || ""}</strong></p>
                    </div>

                    <div class="auth-form-fields" style="margin-top:1rem;">
                        <div>
                            <label class="auth-label">Default Confidence Threshold</label>
                            <input type="number" id="setting-confidence" class="auth-input" min="0.1" max="0.99" step="0.05" value="${this.currentUser?.settings?.confidence_threshold || 0.50}">
                        </div>

                        <div>
                            <label class="auth-label">Auto Refresh History Feed</label>
                            <select id="setting-auto-refresh" class="auth-input">
                                <option value="true" ${this.currentUser?.settings?.auto_refresh_history !== false ? "selected" : ""}>Enabled (Real-time updates)</option>
                                <option value="false" ${this.currentUser?.settings?.auto_refresh_history === false ? "selected" : ""}>Disabled (Manual refresh only)</option>
                            </select>
                        </div>

                        <div>
                            <label class="auth-label">Default Debug Breakdown</label>
                            <select id="setting-debug-view" class="auth-input">
                                <option value="false" ${!this.currentUser?.settings?.debug_view_default ? "selected" : ""}>Off (Saves bandwidth & CPU)</option>
                                <option value="true" ${this.currentUser?.settings?.debug_view_default ? "selected" : ""}>On (Inspect all intermediate crops)</option>
                            </select>
                        </div>

                        <button type="button" id="btn-save-settings" class="auth-btn-primary" style="margin-top:0.5rem;">
                            Save Settings to Database
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
                        alert("Settings successfully saved to database!");
                        modal.style.display = "none";
                    } else {
                        alert("Failed to save settings.");
                    }
                } catch (e) {
                    alert("Error saving settings: " + e.message);
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
