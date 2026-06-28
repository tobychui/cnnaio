/* =====================================================================
 * arozos.js  —  ArozOS Design Framework runtime
 * ---------------------------------------------------------------------
 * Companion to arozos.css. Provides:
 *   - Theme management (light / dark) driven by <body class="white|dark">
 *   - Animation helpers (fade / scale / slide in & out)
 *   - Interactive behaviours: modal, accordion, tab, toast
 *
 * jQuery is OPTIONAL. If jQuery is present, convenience plugins are also
 * registered ($.fn.aoModal, $.fn.aoTab, $.fn.aoAccordion, $.fn.aoFadeIn,
 * $.fn.aoFadeOut). Everything also works through the global `arozos` API
 * and via declarative data-attributes (no JS wiring required).
 *
 * Declarative attributes (auto-wired on DOMContentLoaded):
 *   data-ao-toggle="theme"                 click -> toggle light/dark
 *   data-ao-toggle="modal"  data-ao-target="#id"   click -> open modal
 *   data-ao-dismiss="modal"                click -> close enclosing modal
 *   data-ao-toggle="accordion"             click -> toggle next .content
 *   data-ao-tab="name"      (inside .menu) click -> show .ui.tab[data-tab=name]
 *
 * Usage:
 *   <script src="jquery.min.js"></script>   (optional)
 *   <script src="arozos.js"></script>
 *   arozos.theme.toggle();
 *   arozos.toast("Saved", { type: "success" });
 * ===================================================================== */

(function (root, factory) {
    var api = factory();
    root.arozos = api;
    /* Register jQuery plugins if jQuery is on the page. */
    if (root.jQuery) { api._bindJQuery(root.jQuery); }
})(typeof window !== "undefined" ? window : this, function () {
    "use strict";

    var STORAGE_KEY = "arozos-theme";
    var themeListeners = [];

    /* -----------------------------------------------------------------
     * Theme
     * ----------------------------------------------------------------- */

    var theme = {
        /* Returns "dark" or "white". */
        get: function () {
            return document.body.classList.contains("dark") ? "dark" : "white";
        },

        /* Apply a theme. mode = "dark" | "white" (anything else -> "white"). */
        set: function (mode, opts) {
            opts = opts || {};
            var dark = (mode === "dark" || mode === "darkTheme" || mode === true);
            document.body.classList.toggle("dark", dark);
            document.body.classList.toggle("white", !dark);
            if (opts.persist !== false) {
                try { localStorage.setItem(STORAGE_KEY, dark ? "dark" : "white"); } catch (e) {}
            }
            var resolved = dark ? "dark" : "white";
            themeListeners.forEach(function (fn) {
                try { fn(resolved, dark); } catch (e) {}
            });
            /* ArozOS embedding hook: some host shells look for this callback. */
            if (typeof window.detailPageThemeCallback === "function") {
                try { window.detailPageThemeCallback(dark); } catch (e) {}
            }
            return resolved;
        },

        /* Flip between light and dark. */
        toggle: function () {
            return theme.set(theme.get() === "dark" ? "white" : "dark");
        },

        /* Subscribe to changes; returns an unsubscribe function. */
        onChange: function (fn) {
            if (typeof fn !== "function") return function () {};
            themeListeners.push(fn);
            return function () {
                var i = themeListeners.indexOf(fn);
                if (i !== -1) themeListeners.splice(i, 1);
            };
        },

        /* Resolve the initial theme. Priority:
         *   1. an explicit class already on <body>
         *   2. ArozOS host theme (preferredTheme / ao_module_getSystemThemeColor)
         *   3. a previously persisted choice
         *   4. the OS prefers-color-scheme
         *   5. light
         * Pass {persist:false} so auto-detection does not overwrite a stored
         * user preference. */
        init: function (opts) {
            opts = opts || {};

            /* (1) honour an author-set class verbatim. */
            if (document.body.classList.contains("dark"))  return theme.set("dark",  { persist: false });
            if (document.body.classList.contains("white")) return theme.set("white", { persist: false });

            /* (2) ArozOS host hints. */
            var host = null;
            try {
                if (typeof window.preferredTheme !== "undefined" && window.preferredTheme) host = window.preferredTheme;
                else if (window.parent && typeof window.parent.preferredTheme !== "undefined") host = window.parent.preferredTheme;
            } catch (e) {}
            if (host) return theme.set(host, { persist: false });

            if (typeof window.ao_module_getSystemThemeColor === "function") {
                try {
                    window.ao_module_getSystemThemeColor(function (c) {
                        theme.set(c !== "whiteTheme" ? "dark" : "white", { persist: false });
                    });
                    return theme.get();
                } catch (e) {}
            }

            /* (3) persisted user choice. */
            try {
                var saved = localStorage.getItem(STORAGE_KEY);
                if (saved) return theme.set(saved, { persist: false });
            } catch (e) {}

            /* (4) OS preference. */
            if (opts.respectSystem !== false &&
                window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
                return theme.set("dark", { persist: false });
            }

            /* (5) default. */
            return theme.set("white", { persist: false });
        }
    };

    /* -----------------------------------------------------------------
     * Animation helpers (vanilla, Promise-returning)
     * ----------------------------------------------------------------- */

    function resolveEl(target) {
        if (!target) return null;
        if (typeof target === "string") return document.querySelector(target);
        if (target.jquery) return target[0];
        if (target.nodeType) return target;
        return null;
    }

    var anim = {
        /* Show + fade/scale/slide in. */
        in: function (target, opts) {
            opts = opts || {};
            var el = resolveEl(target);
            if (!el) return Promise.resolve();
            var name = opts.effect || "fade";       // fade | scale | up | down
            var dur  = opts.duration || 180;
            el.style.display = opts.display || "";
            el.style.animation = "none";
            /* force reflow so the animation restarts reliably */
            void el.offsetWidth;
            el.style.animation = keyframeFor(name) + " " + dur + "ms cubic-bezier(0.4,0,0.2,1) both";
            return done(el, dur, opts.done);
        },

        /* Fade/scale/slide out, then hide. */
        out: function (target, opts) {
            opts = opts || {};
            var el = resolveEl(target);
            if (!el) return Promise.resolve();
            var dur = opts.duration || 160;
            el.style.animation = "aoFadeOut " + dur + "ms cubic-bezier(0.4,0,0.2,1) both";
            return new Promise(function (resolve) {
                setTimeout(function () {
                    if (opts.remove) { if (el.parentNode) el.parentNode.removeChild(el); }
                    else el.style.display = "none";
                    if (typeof opts.done === "function") opts.done(el);
                    resolve(el);
                }, dur);
            });
        }
    };

    /* Map a short effect name to its CSS @keyframes identifier (see arozos.css). */
    function keyframeFor(name) {
        switch (name) {
            case "scale": return "aoScaleIn";
            case "up":    return "aoSlideUp";
            case "down":  return "aoSlideDown";
            case "fade":
            default:      return "aoFadeIn";
        }
    }
    function done(el, dur, cb) {
        return new Promise(function (resolve) {
            setTimeout(function () { if (typeof cb === "function") cb(el); resolve(el); }, dur);
        });
    }

    /* -----------------------------------------------------------------
     * Modal
     * ----------------------------------------------------------------- */

    function modal(target) {
        var el = resolveEl(target);
        /* Accept either the .ui.dimmer overlay or the inner .ui.modal. */
        var dimmer = el && el.classList && el.classList.contains("dimmer")
            ? el
            : (el ? el.closest(".ui.dimmer") || el.parentNode : null);

        return {
            el: dimmer,
            show: function () {
                if (!dimmer) return this;
                dimmer.classList.add("active");
                document.body.classList.add("ao-modal-open");
                return this;
            },
            hide: function () {
                if (!dimmer) return this;
                dimmer.classList.remove("active");
                if (!document.querySelector(".ui.dimmer.active")) {
                    document.body.classList.remove("ao-modal-open");
                }
                return this;
            },
            toggle: function () {
                if (!dimmer) return this;
                return dimmer.classList.contains("active") ? this.hide() : this.show();
            }
        };
    }

    /* -----------------------------------------------------------------
     * Accordion / collapsible
     * ----------------------------------------------------------------- */

    /* Resolve the collapsible content controlled by an accordion trigger:
     * an explicit data-ao-target, else the next sibling, else a child .content */
    function accordionContent(trigger) {
        var sel = trigger.getAttribute("data-ao-target");
        return sel ? document.querySelector(sel)
                   : (trigger.nextElementSibling || trigger.querySelector(".content"));
    }

    function toggleAccordion(trigger) {
        var t = resolveEl(trigger);
        if (!t) return;
        var content = accordionContent(t);
        if (!content) return;
        var open = !content.classList.contains("open");
        content.classList.toggle("open", open);
        /* Drive display directly so it works with any wrapper (not just
         * `.ui.accordion`); empty string lets the stylesheet take over. */
        content.style.display = open ? "" : "none";
        var chevron = t.querySelector(".ui.chevron, .chevron");
        if (chevron) chevron.classList.toggle("open", open);
        return open;
    }

    /* Collapse accordion targets on boot unless explicitly marked .open. This
     * makes collapsibles work even without the `.ui.accordion` wrapper class. */
    function initAccordions() {
        var triggers = document.querySelectorAll("[data-ao-toggle='accordion']");
        Array.prototype.forEach.call(triggers, function (t) {
            var content = accordionContent(t);
            if (content && !content.classList.contains("open")) content.style.display = "none";
        });
    }

    /* -----------------------------------------------------------------
     * Tabs
     * ----------------------------------------------------------------- */

    function showTab(name, scope) {
        var ctx = resolveEl(scope) || document;
        /* Activate the matching menu item(s). */
        var items = ctx.querySelectorAll("[data-ao-tab]");
        items.forEach(function (it) {
            it.classList.toggle("active", it.getAttribute("data-ao-tab") === name);
        });
        /* Show the matching pane, hide siblings. */
        var panes = (scope ? ctx : document).querySelectorAll(".ui.tab");
        panes.forEach(function (p) {
            p.classList.toggle("active", p.getAttribute("data-tab") === name);
        });
    }

    /* -----------------------------------------------------------------
     * Toast
     * ----------------------------------------------------------------- */

    function toastHost() {
        var host = document.querySelector(".ao-toast-host");
        if (!host) {
            host = document.createElement("div");
            host.className = "ao-toast-host";
            document.body.appendChild(host);
        }
        return host;
    }

    function toast(message, opts) {
        opts = opts || {};
        var host = toastHost();
        var el = document.createElement("div");
        el.className = "ao-toast" + (opts.type ? " " + opts.type : "");
        if (opts.title) {
            var t = document.createElement("div");
            t.className = "ao-toast-title";
            t.textContent = opts.title;
            el.appendChild(t);
        }
        var body = document.createElement("div");
        body.textContent = message;
        el.appendChild(body);
        host.appendChild(el);

        var life = opts.duration === 0 ? 0 : (opts.duration || 3200);
        function dismiss() {
            el.classList.add("hiding");
            setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 220);
        }
        el.addEventListener("click", dismiss);
        if (life > 0) setTimeout(dismiss, life);
        return { el: el, dismiss: dismiss };
    }

    /* -----------------------------------------------------------------
     * Declarative wiring (delegated; works for dynamic content too)
     * ----------------------------------------------------------------- */

    function wireDeclarative() {
        document.addEventListener("click", function (e) {
            var toggleEl = e.target.closest("[data-ao-toggle]");
            if (toggleEl) {
                var kind = toggleEl.getAttribute("data-ao-toggle");
                if (kind === "theme")     { e.preventDefault(); theme.toggle(); return; }
                if (kind === "modal")     { e.preventDefault(); modal(toggleEl.getAttribute("data-ao-target")).show(); return; }
                if (kind === "accordion") { e.preventDefault(); toggleAccordion(toggleEl); return; }
            }

            var dismissEl = e.target.closest("[data-ao-dismiss='modal']");
            if (dismissEl) { e.preventDefault(); modal(dismissEl).hide(); return; }

            var tabEl = e.target.closest("[data-ao-tab]");
            if (tabEl) {
                e.preventDefault();
                var menu = tabEl.closest(".ui.menu") || tabEl.parentNode;
                showTab(tabEl.getAttribute("data-ao-tab"), menu && menu.parentNode);
                return;
            }

            /* Dismiss a message via its .close button. */
            var msgClose = e.target.closest(".ui.message > .close");
            if (msgClose) anim.out(msgClose.parentNode, { remove: true });
        });

        /* Close on a mousedown that starts on the dimmer backdrop itself.
         * Listening on mousedown (not click) prevents a text-selection drag
         * that begins inside the modal and releases on the backdrop from
         * closing the modal: a click would otherwise fire on the dimmer as the
         * common ancestor of the press and release targets. */
        document.addEventListener("mousedown", function (e) {
            if (e.target.classList && e.target.classList.contains("dimmer") && e.target.classList.contains("active")) {
                modal(e.target).hide();
            }
        });

        /* Esc closes the top-most open modal. */
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") {
                var open = document.querySelectorAll(".ui.dimmer.active");
                if (open.length) modal(open[open.length - 1]).hide();
            }
        });
    }

    /* -----------------------------------------------------------------
     * jQuery plugins (registered only if jQuery exists)
     * ----------------------------------------------------------------- */

    function bindJQuery($) {
        $.fn.aoModal = function (action) {
            this.each(function () { modal(this)[action || "show"](); });
            return this;
        };
        $.fn.aoAccordion = function () {
            this.each(function () {
                $(this).on("click", function () { toggleAccordion(this); });
            });
            return this;
        };
        $.fn.aoTab = function () {
            this.each(function () {
                $(this).on("click", function () {
                    var menu = this.closest(".ui.menu") || this.parentNode;
                    showTab($(this).attr("data-ao-tab"), menu && menu.parentNode);
                });
            });
            return this;
        };
        $.fn.aoFadeIn  = function (opts) { this.each(function () { anim.in(this, opts); }); return this; };
        $.fn.aoFadeOut = function (opts) { this.each(function () { anim.out(this, opts); }); return this; };
        $.aoToast = toast;
        $.aoTheme = theme;
    }

    /* -----------------------------------------------------------------
     * Boot
     * ----------------------------------------------------------------- */

    function boot() {
        theme.init();
        initAccordions();
        wireDeclarative();
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }

    /* Public API */
    return {
        theme:     theme,
        anim:      anim,
        animate:   anim,
        modal:     modal,
        toast:     toast,
        showTab:   showTab,
        accordion: toggleAccordion,
        _bindJQuery: bindJQuery
    };
});
