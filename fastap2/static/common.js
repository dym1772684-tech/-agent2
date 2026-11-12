/* ============================================================
   共享前端工具：登录态、API 封装、提示、格式化。
   所有页面都先引本文件。
   ============================================================ */
(function (global) {
  "use strict";

  const API = "/api/v1";
  const KEY = {
    token: "ecare.token",
    identity: "ecare.identity",
    username: "ecare.username",
    role: "ecare.role",
    elder: "ecare.elder",
  };

  /* ---------------- 登录态 ---------------- */
  function getToken() { return localStorage.getItem(KEY.token) || ""; }
  function getIdentity() { return localStorage.getItem(KEY.identity) || ""; }

  function decodeJwt(token) {
    try {
      const part = token.split(".")[1];
      if (!part) return null;
      const b64 = part.replace(/-/g, "+").replace(/_/g, "/");
      const pad = "=".repeat((4 - (b64.length % 4)) % 4);
      return JSON.parse(decodeURIComponent(escape(atob(b64 + pad))));
    } catch (e) { return null; }
  }

  function payload() { return decodeJwt(getToken()) || {}; }
  function role() { return payload().role || localStorage.getItem(KEY.role) || "common"; }
  function userId() { return payload().user_id || ""; }
  function username() { return localStorage.getItem(KEY.username) || ""; }
  function isAdmin() { return role() === "admin"; }

  function saveSession(loginData, name) {
    localStorage.setItem(KEY.token, loginData.access_token || "");
    if (loginData.identity_id) localStorage.setItem(KEY.identity, loginData.identity_id);
    if (name) localStorage.setItem(KEY.username, name);
    const p = decodeJwt(loginData.access_token || "") || {};
    if (p.role) localStorage.setItem(KEY.role, p.role);
  }

  function clearSession() {
    Object.values(KEY).forEach((k) => { if (k !== KEY.elder) localStorage.removeItem(k); });
  }

  /* ---------------- 请求封装 ---------------- */
  class ApiError extends Error {
    constructor(status, detail) {
      super(detail || ("HTTP " + status));
      this.status = status;
      this.detail = detail;
    }
  }

  async function raw(path, opts) {
    opts = opts || {};
    const headers = Object.assign({}, opts.headers || {});
    const token = getToken();
    if (token && !headers.Authorization) headers.Authorization = "Bearer " + token;

    let body = opts.body;
    if (body !== undefined && body !== null && !(body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(body);
    }

    const resp = await fetch(API + path, {
      method: opts.method || "GET",
      headers,
      body,
    });

    if (resp.status === 401) {
      clearSession();
      if (!location.pathname.startsWith("/auth")) {
        location.replace("/auth?next=" + encodeURIComponent(location.pathname));
      }
      throw new ApiError(401, "登录已过期，请重新登录");
    }

    const text = await resp.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }

    if (!resp.ok) {
      const detail = (data && (data.detail || data.msg)) || text || ("HTTP " + resp.status);
      throw new ApiError(resp.status, typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  /** 统一剥掉 {code,msg,data} 信封；不是信封则原样返回。 */
  function unwrap(data) {
    if (data && typeof data === "object" && !Array.isArray(data) &&
        "code" in data && "msg" in data) {
      return data.data;
    }
    return data;
  }

  async function api(path, opts) { return unwrap(await raw(path, opts)); }

  /* ---------------- 提示 ---------------- */
  let toastWrap = null;
  function toast(message, kind, ms) {
    if (!toastWrap) {
      toastWrap = document.createElement("div");
      toastWrap.className = "toast-wrap";
      document.body.appendChild(toastWrap);
    }
    const el = document.createElement("div");
    el.className = "toast " + (kind || "");
    el.textContent = message;
    toastWrap.appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity .2s, transform .2s";
      el.style.opacity = "0";
      el.style.transform = "translateY(-8px)";
      setTimeout(() => el.remove(), 220);
    }, ms || 2600);
  }

  /* ---------------- 小工具 ---------------- */
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function fmtTime(ts) {
    if (!ts) return "";
    const d = typeof ts === "number" ? new Date(ts) : new Date(String(ts).replace(" ", "T"));
    if (isNaN(d.getTime())) return String(ts);
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getMonth() + 1}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  function copy(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(() => toast("已复制", "ok"));
    }
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); toast("已复制", "ok"); } catch (e) { toast("复制失败", "err"); }
    ta.remove();
  }

  function initials(name) {
    const s = (name || "").trim();
    if (!s) return "?";
    return /[a-zA-Z]/.test(s[0]) ? s[0].toUpperCase() : s[0];
  }

  /* ---------------- 页面守卫 ---------------- */
  function guard(needAdmin) {
    if (!getToken()) {
      location.replace("/auth?next=" + encodeURIComponent(location.pathname));
      return false;
    }
    if (needAdmin && !isAdmin()) {
      location.replace("/");
      return false;
    }
    return true;
  }

  function logout() {
    const token = getToken();
    clearSession();
    if (token) {
      fetch(API + "/logout", { method: "POST", headers: { Authorization: "Bearer " + token } })
        .catch(() => {}).finally(() => location.replace("/auth"));
    } else {
      location.replace("/auth");
    }
  }

  /* ---------------- SSE（fetch 流式解析） ---------------- */
  async function streamChat(body, handlers, opts) {
    const headers = { "Content-Type": "application/json", Accept: "text/event-stream" };
    const token = getToken();
    if (token) headers.Authorization = "Bearer " + token;

    const resp = await fetch(API + "/agent/chat/stream", {
      method: "POST", headers, body: JSON.stringify(body),
      signal: opts && opts.signal,
    });
    if (resp.status === 401) { clearSession(); location.replace("/auth"); throw new ApiError(401, "登录已过期"); }
    if (!resp.ok) {
      const t = await resp.text();
      let d = t;
      try { d = JSON.parse(t).detail || d; } catch (e) {}
      throw new ApiError(resp.status, d);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let event = "message";
    let dataLines = [];

    const flush = () => {
      if (!dataLines.length) { event = "message"; return; }
      const raw = dataLines.join("\n");
      dataLines = [];
      let payload = raw;
      try { payload = JSON.parse(raw); } catch (e) {}
      const ev = event;
      event = "message";
      if (handlers.onEvent) handlers.onEvent(ev, payload);
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).replace(/\r$/, "");
        buf = buf.slice(idx + 1);
        if (line === "") { flush(); continue; }
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
    }
    flush();
  }

  /* ---------------- 思考过程渲染 ---------------- */
  const NODE_LABEL = {
    supervisor: "调度主管", health: "健康专员", knowledge: "知识专员",
    life: "生活助手", aggregate: "汇总", guard: "校验", think: "思考",
    act: "执行", observe: "观察", finalize: "收尾", model: "模型",
    tools: "工具调用", llm: "模型", tool: "工具",
  };
  function nodeLabel(n) { return NODE_LABEL[n] || n || ""; }

  function fmtTokens(n) {
    if (n == null || isNaN(n)) return "—";
    return Number(n) >= 1000 ? (Number(n) / 1000).toFixed(1) + "k" : String(n);
  }
  function fmtCost(x) {
    if (x == null || isNaN(x)) return "—";
    return "¥" + Number(x).toFixed(4);
  }
  function fmtMs(ms) {
    if (ms == null || isNaN(ms)) return "";
    return Number(ms) >= 1000 ? (Number(ms) / 1000).toFixed(1) + "s" : Math.round(ms) + "ms";
  }

  /** 汇总一轮里所有模型调用的 token 与成本。 */
  function sumUsage(events) {
    let calls = 0, input = 0, output = 0, cached = 0, cost = 0;
    (events || []).forEach((e) => {
      if (e && e.type === "llm_end" && e.tokens) {
        calls += 1;
        input += e.tokens.input_tokens || 0;
        output += e.tokens.output_tokens || 0;
        cached += e.tokens.cache_read || 0;
        cost += Number(e.cost || 0);
      }
    });
    return { calls, input, output, cached, cost };
  }

  /**
   * 把 observe.py 的原始事件压成可读步骤。
   * 事件里有 node_start / node_end / llm_start / llm_end / tool_start / tool_end 六种，
   * 这里只保留「思考」「工具」「节点收尾」「告警」四类，其余当噪音丢掉。
   */
  function buildTraceSteps(events) {
    const steps = [];
    const pending = new Map();  // tool_start 先进、tool_end 后到，按名字配对
    (events || []).forEach((ev, i) => {
      if (!ev || typeof ev !== "object") return;
      const type = ev.type || "";
      const seq = ev.seq == null ? i : ev.seq;

      if (type === "llm_end") {
        steps.push({
          kind: "think", seq, node: ev.node, duration: ev.duration_ms,
          tokens: ev.tokens || {}, cost: ev.cost, thinking: ev.thinking,
          text: ev.text, calls: ev.tool_calls || [], error: ev.error,
        });
      } else if (type === "tool_start") {
        pending.set(ev.name + "|" + seq, ev);
      } else if (type === "tool_end") {
        let key = null;
        for (const k of pending.keys()) {
          if (k.startsWith((ev.name || "") + "|")) { key = k; break; }
        }
        const start = key ? pending.get(key) : null;
        if (key) pending.delete(key);
        steps.push({
          kind: "tool", seq, node: ev.node, name: ev.name,
          args: start && start.args, duration: ev.duration_ms,
          content: ev.content, isError: ev.is_error, reused: ev.reused,
        });
      } else if (type === "node_end") {
        steps.push({
          kind: "node", seq, node: ev.node, duration: ev.duration_ms,
          result: ev.result, next: ev.next_action,
        });
      } else if (["guard", "interrupted", "error", "structured_end"].indexOf(type) >= 0) {
        steps.push({
          kind: "alert", seq, node: ev.node, type,
          level: type === "error" ? "err" : "warn",
          detail: ev.error || ev.summary || ev.rule || ev.next_action || type,
        });
      }
    });
    // 还在执行中的工具（流式过程中会出现）
    pending.forEach((start) => steps.push({
      kind: "tool", pending: true, node: start.node, name: start.name, args: start.args,
    }));
    steps.sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0));
    return steps;
  }

  /** 渲染步骤列表（不含外层面板）。 */
  function renderTrace(events, opts) {
    opts = opts || {};
    const steps = buildTraceSteps(events);
    if (!steps.length) {
      return '<div class="step"><div class="step-main"><div class="step-body dim">还没有可展示的步骤</div></div></div>';
    }
    return steps.map((s) => {
      if (s.kind === "think") {
        const tk = s.tokens || {};
        const meta = [
          s.node ? nodeLabel(s.node) : "",
          fmtMs(s.duration),
          tk.input_tokens != null ? "输入 " + fmtTokens(tk.input_tokens) : "",
          tk.output_tokens != null ? "输出 " + fmtTokens(tk.output_tokens) : "",
          tk.cache_read ? "缓存 " + fmtTokens(tk.cache_read) : "",
          s.cost != null ? fmtCost(s.cost) : "",
        ].filter(Boolean).join(" · ");
        const calls = (s.calls || []).map((c) => {
          const args = typeof c.args === "string" ? c.args : JSON.stringify(c.args || {});
          const short = args.length > 76 ? args.slice(0, 76) + "…" : args;
          return `<span class="call-chip">→ ${esc(c.name || "?")}(${esc(short)})</span>`;
        }).join("");
        const body = [
          s.thinking ? `<div><b>思考：</b>${esc(s.thinking)}</div>` : "",
          s.text ? `<div>${esc(s.text)}</div>` : "",
          s.error ? `<div style="color:var(--danger)">✕ ${esc(s.error)}</div>` : "",
        ].filter(Boolean).join("");
        return `<div class="step">
          <div class="step-ico">✻</div>
          <div class="step-main">
            <div class="step-title">思考${s.node ? " · " + esc(nodeLabel(s.node)) : ""}</div>
            <div class="step-meta">${esc(meta)}</div>
            ${body ? `<div class="step-body">${body}</div>` : ""}
            ${calls ? `<div>${calls}</div>` : ""}
          </div></div>`;
      }

      if (s.kind === "tool") {
        const meta = [
          fmtMs(s.duration),
          s.reused ? "复用上次结果" : "",
          s.isError ? "执行失败" : "",
        ].filter(Boolean).join(" · ");
        return `<div class="step">
          <div class="step-ico tool">${s.pending ? '<span class="spin-mini"></span>' : "⚙"}</div>
          <div class="step-main">
            <div class="step-title">调用工具 · ${esc(s.name || "")}</div>
            <div class="step-meta">${esc(meta || (s.pending ? "执行中…" : ""))}</div>
            ${s.args ? `<div class="step-body dim">参数：${esc(String(s.args))}</div>` : ""}
            ${s.content ? `<div class="step-body">${esc(String(s.content))}</div>` : ""}
          </div></div>`;
      }

      if (s.kind === "node") {
        let body = "";
        if (s.result && typeof s.result === "object") {
          let txt = JSON.stringify(s.result);
          if (txt.length > 420) txt = txt.slice(0, 420) + "…";
          body = `<div class="step-body dim">${esc(txt)}</div>`;
        } else if (typeof s.result === "string" && s.result) {
          body = `<div class="step-body">${esc(s.result)}</div>`;
        }
        const meta = [fmtMs(s.duration), s.next ? "下一步：" + s.next : ""].filter(Boolean).join(" · ");
        return `<div class="step">
          <div class="step-ico node">✓</div>
          <div class="step-main">
            <div class="step-title">${esc(nodeLabel(s.node))} 完成</div>
            <div class="step-meta">${esc(meta)}</div>
            ${body}
          </div></div>`;
      }

      return `<div class="step">
        <div class="step-ico ${s.level}">!</div>
        <div class="step-main">
          <div class="step-title">${esc(s.type)}${s.node ? " · " + esc(nodeLabel(s.node)) : ""}</div>
          <div class="step-body">${esc(String(s.detail || ""))}</div>
        </div></div>`;
    }).join("");
  }

  /** 一行 token 摘要：N 次调用 · 输入 x / 输出 y · 缓存 z · ¥cost */
  function usageLine(usage) {
    if (!usage) return "";
    const parts = [];
    if (usage.calls != null) parts.push(`<b>${usage.calls}</b> 次模型调用`);
    if (usage.input != null) parts.push(`输入 <b>${fmtTokens(usage.input)}</b>`);
    if (usage.output != null) parts.push(`输出 <b>${fmtTokens(usage.output)}</b>`);
    if (usage.cached) parts.push(`缓存命中 <b>${fmtTokens(usage.cached)}</b>`);
    if (usage.cost) parts.push(`成本 <b>${fmtCost(usage.cost)}</b>`);
    return parts.join(" · ");
  }

  global.App = {
    API, KEY, getToken, getIdentity, payload, role, userId, username,
    isAdmin, saveSession, clearSession, api, raw, unwrap,
    toast, esc, fmtTime, copy, initials, guard, logout, streamChat, ApiError,
    buildTraceSteps, renderTrace, sumUsage, usageLine,
    fmtTokens, fmtCost, fmtMs, nodeLabel,
  };
})(window);
