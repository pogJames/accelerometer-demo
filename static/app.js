// Shared helpers used by dashboard.html and record.html.

// Only assigns when the value differs — avoids touching the DOM at all when
// the displayed string would be identical.
export function setIfChanged(el, attr, value) {
  if (el[attr] !== value) el[attr] = value;
}

export function fmtNumber(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toLocaleString();
}

export function fmtBytes(b) {
  if (b == null) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0, v = b;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

export function fmtRelative(ts) {
  if (!ts) return "—";
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleString();
}

export function showToast(msg, { error = false, durationMs = 3500 } = {}) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("aside");
    el.id = "toast";
    el.className = "toast";
    el.setAttribute("role", "status");
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.toggle("error", !!error);
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), durationMs);
}
