const safe = (value) => String(value ?? "");

export const money = (value) => Number(value || 0).toLocaleString(undefined, {
  currency: "USD",
  maximumFractionDigits: 6,
  style: "currency"
});

export const number = (value) => Number(value || 0).toLocaleString();
export const percent = (value) => `${Number(value || 0).toFixed(1)}%`;

export function escapeHtml(value) {
  return safe(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function escapeAttr(value) {
  return escapeHtml(value);
}

export function compactDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return safe(value);
  return date.toLocaleString();
}

export function statusPill(value) {
  const status = safe(value || "unknown");
  let cls = "";
  if (["active", "success", "healthy", "delivered"].includes(status)) cls = "ok";
  if (["draft", "pending", "retrying"].includes(status)) cls = "warn";
  if (["archived", "error", "blocked", "dead_letter", "failed"].includes(status)) cls = "bad";
  return `<span class="pill ${cls}">${escapeHtml(status)}</span>`;
}

export function jsonBlock(value) {
  return `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
}

export function table(headers, rows) {
  if (!rows.length) return `<div class="empty">No rows</div>`;
  const head = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

export function metric(label, value, detail = "") {
  return `
    <article class="metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${detail ? `<small class="muted">${escapeHtml(detail)}</small>` : ""}
    </article>
  `;
}

export function barList(items, valueKey = "count", label = "count") {
  const max = Math.max(1, ...items.map((item) => Number(item[valueKey] || 0)));
  if (!items.length) return `<div class="empty">No activity yet</div>`;
  return `
    <div class="bar-list">
      ${items.slice(0, 8).map((item) => {
        const value = Number(item[valueKey] || 0);
        const width = Math.max(4, (value / max) * 100);
        return `
          <div class="bar-row">
            <div class="bar-meta">
              <span>${escapeHtml(item.key)}</span>
              <span class="muted">${escapeHtml(number(value))} ${escapeHtml(label)}</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width: ${width}%"></div></div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}
