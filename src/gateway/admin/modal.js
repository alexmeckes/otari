import { $ } from "./dom.js";
import { escapeHtml, jsonBlock } from "./format.js";

let toastTimer = null;

export function setConnection(message, connected = false) {
  $("connection-state").textContent = message;
  $("connection-state").className = connected ? "ok-text" : "";
}

export function setUpdated() {
  $("last-updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

export function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.add("visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => element.classList.remove("visible"), 2600);
}

export function showModal(title, body, actions = "") {
  $("modal-title").textContent = title;
  $("modal-body").innerHTML = body;
  $("modal-actions").innerHTML = actions;
  $("modal").showModal();
}

export function closeModal() {
  $("modal").close();
  $("modal-actions").innerHTML = "";
}

export function showJson(title, value) {
  showModal(title, jsonBlock(value));
}

export function showJsonEditor({ title, value, submitLabel, onSubmit }) {
  showModal(
    title,
    `<textarea class="json-editor" id="json-editor" spellcheck="false">${escapeHtml(JSON.stringify(value, null, 2))}</textarea>`,
    `<button type="button" id="modal-submit" class="primary">${escapeHtml(submitLabel)}</button>`
  );
  $("modal-submit").addEventListener("click", async () => {
    try {
      const payload = JSON.parse($("json-editor").value);
      await onSubmit(payload);
      closeModal();
      toast(`${submitLabel} complete`);
    } catch (error) {
      toast(error.message);
    }
  });
}
