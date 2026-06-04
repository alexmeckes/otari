import { api, query } from "./api.js";
import { $, selectedValue } from "./dom.js";
import { escapeAttr, escapeHtml, statusPill, table } from "./format.js";
import {
  closeModal,
  setConnection,
  setUpdated,
  showJson,
  showJsonEditor,
  showModal,
  toast
} from "./modal.js";
import {
  renderAll,
  renderBudgets,
  renderFilters,
  renderOverview,
  renderPolicies,
  renderResolver,
  renderTraces,
  renderUsage
} from "./render.js";

const state = {
  key: sessionStorage.getItem("gateway-admin-master-key") || "",
  policies: [],
  projects: [],
  budgets: [],
  alerts: [],
  traces: [],
  usageEntries: [],
  usage: null,
  traceSummary: null,
  resolverResult: null
};

async function loadPolicies() {
  state.policies = await api(state, "/v1/routing-policies?limit=200");
}

async function loadProjects() {
  state.projects = await api(state, "/v1/projects?limit=200");
}

async function loadBudgets() {
  state.budgets = await api(state, "/v1/budgets?limit=200");
}

async function loadAlerts() {
  state.alerts = await api(state, "/v1/budgets/alerts?limit=100");
}

function traceFilterParams(limit) {
  return {
    limit,
    project_id: selectedValue("trace-project-filter"),
    policy_id: selectedValue("trace-policy-filter"),
    status: selectedValue("trace-status-filter")
  };
}

async function loadTraces() {
  state.traces = await api(state, `/v1/route-traces${query(traceFilterParams(50))}`);
  state.traceSummary = await api(state, `/v1/route-traces/summary${query(traceFilterParams(1000))}`);
}

function usageFilterParams(limit) {
  return {
    limit,
    project_id: selectedValue("usage-project-filter")
  };
}

async function loadUsage() {
  state.usage = await api(state, `/v1/usage/summary${query(usageFilterParams(1000))}`);
  state.usageEntries = await api(state, `/v1/usage${query(usageFilterParams(50))}`);
}

async function reloadAndRender() {
  await Promise.all([loadPolicies(), loadProjects(), loadBudgets(), loadAlerts()]);
  renderFilters(state);
  await Promise.all([loadTraces(), loadUsage()]);
  renderAll(state);
  setUpdated();
}

async function refreshAll() {
  state.key = $("master-key").value.trim();
  sessionStorage.setItem("gateway-admin-master-key", state.key);
  setConnection("Loading...");
  try {
    await reloadAndRender();
    setConnection("Connected", true);
  } catch (error) {
    setConnection("Disconnected");
    toast(error.message);
  }
}

async function refreshTracesView() {
  await loadTraces();
  renderTraces(state);
  renderOverview(state);
  setUpdated();
}

async function refreshUsageView() {
  await loadUsage();
  renderUsage(state);
  renderOverview(state);
  setUpdated();
}

function currentPolicy(id) {
  return state.policies.find((policy) => policy.policy_id === id);
}

function currentProject(id) {
  return state.projects.find((project) => project.project_id === id);
}

function currentBudget(id) {
  return state.budgets.find((budget) => budget.budget_id === id);
}

function currentAlert(id) {
  return state.alerts.find((alert) => String(alert.id) === String(id));
}

function createPolicyEditor() {
  showJsonEditor({
    title: "Create routing policy",
    submitLabel: "Create policy",
    value: {
      name: "Production router",
      is_default: false,
      status: "draft",
      default_strategy: {
        type: "fallback",
        providers: [{ provider: "openai", model: "gpt-4o-mini", priority: 1 }]
      },
      change_note: "Created from admin dashboard"
    },
    onSubmit: async (payload) => {
      await api(state, "/v1/routing-policies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      await reloadAndRender();
    }
  });
}

function editPolicy(policy) {
  showJsonEditor({
    title: `Edit ${policy.name}`,
    submitLabel: "Update policy",
    value: {
      name: policy.name,
      status: policy.status,
      is_default: policy.is_default,
      strategy: policy.strategy,
      config: policy.config,
      change_note: "Updated from admin dashboard"
    },
    onSubmit: async (payload) => {
      await api(state, `/v1/routing-policies/${encodeURIComponent(policy.policy_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      await reloadAndRender();
    }
  });
}

async function showRevisions(policyId) {
  const revisions = await api(state, `/v1/routing-policies/${encodeURIComponent(policyId)}/revisions?limit=50`);
  const rows = revisions.map((revision) => `
    <tr>
      <td>${escapeHtml(revision.revision)}</td>
      <td>${escapeHtml(revision.action)}</td>
      <td>${statusPill(revision.status)}</td>
      <td>${escapeHtml(revision.change_note || "")}</td>
      <td>
        <div class="row">
          <button type="button" data-action="revision-json" data-policy-id="${escapeAttr(policyId)}" data-revision="${escapeAttr(revision.revision)}">JSON</button>
          <button class="warn" type="button" data-action="apply-revision" data-policy-id="${escapeAttr(policyId)}" data-revision="${escapeAttr(revision.revision)}">Apply</button>
        </div>
      </td>
    </tr>
  `);
  showModal("Policy revisions", table(["Rev", "Action", "Status", "Note", ""], rows));
}

function createProjectEditor() {
  showJsonEditor({
    title: "Create project",
    submitLabel: "Create project",
    value: {
      project_id: "prod-chat",
      name: "Production chat",
      routing_policy_id: null,
      budget_id: null,
      blocked: false,
      is_active: true,
      metadata: {}
    },
    onSubmit: async (payload) => {
      await api(state, "/v1/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      await reloadAndRender();
    }
  });
}

function editProject(project) {
  showJsonEditor({
    title: `Edit ${project.project_id}`,
    submitLabel: "Update project",
    value: {
      name: project.name,
      routing_policy_id: project.routing_policy_id,
      budget_id: project.budget_id,
      blocked: project.blocked,
      is_active: project.is_active,
      metadata: project.metadata
    },
    onSubmit: async (payload) => {
      await api(state, `/v1/projects/${encodeURIComponent(project.project_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      await reloadAndRender();
    }
  });
}

function createBudgetEditor() {
  showJsonEditor({
    title: "Create budget",
    submitLabel: "Create budget",
    value: {
      max_budget: 1000,
      budget_duration_sec: 2592000,
      scope_type: "tag",
      match_tags: { team: "platform" },
      alert_thresholds: [0.5, 0.8, 0.95],
      alert_webhook_url: null,
      blocked: false,
      is_active: true
    },
    onSubmit: async (payload) => {
      await api(state, "/v1/budgets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      await reloadAndRender();
    }
  });
}

async function runResolver() {
  try {
    const payload = JSON.parse($("resolver-payload").value);
    state.resolverResult = await api(state, "/v1/routing/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    renderResolver(state);
    toast("Route resolved");
  } catch (error) {
    toast(error.message);
  }
}

async function handleAction(event) {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  const id = target.dataset.id;

  try {
    if (action === "create-policy") createPolicyEditor();
    if (action === "policy-json") showJson("Policy JSON", currentPolicy(id));
    if (action === "edit-policy") editPolicy(currentPolicy(id));
    if (action === "policy-revisions") await showRevisions(id);
    if (action === "clone-policy") {
      await api(state, `/v1/routing-policies/${encodeURIComponent(id)}/clone`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ change_note: "Cloned from admin dashboard" })
      });
      await reloadAndRender();
      toast("Policy cloned");
    }
    if (action === "promote-policy") {
      await api(state, `/v1/routing-policies/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "active", is_default: true, change_note: "Promoted from admin dashboard" })
      });
      await reloadAndRender();
      toast("Default policy updated");
    }
    if (action === "toggle-policy-status") {
      const policy = currentPolicy(id);
      const nextStatus = policy.status === "archived" ? "active" : "archived";
      await api(state, `/v1/routing-policies/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: nextStatus, change_note: `${nextStatus} from admin dashboard` })
      });
      await reloadAndRender();
      toast("Policy status updated");
    }
    if (action === "delete-policy") {
      if (!window.confirm("Delete this routing policy?")) return;
      await api(state, `/v1/routing-policies/${encodeURIComponent(id)}${query({ change_note: "Deleted from admin dashboard" })}`, {
        method: "DELETE"
      });
      await reloadAndRender();
      toast("Policy deleted");
    }
    if (action === "revision-json") {
      const revision = await api(state, `/v1/routing-policies/${encodeURIComponent(target.dataset.policyId)}/revisions/${target.dataset.revision}`);
      showJson("Policy revision JSON", revision);
    }
    if (action === "apply-revision") {
      await api(state, `/v1/routing-policies/${encodeURIComponent(target.dataset.policyId)}/revisions/${target.dataset.revision}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ change_note: `Applied revision ${target.dataset.revision} from admin dashboard` })
      });
      closeModal();
      await reloadAndRender();
      toast("Revision applied");
    }
    if (action === "create-project") createProjectEditor();
    if (action === "project-json") showJson("Project JSON", currentProject(id));
    if (action === "edit-project") editProject(currentProject(id));
    if (action === "trace-json") showJson("Route trace", await api(state, `/v1/route-traces/${encodeURIComponent(id)}`));
    if (action === "resolver-json") showJson("Resolved route JSON", state.resolverResult);
    if (action === "create-budget") createBudgetEditor();
    if (action === "budget-json") showJson("Budget JSON", currentBudget(id));
    if (action === "alert-json") showJson("Budget alert JSON", currentAlert(id));
    if (action === "retry-alert") {
      await api(state, `/v1/budgets/alerts/${encodeURIComponent(id)}/deliver`, { method: "POST" });
      await loadAlerts();
      renderBudgets(state);
      renderOverview(state);
      toast("Alert delivery retried");
    }
  } catch (error) {
    toast(error.message);
  }
}

function activateTab(view) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
  document.querySelectorAll(".view").forEach((panel) => panel.classList.toggle("active", panel.id === `view-${view}`));
}

function initResolverPayload() {
  $("resolver-payload").value = JSON.stringify({
    model: "default_routing",
    project_id: null,
    tags: { surface: "chat" },
    messages: [{ role: "user", content: "Compare the cost and latency tradeoffs for this request." }]
  }, null, 2);
}

function init() {
  $("master-key").value = state.key;
  initResolverPayload();
  renderAll(state);
  if (state.key) setConnection("Key loaded");

  $("auth-form").addEventListener("submit", (event) => {
    event.preventDefault();
    refreshAll();
  });
  $("refresh-all").addEventListener("click", refreshAll);
  $("tabs").addEventListener("click", (event) => {
    const tab = event.target.closest("[data-view]");
    if (tab) activateTab(tab.dataset.view);
  });
  $("policy-status-filter").addEventListener("change", async () => {
    await loadPolicies();
    renderPolicies(state);
    renderOverview(state);
  });
  $("reload-traces").addEventListener("click", refreshTracesView);
  ["trace-project-filter", "trace-policy-filter", "trace-status-filter"].forEach((id) => {
    $(id).addEventListener("change", refreshTracesView);
  });
  $("reload-usage").addEventListener("click", refreshUsageView);
  $("usage-project-filter").addEventListener("change", refreshUsageView);
  $("run-resolver").addEventListener("click", runResolver);
  $("modal-close").addEventListener("click", closeModal);
  document.body.addEventListener("click", handleAction);
}

document.addEventListener("DOMContentLoaded", init);
