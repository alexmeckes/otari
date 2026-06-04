import { $, selectedValue } from "./dom.js";
import {
  barList,
  compactDate,
  escapeAttr,
  escapeHtml,
  metric,
  money,
  number,
  percent,
  statusPill,
  table
} from "./format.js";

function setSelectOptions(id, options, firstLabel) {
  const element = $(id);
  if (!element) return;
  const current = element.value;
  element.innerHTML = `<option value="">${escapeHtml(firstLabel)}</option>${options.map((option) => `
    <option value="${escapeAttr(option.value)}">${escapeHtml(option.label)}</option>
  `).join("")}`;
  if (options.some((option) => option.value === current)) element.value = current;
}

export function renderFilters(state) {
  const projectOptions = state.projects.map((project) => ({
    value: project.project_id,
    label: project.name ? `${project.name} (${project.project_id})` : project.project_id
  }));
  const policyOptions = state.policies.map((policy) => ({
    value: policy.policy_id,
    label: `${policy.name} (${policy.policy_id})`
  }));
  setSelectOptions("trace-project-filter", projectOptions, "All projects");
  setSelectOptions("usage-project-filter", projectOptions, "All projects");
  setSelectOptions("trace-policy-filter", policyOptions, "All policies");
}

export function renderOverview(state) {
  const usage = state.usage || {};
  const traceSummary = state.traceSummary || {};
  const totalTraces = Number(traceSummary.total_count || 0);
  const successRate = totalTraces ? (Number(traceSummary.success_count || 0) / totalTraces) * 100 : 0;
  const activePolicies = state.policies.filter((policy) => policy.status === "active").length;
  const defaultPolicy = state.policies.find((policy) => policy.is_default);
  $("overview-metrics").innerHTML = [
    metric("Policies", state.policies.length, `${activePolicies} active`),
    metric("Projects", state.projects.length),
    metric("Trace success", percent(successRate), `${number(totalTraces)} traced requests`),
    metric("Usage cost", money(usage.cost || 0), `${number(usage.total_tokens || 0)} tokens`),
    metric("Budget alerts", state.alerts.length),
    metric("Budgets", state.budgets.length),
    metric("Errors", number((usage.error_count || 0) + (traceSummary.error_count || 0))),
    metric("Providers", new Set((usage.by_provider || []).map((bucket) => bucket.key)).size)
  ].join("");

  $("default-policy").innerHTML = defaultPolicy ? `
    <div class="stack">
      <div><strong>${escapeHtml(defaultPolicy.name)}</strong></div>
      <div class="mono muted">${escapeHtml(defaultPolicy.policy_id)}</div>
      <div class="row">
        ${statusPill(defaultPolicy.status)}
        <span class="pill">${escapeHtml(defaultPolicy.strategy)}</span>
        <span class="pill">rev ${escapeHtml(defaultPolicy.revision)}</span>
      </div>
      <button type="button" data-action="policy-json" data-id="${escapeAttr(defaultPolicy.policy_id)}">Open policy</button>
    </div>
  ` : `<div class="empty">No default policy is configured</div>`;

  $("routing-mix").innerHTML = barList(traceSummary.by_provider || [], "count", "routes");
}

export function renderPolicies(state) {
  const statusFilter = selectedValue("policy-status-filter");
  const policies = statusFilter
    ? state.policies.filter((policy) => policy.status === statusFilter)
    : state.policies;
  const rows = policies.map((policy) => `
    <tr>
      <td>
        <strong>${escapeHtml(policy.name)}</strong>
        <div class="mono muted">${escapeHtml(policy.policy_id)}</div>
      </td>
      <td>${statusPill(policy.status)}</td>
      <td>${policy.is_default ? '<span class="pill ok">default</span>' : '<span class="muted">no</span>'}</td>
      <td>${escapeHtml(policy.strategy)}</td>
      <td>${escapeHtml(policy.revision)}</td>
      <td>${compactDate(policy.updated_at)}</td>
      <td>
        <div class="row">
          <button type="button" data-action="policy-json" data-id="${escapeAttr(policy.policy_id)}">Details</button>
          <button type="button" data-action="edit-policy" data-id="${escapeAttr(policy.policy_id)}">Edit</button>
          <button type="button" data-action="policy-revisions" data-id="${escapeAttr(policy.policy_id)}">Revisions</button>
          <button type="button" data-action="clone-policy" data-id="${escapeAttr(policy.policy_id)}">Clone</button>
          ${policy.is_default ? "" : `<button type="button" data-action="promote-policy" data-id="${escapeAttr(policy.policy_id)}">Make default</button>`}
          <button class="warn" type="button" data-action="toggle-policy-status" data-id="${escapeAttr(policy.policy_id)}">
            ${policy.status === "archived" ? "Activate" : "Archive"}
          </button>
          <button class="danger" type="button" data-action="delete-policy" data-id="${escapeAttr(policy.policy_id)}">Delete</button>
        </div>
      </td>
    </tr>
  `);
  $("policies-table").innerHTML = table(["Policy", "Status", "Default", "Strategy", "Rev", "Updated", ""], rows);
}

export function renderProjects(state) {
  const rows = state.projects.map((project) => `
    <tr>
      <td>
        <strong>${escapeHtml(project.name || project.project_id)}</strong>
        <div class="mono muted">${escapeHtml(project.project_id)}</div>
      </td>
      <td>${project.is_active ? statusPill("active") : statusPill("archived")}</td>
      <td>${project.blocked ? statusPill("blocked") : '<span class="pill ok">allowed</span>'}</td>
      <td><span class="mono">${escapeHtml(project.routing_policy_id || "default")}</span></td>
      <td>${money(project.spend || 0)}</td>
      <td>
        <div class="row">
          <button type="button" data-action="project-json" data-id="${escapeAttr(project.project_id)}">Details</button>
          <button type="button" data-action="edit-project" data-id="${escapeAttr(project.project_id)}">Edit</button>
        </div>
      </td>
    </tr>
  `);
  $("projects-table").innerHTML = table(["Project", "State", "Traffic", "Policy", "Spend", ""], rows);
}

export function renderTraces(state) {
  const rows = state.traces.map((trace) => `
    <tr>
      <td>
        <span class="mono">${escapeHtml(trace.trace_id)}</span>
        <div class="muted">${compactDate(trace.timestamp)}</div>
      </td>
      <td>${statusPill(trace.status)}</td>
      <td>${escapeHtml(trace.selected_model || "")}</td>
      <td>${escapeHtml(trace.policy_source || "")}</td>
      <td>${money(trace.estimated_cost || 0)}</td>
      <td><button type="button" data-action="trace-json" data-id="${escapeAttr(trace.trace_id)}">Open</button></td>
    </tr>
  `);
  $("traces-table").innerHTML = table(["Trace", "Status", "Selected model", "Source", "Cost", ""], rows);
}

function bucketSection(title, buckets, valueKey = "cost", valueFormatter = money) {
  const rows = (buckets || []).slice(0, 8).map((bucket) => `
    <tr>
      <td>${escapeHtml(bucket.key)}</td>
      <td>${number(bucket.count)}</td>
      <td>${escapeHtml(valueFormatter(bucket[valueKey] || 0))}</td>
    </tr>
  `);
  return `
    <div class="stack">
      <h3>${escapeHtml(title)}</h3>
      ${table(["Key", "Count", valueKey === "cost" ? "Cost" : "Value"], rows)}
    </div>
  `;
}

export function renderUsage(state) {
  const usage = state.usage || {};
  $("usage-metrics").innerHTML = [
    metric("Requests", number(usage.total_count || 0)),
    metric("Tokens", number(usage.total_tokens || 0)),
    metric("Cost", money(usage.cost || 0)),
    metric("Errors", number(usage.error_count || 0))
  ].join("");
  $("usage-breakdown").innerHTML = `
    <div class="stack">
      ${bucketSection("By project", usage.by_project || [])}
      ${bucketSection("By provider", usage.by_provider || [])}
      ${bucketSection("By tag", usage.by_tag || [])}
    </div>
  `;
  const rows = state.usageEntries.map((entry) => `
    <tr>
      <td>
        <span class="mono">${escapeHtml(entry.id)}</span>
        <div class="muted">${compactDate(entry.timestamp)}</div>
      </td>
      <td>${statusPill(entry.status)}</td>
      <td>${escapeHtml(entry.model)}</td>
      <td>${escapeHtml(entry.project_id || "")}</td>
      <td>${number(entry.total_tokens || 0)}</td>
      <td>${money(entry.cost || 0)}</td>
    </tr>
  `);
  $("usage-table").innerHTML = table(["Request", "Status", "Model", "Project", "Tokens", "Cost"], rows);
}

export function renderBudgets(state) {
  const budgetRows = state.budgets.map((budget) => `
    <tr>
      <td><span class="mono">${escapeHtml(budget.budget_id)}</span></td>
      <td>${statusPill(budget.scope_type)}</td>
      <td>${money(budget.spend || 0)} / ${budget.max_budget === null ? "unlimited" : money(budget.max_budget)}</td>
      <td>${budget.is_active ? statusPill("active") : statusPill("archived")}</td>
      <td>${budget.blocked ? statusPill("blocked") : '<span class="pill ok">allowed</span>'}</td>
      <td><button type="button" data-action="budget-json" data-id="${escapeAttr(budget.budget_id)}">Details</button></td>
    </tr>
  `);
  $("budgets-table").innerHTML = table(["Budget", "Scope", "Spend", "State", "Traffic", ""], budgetRows);

  const alertRows = state.alerts.map((alert) => `
    <tr>
      <td><span class="mono">${escapeHtml(alert.id)}</span></td>
      <td>${escapeHtml(alert.scope_type)}:${escapeHtml(alert.scope_id || "")}</td>
      <td>${statusPill(alert.delivery_status)}</td>
      <td>${money(alert.spend || 0)}</td>
      <td>${compactDate(alert.created_at)}</td>
      <td>
        <div class="row">
          <button type="button" data-action="alert-json" data-id="${escapeAttr(alert.id)}">Details</button>
          <button type="button" data-action="retry-alert" data-id="${escapeAttr(alert.id)}">Retry</button>
        </div>
      </td>
    </tr>
  `);
  $("alerts-table").innerHTML = table(["Alert", "Scope", "Delivery", "Spend", "Created", ""], alertRows);
}

export function renderResolver(state) {
  const result = state.resolverResult;
  if (!result) {
    $("resolver-result").innerHTML = `<div class="empty">Run a dry route resolution to inspect policy decisions</div>`;
    return;
  }
  const candidateRows = (result.candidates || []).map((candidate) => `
    <tr>
      <td>${escapeHtml(candidate.model)}</td>
      <td>${escapeHtml(candidate.provider)}</td>
      <td>${money(candidate.estimated_cost || 0)}</td>
      <td>${candidate.routing_score === null ? "" : escapeHtml(Number(candidate.routing_score).toFixed(4))}</td>
      <td>${escapeHtml(candidate.provider_health ? candidate.provider_health.status : "")}</td>
    </tr>
  `);
  const rejectedRows = (result.rejected_candidates || []).map((candidate) => `
    <tr>
      <td>${escapeHtml(candidate.model)}</td>
      <td>${escapeHtml(candidate.provider)}</td>
      <td>${escapeHtml(candidate.reason)}</td>
    </tr>
  `);
  $("resolver-result").innerHTML = `
    <div class="stack">
      <div class="row">
        <span class="pill ok">selected</span>
        <strong>${escapeHtml(result.selected_model)}</strong>
        <span class="muted">${escapeHtml(result.reason || "")}</span>
      </div>
      <div class="row">
        <span class="pill">${escapeHtml(result.strategy)}</span>
        <span class="pill">${escapeHtml(result.policy_name)}</span>
        <span class="pill">${money(result.estimated_cost || 0)}</span>
      </div>
      <h3>Candidates</h3>
      ${table(["Model", "Provider", "Cost", "Score", "Health"], candidateRows)}
      <h3>Rejected candidates</h3>
      ${table(["Model", "Provider", "Reason"], rejectedRows)}
      <button type="button" data-action="resolver-json">Open JSON</button>
    </div>
  `;
}

export function renderAll(state) {
  renderFilters(state);
  renderOverview(state);
  renderPolicies(state);
  renderProjects(state);
  renderTraces(state);
  renderUsage(state);
  renderBudgets(state);
  renderResolver(state);
}
