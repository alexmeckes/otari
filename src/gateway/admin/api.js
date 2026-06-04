const API_KEY_HEADER = "Otari-Key";

function authHeaders(state, extra = {}) {
  return { ...extra, [API_KEY_HEADER]: `Bearer ${state.key}` };
}

export function query(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") search.set(key, value);
  });
  const text = search.toString();
  return text ? `?${text}` : "";
}

export async function api(state, path, options = {}) {
  if (!state.key) throw new Error("Master key is required");
  const response = await fetch(path, {
    ...options,
    headers: authHeaders(state, options.headers || {})
  });
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    throw new Error(`${response.status} ${detail}`);
  }
  return payload;
}
