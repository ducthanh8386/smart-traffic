export async function readJsonResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload.detail) || `API request failed (${response.status}).`);
  }
  return payload;
}

export function formatApiError(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || item.message || String(item)).join("; ");
  }
  if (detail && typeof detail === "object") {
    return detail.message || JSON.stringify(detail);
  }
  return String(detail || "");
}

export async function fetchAvailableModels() {
  try {
    const response = await fetch("/api/models");
    const payload = await readJsonResponse(response);
    return Array.isArray(payload.models) ? payload.models : [];
  } catch {
    return [];
  }
}

export async function fetchViolations() {
  try {
    const response = await fetch("/api/violations");
    return await readJsonResponse(response);
  } catch {
    return [];
  }
}

export async function createSession(formData) {
  const response = await fetch("/api/sessions", {
    method: "POST",
    body: formData,
  });
  return await readJsonResponse(response);
}

export async function fetchNextFrame(sessionId) {
  const response = await fetch(`/api/sessions/${sessionId}/next-frame`, { method: "POST" });
  return await readJsonResponse(response);
}

export async function stopSessionApi(sessionId) {
  if (!sessionId) return;
  await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {});
}
