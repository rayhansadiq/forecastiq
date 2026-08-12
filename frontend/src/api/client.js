// Thin wrapper around fetch for the ForecastIQ API.
//
// Every call goes through requestJson so error handling is written once. The
// backend returns a { detail: "..." } body on failures, and that message is
// what gets surfaced in the UI -- the user should see "Store 9999 does not
// exist", not "Request failed".

async function requestJson(path) {
  let response;

  try {
    response = await fetch(path);
  } catch (networkError) {
    throw new Error(
      "Could not reach the API. Is the backend running on port 8000?"
    );
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const detail = body && body.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : `Request to ${path} failed with status ${response.status}.`
    );
  }

  return body;
}

export function fetchHealth() {
  return requestJson("/api/health");
}

export function fetchModelInfo() {
  return requestJson("/api/model");
}

export function fetchStores() {
  return requestJson("/api/stores");
}

export function fetchHistory(storeId, days) {
  return requestJson(`/api/stores/${storeId}/history?days=${days}`);
}

export function fetchForecast(storeId, { days, promo, schoolHoliday }) {
  const query = new URLSearchParams({
    days: String(days),
    promo: String(promo),
    school_holiday: String(schoolHoliday),
  });
  return requestJson(`/api/stores/${storeId}/forecast?${query}`);
}
