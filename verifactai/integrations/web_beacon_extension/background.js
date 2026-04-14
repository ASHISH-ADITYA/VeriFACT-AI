function randomToken() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 18)}`;
}

chrome.runtime.onInstalled.addListener(async () => {
  const existing = await chrome.storage.local.get(["verifactApiToken"]);
  const token = existing.verifactApiToken || randomToken();

  chrome.storage.local.set({
    verifactApiUrl: "http://127.0.0.1:8765/analyze",
    verifactEnabled: true,
    verifactApiToken: token
  });
});
