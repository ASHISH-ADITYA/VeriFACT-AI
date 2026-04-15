function randomToken() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 18)}`;
}

chrome.runtime.onInstalled.addListener(async () => {
  const existing = await chrome.storage.local.get(["verifactApiToken"]);
  const token = existing.verifactApiToken || randomToken();

  chrome.storage.local.set({
    verifactApiUrl: "https://adiashish-verifact-ai.hf.space/analyze",
    verifactEnabled: true,
    verifactApiToken: token
  });
});
