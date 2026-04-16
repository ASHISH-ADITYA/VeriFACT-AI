function randomToken() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 18)}`;
}

// Reset API URL on every install/update to ensure latest endpoint
chrome.runtime.onInstalled.addListener(async () => {
  const existing = await chrome.storage.local.get(["verifactApiToken"]);
  const token = existing.verifactApiToken || randomToken();

  chrome.storage.local.set({
    verifactApiUrl: "https://adiashish-verifact-ai.hf.space/analyze/fast",
    verifactEnabled: true,
    verifactApiToken: token
  });
});

// Also reset on browser startup in case storage was corrupted
chrome.runtime.onStartup.addListener(() => {
  chrome.storage.local.set({
    verifactApiUrl: "https://adiashish-verifact-ai.hf.space/analyze/fast",
    verifactEnabled: true
  });
});
