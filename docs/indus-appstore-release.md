# Indus Appstore Release Plan

## Architecture: Android Wrapper App

The Indus Appstore release wraps the VeriFACT web frontend in an Android WebView using Capacitor.

```
┌──────────────────────────────┐
│  Android App (Capacitor)     │
│  ┌────────────────────────┐  │
│  │  WebView               │  │
│  │  → loads verifact web  │  │
│  │  → calls backend API   │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
         │ HTTPS
         ▼
┌──────────────────────────────┐
│  VeriFACT Backend            │
│  (Railway / VPS / self-host) │
└──────────────────────────────┘
```

## Build Steps

### 1. Install Capacitor
```bash
cd web
npm install @capacitor/core @capacitor/cli @capacitor/android
npx cap init "VeriFACT AI" "com.verifact.ai" --web-dir out
```

### 2. Configure API URL
```javascript
// capacitor.config.ts
const config = {
  appId: 'com.verifact.ai',
  appName: 'VeriFACT AI',
  webDir: 'out',
  server: {
    // Dev: local backend
    // url: 'http://10.0.2.2:8765',
    // Prod: deployed backend
    // url: 'https://your-backend.railway.app',
  }
};
```

### 3. Build Variants

| Variant | API URL | Purpose |
|---|---|---|
| `dev` | `http://10.0.2.2:8765` | Local testing (Android emulator → host) |
| `staging` | `https://staging-api.verifact.ai` | Pre-release testing |
| `prod` | `https://api.verifact.ai` | Production release |

### 4. Generate APK
```bash
npm run build        # Build Next.js static export
npx cap add android  # Add Android platform
npx cap sync         # Sync web assets
cd android && ./gradlew assembleRelease
```

## Release Checklist

### Pre-Submission
- [ ] APK builds without errors
- [ ] App opens and displays UI correctly
- [ ] API connection works from app
- [ ] Error states display gracefully (backend down, no network)
- [ ] App icon and splash screen set
- [ ] Version number set in `build.gradle`

### Indus Appstore Specifics
- [ ] App title: "VeriFACT AI"
- [ ] Category: "Productivity" or "Education"
- [ ] Description submitted (use README summary)
- [ ] Screenshots captured (3 minimum: input, results, claims)
- [ ] Privacy policy URL provided
- [ ] Terms of service URL provided
- [ ] APK signed with release keystore
- [ ] Target SDK level meets Indus requirements

### Post-Submission
- [ ] Monitor for review feedback
- [ ] Prepare hotfix process (update web assets → rebuild APK)
- [ ] Set up crash reporting (Firebase Crashlytics or equivalent)

## Privacy Policy Template

VeriFACT AI does not collect, store, or transmit personal data. All text entered for verification is sent only to the configured backend API endpoint. No analytics, tracking, or advertising SDKs are included.

## Rollback Plan

1. Revert to previous APK version in Indus Appstore dashboard
2. If backend issue: redeploy previous Docker image tag
3. If web issue: redeploy previous Vercel deployment
