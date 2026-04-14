# Indus Appstore Release Checklist

## Product Packaging
- [ ] Build Android shell app using Capacitor or Trusted Web Activity.
- [ ] Configure app name, icons, splash, and package ID.
- [ ] Point app to production web URL.

## API/Backend Readiness
- [ ] Production API deployed with HTTPS.
- [ ] Auth token enabled for private endpoints.
- [ ] Uptime monitoring and error logging enabled.
- [ ] Abuse protection (rate limit + payload size limits).

## Compliance and Trust
- [ ] Privacy policy URL published.
- [ ] Terms of service URL published.
- [ ] Contact/support email configured.
- [ ] Data handling statement added in app listing.

## QA Gates
- [ ] All tests pass (`python -m pytest -q`).
- [ ] Smoke test passes (`python smoke_test.py`).
- [ ] Manual E2E test from mobile shell app completed.
- [ ] Performance check completed on slow network.

## Release Ops
- [ ] Versioning and changelog updated.
- [ ] Crash analytics integrated.
- [ ] Rollback plan documented.
- [ ] Post-release monitoring checklist prepared.
