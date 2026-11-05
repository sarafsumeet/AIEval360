# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 2.x | ✅ Active |
| 1.x | ⚠️ Security fixes only |

## Reporting a Vulnerability

AIEval360 is a portfolio demonstration platform. If you identify a security issue in the code, methodology, or evidence artefacts:

**Contact:** saraf.sumeet@gmail.com  
**Subject line:** `[AIEval360 Security] <brief description>`  
**Expected response time:** 5 business days

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested remediation (optional)

## Security Design Principles

This platform was built with the following security controls:

- **No real credentials in source** — all API keys loaded via environment variables; `AIEVAL360_*` prefix
- **No real customer data** — all datasets are synthetic; no PII is stored or processed
- **No external API calls in tests** — test suite uses mock providers only; no network egress required
- **Adversarial probe testing** — the safety evaluation layer tests for prompt injection, jailbreak, PII leakage, and toxicity before any deployment recommendation
- **Audit trail retention** — governance records retained for 7 years per APRA CPS 234 requirements
- **Mandatory gate** — any CRITICAL safety probe failure blocks deployment recommendation until human review

## Responsible Disclosure

This project follows coordinated disclosure. We ask that you:
1. Report privately before any public disclosure
2. Allow reasonable time for a fix before publishing findings
3. Do not access, modify, or delete data beyond what is necessary to demonstrate the vulnerability

We will credit researchers in release notes unless anonymity is requested.
