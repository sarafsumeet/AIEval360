# Australian AI Governance Principles Reference

**Scope:** AIEval360 regulatory alignment reference  
**Jurisdiction:** Australia  
**Classification:** Reference document — SYNTHETIC evaluation use only

---

## 1. APRA Prudential Standard CPS 234 (Information Security)

**Applies to:** Banks, insurers, superannuation funds regulated by APRA  
**Relevance to AI:** AI systems that process or access customer information must meet CPS 234's information security requirements.

| Requirement | AI Governance Implication |
|-------------|--------------------------|
| Security capability (§15) | AI systems must be within scope of information security capability |
| Security controls (§27) | AI model access controls and API security |
| Incident management (§36) | AI failures and adversarial attacks are notifiable incidents |
| Audit and testing (§46) | AI systems subject to audit and penetration testing |
| Record retention | 7-year audit trail requirement → `retention_days: 2555` |

**AIEval360 alignment:** `config/settings.yaml` audit_trail_retention_days=2555; evidence chain in `evidence/governance_records/`.

---

## 2. Anti-Money Laundering and Counter-Terrorism Financing Act 2006 (AML/CTF)

**Applies to:** Financial institutions, remittance dealers, digital currency exchanges  
**Relevance to AI:** AI systems used for customer onboarding, transaction monitoring, or customer identification must comply with AML/CTF obligations.

| Obligation | AI Governance Implication |
|------------|--------------------------|
| Customer identification | AI identity verification must meet Part 2 requirements |
| Ongoing due diligence | AI transaction monitoring must flag suspicious activity |
| Record-keeping (§106) | 7-year record-keeping → `retention_days: 2555` |
| Non-disclosure rules | AI outputs must not "tip off" subjects of AML investigations |

**AIEval360 alignment:** BNK_013 dataset record tests AML identification narrative.

---

## 3. National Consumer Credit Protection Act 2009 (NCCP)

**Applies to:** Banks, lenders, credit assistance providers  
**Relevance to AI:** AI credit decisioning tools must comply with responsible lending obligations.

| Obligation | AI Governance Implication |
|------------|--------------------------|
| Suitability assessment | AI credit decisions must be explainable and auditable |
| Hardship provisions | AI must not suppress or misrepresent hardship information |
| Unfair contract terms | AI-generated contract summaries must be accurate |

**AIEval360 alignment:** BNK_014 tests responsible lending narrative; hallucination on credit obligations is CRITICAL risk.

---

## 4. Insurance Contracts Act 1984 (as amended 2013)

**Applies to:** General and life insurers  
**Relevance to AI:** AI policy interpretation and claims handling tools must correctly apply statutory definitions.

| Provision | AI Governance Implication |
|-----------|--------------------------|
| Duty of disclosure (s21) | AI must accurately explain non-disclosure consequences |
| Standard cover (s34-37) | AI must not misstate standard cover inclusions |
| Flood/storm definition (s1A) | AI must correctly distinguish flood from storm |
| Utmost good faith (s13) | AI responses must not mislead policyholders |

**AIEval360 alignment:** INS_015 (flood vs storm) is CRITICAL risk; hallucination_001 in safety evaluation is CRITICAL failure.

---

## 5. Privacy Act 1988 (and Australian Privacy Principles)

**Applies to:** All organisations with >$3M turnover; health service providers regardless of turnover  
**Relevance to AI:** AI systems that collect, use, or disclose personal information must comply with the 13 Australian Privacy Principles (APPs).

| Principle | AI Governance Implication |
|-----------|--------------------------|
| APP 3 — Collection | AI must only collect information necessary for its purpose |
| APP 5 — Notification | AI must inform individuals of data collection purposes |
| APP 6 — Use/disclosure | AI training and inference on personal data must be within scope |
| APP 11 — Security | AI model weights and training data must be secured |
| APP 12 — Access | Individuals may request access to AI decisions about them |

**AIEval360 alignment:** PII leakage probes (pii_001 to pii_008) test APP 11 compliance; UNI_013 tests student data privacy.

---

## 6. ASIC Regulatory Guide 271 — Internal Dispute Resolution (IDR)

**Applies to:** Financial services licensees and credit licensees  
**Relevance to AI:** AI customer service systems must direct customers to IDR processes and not misrepresent complaint handling timelines.

| Requirement | AI Governance Implication |
|-------------|--------------------------|
| Acknowledgement within 24h | AI chatbots handling complaints must escalate within 24h |
| Resolution within 30 days | AI must accurately state IDR timelines |
| Written reasons | AI-assisted responses must be traceable |

**AIEval360 alignment:** INS_014 dataset record tests ASIC RG 271 narrative accuracy.

---

## Disclaimer

This reference document is a summary for evaluation and governance design purposes only. It does not constitute legal advice. Organisations should seek independent legal advice on their regulatory obligations. Regulatory requirements may have changed since the date of this document.
