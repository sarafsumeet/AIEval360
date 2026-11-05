# Insurance Datasets

**Sector:** Life, Health, and General Insurance  
**Jurisdiction:** Australia  
**Data classification:** SYNTHETIC — All records are fabricated for evaluation purposes. No real policyholder data.

---

## Files

| File | Records | Use Case | Stage |
|------|---------|----------|-------|
| `policy_qa_dataset.csv` | 16 | Policy information, underwriting, claims handling, regulatory disclosure | Stage 2 (metrics), Stage 3 (RAG) |

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | string | Unique ID (INS_NNN) |
| `question` | string | Policyholder or prospect question |
| `context` | string | Relevant PDS/policy text (simulates retrieved RAG context) |
| `ground_truth_answer` | string | Correct answer per PDS/regulatory source |
| `model_answer` | string | Simulated AI model response |
| `sector` | string | "insurance" |
| `use_case` | string | policy_information / underwriting / claims_handling / regulatory_disclosure / policy_administration |
| `risk_level` | string | CRITICAL / HIGH / MEDIUM — governance risk classification |

---

## High-Risk Records

| Record | Reason | Expected Behaviour |
|--------|--------|--------------------|
| INS_002 | Pre-existing condition underwriting — incorrect answer could cause policy voidance | Model must not misrepresent exclusion rules |
| INS_015 | Flood vs storm definition — CRITICAL legal/claims distinction | Accurate classification of flood vs storm required |

---

## Regulatory Context

- **Insurance Contracts Act 1984** — governs duty of disclosure, cooling-off, flood/storm definitions
- **Corporations Act 2001** — mandates PDS requirements
- **ASIC RG 271** — Internal Dispute Resolution (IDR) obligations
- **Insurance Contracts Amendment Act 2013** — standardised flood/storm definitions
- **Australian Financial Complaints Authority (AFCA)** — external dispute resolution

---

## Governance Notes

- No real policy numbers, premium schedules, or policyholder names are included
- Phone numbers use placeholder format (1300 XXX XXX)
- All content is for evaluation/demonstration only and must not be used for insurance advice
