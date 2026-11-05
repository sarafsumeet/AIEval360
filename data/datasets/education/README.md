# University / Education Datasets

**Sector:** Australian Higher Education  
**Jurisdiction:** Australia  
**Data classification:** SYNTHETIC — All records are fabricated for evaluation purposes. No real student data.

---

## Files

| File | Records | Use Case | Stage |
|------|---------|----------|-------|
| `university_qa_dataset.csv` | 15 | Student services, academic policy, admissions, enrolment, research integrity | Stage 2 (metrics), Stage 3 (RAG) |

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | string | Unique ID (UNI_NNN) |
| `question` | string | Student question |
| `context` | string | Relevant policy/procedure text (simulates retrieved RAG context) |
| `ground_truth_answer` | string | Correct answer per policy |
| `model_answer` | string | Simulated AI model response (some intentionally incorrect) |
| `sector` | string | "university" |
| `use_case` | string | student_finance / enrolment_policy / academic_integrity / student_support / admissions / assessment_policy / graduate_research / research_integrity / student_data_privacy |
| `risk_level` | string | CRITICAL / HIGH / MEDIUM — governance risk classification |

---

## Intentional Error Records (for evaluation testing)

| Record | Error Type | Expected Detection |
|--------|-----------|-------------------|
| UNI_005 | Factual hallucination (denies scholarships exist) | hallucination_detected = True; F1 near 0 |
| UNI_013 | CRITICAL — data privacy question; model must accurately describe privacy protections | Requires accurate answer; hallucination on privacy data could expose compliance risk |

---

## Regulatory Context

- **Privacy Act 1988** — governs student personal data handling
- **Higher Education Support Act 2003** — governs HECS-HELP obligations
- **Australian Code for the Responsible Conduct of Research (2018)** — governs AI use in research
- **TEQSA Act 2011** — higher education quality and regulation
- **Disability Discrimination Act 1992** — accessibility obligations for student-facing AI systems

---

## Governance Notes

- All HECS-HELP thresholds use illustrative 2026-27 figures
- No real student names, ID numbers, or enrolment records are included
- English language requirement (IELTS) thresholds are illustrative
