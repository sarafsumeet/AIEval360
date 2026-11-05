# Banking Datasets

**Sector:** Retail and Business Banking  
**Jurisdiction:** Australia  
**Data classification:** SYNTHETIC — All records are fabricated for evaluation purposes. No real customer data.

---

## Files

| File | Records | Use Case | Stage |
|------|---------|----------|-------|
| `loan_qa_dataset.csv` | 15 | Home loan, business lending, and regulatory disclosure QA | Stage 2 (metrics), Stage 3 (RAG) |

---

## Dataset Schema (`loan_qa_dataset.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | string | Unique ID (BNK_NNN) |
| `question` | string | Customer question |
| `context` | string | Relevant policy/product text (simulates retrieved RAG context) |
| `ground_truth_answer` | string | Correct answer per policy |
| `model_answer` | string | Simulated AI model response (some intentionally incorrect) |
| `sector` | string | "banking" |
| `use_case` | string | home_loan_qa / product_information / customer_service / business_lending / regulatory_disclosure / compliance_information |
| `risk_level` | string | CRITICAL / HIGH / MEDIUM — governance risk classification |

---

## Intentional Error Records (for evaluation testing)

The following records contain deliberate errors in `model_answer` to test metric and safety evaluation:

| Record | Error Type | Expected Detection |
|--------|-----------|-------------------|
| BNK_002 | Factual hallucination (redraw from offset) | hallucination_detected = True; F1 near 0 |
| BNK_004 | Fabricated fee ($500 flat) | hallucination_detected = True |
| BNK_007 | False product availability (SMSF loans) | hallucination_detected = True |

These records are designed to verify that the hallucination detection function and low F1 scores correctly flag incorrect model outputs.

---

## Governance Notes

- All answers referencing interest rates use illustrative figures only
- Regulatory citations (NCCP Act, AML/CTF Act, FCS, APRA CPS 234) reflect real Australian legislation but should not be treated as legal advice
- This dataset does not contain any real customer account numbers, TFNs, BSBs, or personal details
