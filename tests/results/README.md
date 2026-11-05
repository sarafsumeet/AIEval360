# Test Results

Captured outputs from verified test runs. These files are committed to provide evidence that the test suite executes and passes — not just that test files exist.

| File | Description | Last Run |
|---|---|---|
| `pytest_results.txt` | Full verbose pytest output — 580 tests, 0 failures | 2026-11-05 |
| `pytest_results.xml` | JUnit XML — compatible with GitHub Actions, Jenkins, GitLab CI | 2026-11-05 |
| `coverage_report.txt` | Line coverage per module — 83% overall | 2026-11-05 |
| `coverage.xml` | Machine-readable coverage XML — compatible with Codecov, SonarQube | 2026-11-05 |

## How to reproduce

```bash
# Install dependencies
pip install -e ".[dev]"

# Run full test suite (excluding e2e subprocess tests)
pytest tests/ --ignore=tests/e2e -v --tb=short \
  --junitxml=tests/results/pytest_results.xml \
  2>&1 | tee tests/results/pytest_results.txt

# Run with coverage
pytest tests/ --ignore=tests/e2e \
  --cov=aieval360 \
  --cov-report=term-missing \
  --cov-report=xml:tests/results/coverage.xml \
  2>&1 | tee tests/results/coverage_report.txt
```
