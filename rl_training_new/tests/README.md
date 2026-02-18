# Tests

This directory holds repository-level tests for behavior that must remain stable across refactors.

## Current Coverage
- `test_history_reset_semantics.py`
  - Verifies default history reset-on-done behavior.
  - Verifies debug-only legacy leak mode behind `LITE3_UNREALISTIC_HISTORY_FEED=1`.

## Run
```bash
pytest -q rl_training_new/tests/test_history_reset_semantics.py
```
