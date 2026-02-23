# Tests

This directory holds repository-level tests for behavior that must remain stable across refactors.

## Current Coverage
- `test_history_reset_semantics.py`
  - Verifies default history reset-on-done behavior.
  - Verifies debug-only legacy leak mode behind `LITE3_UNREALISTIC_HISTORY_FEED=1`.
- `test_safe_slow_low_power_task_scaffold.py`
  - Verifies the isolated safe/slow/low-power task scaffold is registered.
  - Verifies the new scaffold disables positive-only reward clipping.
- `test_safe_slow_low_power_curriculum_plan.py`
  - Verifies the exploration-first to strict-late phase factory is present.
  - Verifies the new task env config points to the isolated phase factory.

## Run
```bash
pytest -q rl_training_new/tests/test_history_reset_semantics.py
pytest -q rl_training_new/tests/test_safe_slow_low_power_task_scaffold.py
pytest -q rl_training_new/tests/test_safe_slow_low_power_curriculum_plan.py
```
