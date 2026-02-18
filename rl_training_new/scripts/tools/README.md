# Tools

This directory contains utility scripts for environment discovery, parity checks, and debug data processing.

## Scripts
- `list_envs.py`: Lists registered Isaac Lab environments.
- `compare_deploy_train_dumps.py`: Compares deploy debug dumps against training-play dumps by stage (preprocess/history/state/control).
- `validate_deploy_parity.py`: Quick parity sanity checks for deploy policy/config alignment.
- `export_history_seed_from_train_dump.py`: Exports history seed text from a training dump for controlled replay experiments.

## Typical Usage
```bash
python scripts/tools/list_envs.py

python scripts/tools/compare_deploy_train_dumps.py \
  --deploy-dir rl_training_new/lite3_debug/deploy \
  --train-dir rl_training_new/lite3_debug/train/<run_id> \
  --steps 5 \
  --out rl_training_new/lite3_debug/compare_report.json
```
