#!/bin/bash
set -e

# Usage: ./start_training_session.sh <session_name>

SESSION_NAME=$1

if [ -z "$SESSION_NAME" ]; then
  echo "❌ Usage: $0 <session_name>"
  exit 1
fi

echo "======================================"
echo " Starting tmux session: $SESSION_NAME"
echo "======================================"

# Start or attach to tmux session
tmux new-session -d -s "$SESSION_NAME" "cd ~/Lite3/Lite3_rl_training/docker && ./launch.sh && source venv/bin/activate"

# Attach to the session interactively
tmux attach -t "$SESSION_NAME"
