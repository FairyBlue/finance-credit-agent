#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
The experiment queue is disabled fail-closed for G1 validation.
Run the documented direct launcher command with --expected-commit instead.
No preflight or training process was started.
EOF
exit 2
