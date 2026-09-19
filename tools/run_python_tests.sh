#!/usr/bin/env bash
# =============================================================================
# Run every Python module in Docker and report pass/fail.
#   bash tools/run_python_tests.sh            # all modules
#   bash tools/run_python_tests.sh core       # only statsPy/core
#   bash tools/run_python_tests.sh bioinformatics
#   bash tools/run_python_tests.sh foundations
#   bash tools/run_python_tests.sh exercises
# =============================================================================
set -uo pipefail
IMAGE="${IMAGE:-learn-stats-py:1.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILTER="${1:-}"
cd "$ROOT"

mapfile -t FILES < <(find statsPy \( -name '[0-9]*.py' -o -name 'E[0-9]*.py' -o -name 'F[0-9]*.py' \) | grep -v '/notebooks/' | sort)
if [[ -n "$FILTER" ]]; then
  mapfile -t FILES < <(printf '%s\n' "${FILES[@]}" | grep "/$FILTER/")
fi

pass=0; fail=0; failed=()
printf '%-62s %-8s %8s\n' "MODULE" "STATUS" "SECONDS"
printf '%s\n' "--------------------------------------------------------------------------------"
for f in "${FILES[@]}"; do
  start=$(date +%s)
  if docker run --rm -e STATS_OUT=/tmp/stats-out -v "$ROOT":/work -w /work \
       "$IMAGE" python "$f" > "/tmp/$(basename "$f").log" 2>&1; then
    status="PASS"; pass=$((pass+1))
  else
    status="FAIL"; fail=$((fail+1)); failed+=("$f")
  fi
  printf '%-62s %-8s %8s\n' "$f" "$status" "$(( $(date +%s) - start ))"
done
printf '%s\n' "--------------------------------------------------------------------------------"
echo "PASS: $pass   FAIL: $fail"
if (( fail )); then
  echo "Failed modules:"; printf '  %s\n' "${failed[@]}"
  echo "Logs are in /tmp/<module>.log"
  exit 1
fi
