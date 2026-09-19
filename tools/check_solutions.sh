#!/usr/bin/env bash
# Run every teaching module with its PROBLEMS solutions ENABLED.
#
# The normal test runners (run_python_tests.sh / run_r_tests.sh) execute each
# module as a learner first sees it, with the solutions commented out - so they
# never touch the solution code.  This runner uncomments the solutions and runs
# the modules again, which is the only way to know the answers we ship actually
# work.  Everything runs inside Docker; nothing is installed on the host.
#
# Usage:
#   tools/check_solutions.sh            # both languages, all modules
#   tools/check_solutions.sh python     # Python only
#   tools/check_solutions.sh r core     # R core modules only
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANG_FILTER="${1:-all}"
DIR_FILTER="${2:-}"
PY_IMAGE="learn-stats-py:1.0"
R_IMAGE="learn-stats-r:1.0"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/statsPy" "$WORK/statsR"

collect() {                        # $1 = statsPy|statsR, $2 = extension
  local base="$1" ext="$2" sub
  for sub in foundations core bioinformatics exercises; do
    [ -n "$DIR_FILTER" ] && [ "$sub" != "$DIR_FILTER" ] && continue
    [ -d "$ROOT/$base/$sub" ] || continue
    find "$ROOT/$base/$sub" -maxdepth 1 -name "*.$ext" | sort
  done
}

pass=0; fail=0; failed_list=()
printf '%-52s %-8s %8s\n' MODULE STATUS SECONDS
printf '%s\n' "------------------------------------------------------------------------"

run_one() {                        # $1 = source path, $2 = image, $3 = runner
  local src="$1" image="$2" runner="$3"
  local rel="${src#$ROOT/}"
  local dst="$WORK/$rel"
  mkdir -p "$(dirname "$dst")"
  python3 "$ROOT/tools/uncomment_solutions.py" "$src" "$dst" >/dev/null
  local start; start=$(date +%s)
  # Mount the repo read-only at /work for any data the module reads, and the
  # rewritten module at /solution so we never execute from the repo itself.
  if docker run --rm \
        -e STATS_OUT=/tmp/out \
        -v "$ROOT":/work:ro \
        -v "$WORK":/solution:ro \
        -w /work "$image" \
        bash -c "$runner /solution/$rel" >"$WORK/log" 2>&1; then
    local secs=$(( $(date +%s) - start ))
    printf '%-52s %-8s %8d\n' "$rel" PASS "$secs"
    pass=$((pass + 1))
  else
    local secs=$(( $(date +%s) - start ))
    printf '%-52s %-8s %8d\n' "$rel" FAIL "$secs"
    echo "--- last 25 lines ---"
    tail -25 "$WORK/log" | sed 's/^/    /'
    fail=$((fail + 1))
    failed_list+=("$rel")
  fi
}

if [ "$LANG_FILTER" = all ] || [ "$LANG_FILTER" = python ]; then
  while read -r f; do [ -n "$f" ] && run_one "$f" "$PY_IMAGE" "python"; done < <(collect statsPy py)
fi
if [ "$LANG_FILTER" = all ] || [ "$LANG_FILTER" = r ]; then
  while read -r f; do [ -n "$f" ] && run_one "$f" "$R_IMAGE" "Rscript"; done < <(collect statsR R)
fi

printf '%s\n' "------------------------------------------------------------------------"
printf 'solutions: %d passed, %d failed\n' "$pass" "$fail"
if [ "$fail" -gt 0 ]; then
  printf 'failed:\n'; printf '  %s\n' "${failed_list[@]}"
  exit 1
fi
