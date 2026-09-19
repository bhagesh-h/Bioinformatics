#!/usr/bin/env bash
# Generate notebooks from the plain-text modules.
#
# Every module in this repository is authored ONCE, as a plain script:
#
#   statsPy/**/*.py   jupytext "percent" format   (`# %%` cells, `# %% [markdown]`)
#   statsR/**/*.R     knitr::spin format          (`#'` markdown, `#+` chunk options)
#
# The script is the source of truth: it is what the test runners execute and
# what you edit. This tool renders the notebook views of those same files, so
# the two can never drift apart. Notebooks are regenerated, never hand-edited.
#
# Everything runs inside Docker; nothing is installed on the host.
#
# Usage:
#   tools/build_notebooks.sh          # both languages
#   tools/build_notebooks.sh python   # Python -> .ipynb only
#   tools/build_notebooks.sh r        # R -> .Rmd only
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANG_FILTER="${1:-all}"
PY_IMAGE="learn-stats-py:1.0"
R_IMAGE="learn-stats-r:1.0"
built=0; failed=0

if [ "$LANG_FILTER" = all ] || [ "$LANG_FILTER" = python ]; then
  echo "Python: .py (jupytext percent) -> .ipynb"
  for sub in foundations core bioinformatics exercises; do
    [ -d "$ROOT/statsPy/$sub" ] || continue
    mkdir -p "$ROOT/statsPy/notebooks/$sub"
    for f in "$ROOT/statsPy/$sub"/*.py; do
      [ -e "$f" ] || continue
      rel="${f#$ROOT/}"
      base="$(basename "${f%.py}")"
      out="statsPy/notebooks/$sub/$base.ipynb"
      # --to ipynb converts without executing: the learner runs the cells.
      if docker run --rm -v "$ROOT":/work -w /work "$PY_IMAGE" \
           jupytext --to ipynb --output "$out" "$rel" >/dev/null 2>&1; then
        printf '  %-58s -> %s\n' "$rel" "$out"; built=$((built + 1))
      else
        printf '  %-58s    FAILED\n' "$rel"; failed=$((failed + 1))
      fi
    done
  done
fi

if [ "$LANG_FILTER" = all ] || [ "$LANG_FILTER" = r ]; then
  echo "R: .R (knitr::spin) -> .Rmd"
  for sub in foundations core bioinformatics exercises; do
    [ -d "$ROOT/statsR/$sub" ] || continue
    mkdir -p "$ROOT/statsR/notebooks/$sub"
    for f in "$ROOT/statsR/$sub"/*.R; do
      [ -e "$f" ] || continue
      rel="${f#$ROOT/}"
      base="$(basename "${f%.R}")"
      out="statsR/notebooks/$sub/$base.Rmd"
      # knit = FALSE emits the .Rmd without running it, which is both fast and
      # what we want: the learner executes the chunks in RStudio or via
      # rmarkdown::render().
      if docker run --rm -v "$ROOT":/work -w /work "$R_IMAGE" Rscript -e \
           "knitr::spin('$rel', knit = FALSE, format = 'Rmd'); \
            file.rename(sub('[.]R$', '.Rmd', '$rel'), '$out')" >/dev/null 2>&1; then
        printf '  %-58s -> %s\n' "$rel" "$out"; built=$((built + 1))
      else
        printf '  %-58s    FAILED\n' "$rel"; failed=$((failed + 1))
      fi
    done
  done
fi

echo "------------------------------------------------------------------------"
printf 'notebooks: %d built, %d failed\n' "$built" "$failed"
[ "$failed" -gt 0 ] && exit 1
exit 0
