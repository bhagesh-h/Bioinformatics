#!/usr/bin/env python3
"""Turn the commented-out SOLUTION blocks of a teaching module into live code.

Every module in statsPy/ and statsR/ ends with a PROBLEMS section in which each
problem ships a full worked solution, commented out so the learner can attempt
the problem first.  Those solutions are code, and code that is never executed
rots.  This script rewrites a module with the solutions ENABLED so that
tools/check_solutions.sh can actually run them.

Block format (identical in both languages apart from the comment marker):

    ## ---- SOLUTION (uncomment to check) ----
    # <line of solution code>
    #
    # ## explanatory prose, stays a comment after uncommenting

A block starts at the SOLUTION marker and ends at the first line that is blank
or begins a new literate cell (`# %%` for jupytext, `#'` for knitr::spin).
Usage:  uncomment_solutions.py <in> <out>
"""
import re
import sys

# A cell boundary ends a solution block.  `# %%` is jupytext's cell marker and
# `#'` is knitr::spin's markdown marker; neither can appear inside a block.
END = re.compile(r"^\s*(#\s*%%|#')")
START = re.compile(r"^\s*#+\s*-*\s*SOLUTION\b")


def uncomment(lines):
    out, in_block, n = [], False, 0
    for line in lines:
        if START.search(line):
            in_block = True
            out.append(line)
            continue
        if in_block:
            if not line.strip() or END.match(line):
                in_block = False
                out.append(line)
                continue
            m = re.match(r"^(\s*)#(?: ?)(.*)$", line)
            if m:
                # An empty `#` becomes an empty line; `# code` becomes `code`.
                out.append(f"{m.group(1)}{m.group(2)}".rstrip() + "\n")
                n += 1
                continue
            # Not a comment inside a block: leave it alone rather than guess.
            out.append(line)
            continue
        out.append(line)
    return out, n


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as fh:
        lines = fh.readlines()
    out, n = uncomment(lines)
    with open(dst, "w") as fh:
        fh.writelines(out)
    print(f"{src}: uncommented {n} solution lines")


if __name__ == "__main__":
    main()
