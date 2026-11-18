#!/usr/bin/env bash
# run_all.sh — the whole data pipeline, in order.
#
# RUN      scripts/data/run_all.sh            every step, from the beginning
#          scripts/data/run_all.sh 6          start at step 06
#
# Every step skips work it has already done, so running this again after an
# interruption continues where it stopped. To run one step on its own, call
# its script directly; each script's header says how.
#
# Needs: Julia on the PATH (or set JULIA=/path/to/julia), Docker, and the four
# tool images built (see dockerfiles/README.md).

set -euo pipefail
cd "$(dirname "$0")/../.."

JULIA="${JULIA:-julia}"
FROM="${1:-1}"
THREADS="${THREADS:-16}"

step() {                      # step <number> <script> [arguments...]
    local number="$1"; shift
    if [ "$number" -lt "$FROM" ]; then return; fi
    echo
    echo "=================================================================="
    echo "  step $number: $*"
    echo "  started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="
    "$JULIA" -t "$THREADS" --project=scripts/data "scripts/data/$@"
}

step  1 01_fetch_sources.jl          # download the index files declared in sources.toml
step  2 02_index_entries.jl          # one row per PDB entry
step  3 03_fetch_release_dates.jl    # release date of each entry
step  4 04_index_chains.jl           # one row per chain; date tiers; what to download
step  5 05_fetch_structures.jl --all # the structure files (hours, tens of GB)
step  6 06_prepare_chains.jl         # our own per-residue measurements; chain files for STRIDE
step  7 07_run_dssp.jl               # DSSP on every entry
step  8 08_run_stride.jl             # STRIDE on every chain
step  9 09_cluster_sequences.jl      # sequence clusters at 30/50/70/90% identity
step 10 10_assemble_residues.jl      # one per-residue table from steps 06-08
step 11 11_annotate_chains.jl        # clusters, CATH and what was observed, per chain
step 20 20_build_sets.jl             # train, val and the two test tiers, into data/
step 90 90_report.jl                 # data/REPORT.md

echo
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
