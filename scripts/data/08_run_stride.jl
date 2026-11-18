# 08_run_stride.jl — run STRIDE on every prepared chain.
#
# READS    work/tools/chain_pdb/<ab>/<1abc>_<asym>.pdb.gz  written by step 06
# WRITES   work/tools/stride/<ab>.stride.gz                all chains of a folder
#          work/logs/08_stride_failures.txt                chains STRIDE could not do
#
# RUN      julia --project=scripts/data scripts/data/08_run_stride.jl
#          julia --project=scripts/data scripts/data/08_run_stride.jl cx c7
#
# NEEDS    Docker image tintinet/stride (see dockerfiles/stride)
#
# What STRIDE gives us, per residue: a secondary-structure assignment made by
# a different method from DSSP's, its own backbone angles, and solvent
# accessibility. Each chain is given to STRIDE on its own, so this
# accessibility is that of the ISOLATED chain. DSSP (step 07) sees the whole
# file, so comparing the two tells us which residues are buried by other
# chains.
#
# Output format: one text file per folder. For each chain there is a line
#     CHAIN <1abc>_<asym>
# followed by STRIDE's per-residue lines, which start with ASG:
#     ASG  ALA A   24    3    H    AlphaHelix    -59.62    -34.42      11.2      1C75
#          res chain num ord  ss   ss name         phi       psi       area
# The residue number (third field) is the position in the full sequence,
# because that is how step 06 numbered the chain files.
#
# A folder whose output exists is skipped. Name folders to do only those.

include("common.jl")

# ---- knobs -----------------------------------------------------------------
const WORKERS         = 20       # containers running at once
const TIMEOUT_SECONDS = 120      # give up on one chain after this long
# ----------------------------------------------------------------------------

const CHAIN_PDB_DIR = joinpath(TOOLS_DIR, "chain_pdb")
const STRIDE_DIR    = joinpath(TOOLS_DIR, "stride")

# This runs INSIDE the container, once per worker. Each line of the list is a
# folder name. Paths are relative to work/.
const WORKER_SCRIPT = raw"""
while read folder; do
    out="tools/stride/$folder.stride.gz"
    [ -s "$out" ] && continue
    tmp="/tmp/$folder.stride"
    : > "$tmp"
    for pdb in tools/chain_pdb/$folder/*.pdb.gz; do
        [ -e "$pdb" ] || continue
        name=$(basename "$pdb" .pdb.gz)
        gzip -dc "$pdb" > /tmp/one.pdb            # STRIDE cannot read gzipped files
        if timeout TIMEOUT stride /tmp/one.pdb > /tmp/one.txt 2>/dev/null; then
            echo "CHAIN $name" >> "$tmp"
            grep '^ASG' /tmp/one.txt >> "$tmp"
        else
            echo "$name" >> logs/08_stride_failures.txt
        fi
    done
    gzip -c "$tmp" > "$out.part" && mv "$out.part" "$out"
    rm -f "$tmp"
done < "$LIST"
"""

function main(arguments)
    named   = filter(a -> !startswith(a, "--"), arguments)
    folders = isempty(named) ? sort(readdir(CHAIN_PDB_DIR)) : named
    todo    = [f for f in folders if !isfile(joinpath(STRIDE_DIR, f * ".stride.gz"))]
    say(length(folders), " folders of chains; ", length(todo), " to run")
    isempty(todo) && return

    mkpath(STRIDE_DIR); mkpath(LOGS_DIR)
    script  = replace(WORKER_SCRIPT, "TIMEOUT" => string(TIMEOUT_SECONDS))
    seconds = @elapsed run_in_workers(STRIDE_IMAGE, script, todo, joinpath(STRIDE_DIR, "_lists");
                                      workers = WORKERS)
    done = count(f -> isfile(joinpath(STRIDE_DIR, f * ".stride.gz")), todo)
    say("done in ", round(Int, seconds), " s: ", done, " of ", length(todo), " folders written")
end

main(ARGS)
