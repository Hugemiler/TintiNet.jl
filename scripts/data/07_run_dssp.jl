# 07_run_dssp.jl — run DSSP on every downloaded structure.
#
# READS    work/index/chains.tsv                  which entries are selected
#          work/raw/mmCIF/<ab>/<1abc>.cif.gz
# WRITES   work/tools/dssp/<ab>/<1abc>.dssp.cif.gz    DSSP's per-residue tables
#          work/logs/07_dssp_failures.txt             entries DSSP could not do
#
# RUN      julia --project=scripts/data scripts/data/07_run_dssp.jl
#          julia --project=scripts/data scripts/data/07_run_dssp.jl --limit 200
#
# NEEDS    Docker image tintinet/dssp (see dockerfiles/dssp)
#
# What DSSP gives us, per residue: secondary structure (8 states), solvent
# accessibility, backbone angles, and which strand ladders the residue is in.
# DSSP is run on the whole file, so accessibility and strand partners take
# every chain in the file into account.
#
# DSSP writes back a copy of the entire structure with its own tables added.
# We keep only its two tables (_dssp_struct_summary and _dssp_struct_ladder),
# which makes the output about a hundred times smaller.
#
# Entries that already have an output file are skipped, so this is safe to
# stop and restart, and safe to run while structures are still downloading
# (run it again afterwards to pick up the rest).

include("common.jl")

# ---- knobs -----------------------------------------------------------------
const WORKERS         = 20       # containers running at once
const TIMEOUT_SECONDS = 900      # give up on one entry after this long
# ----------------------------------------------------------------------------

const CHAINS      = joinpath(INDEX_DIR, "chains.tsv")
const MMCIF_DIR   = joinpath(RAW_DIR, "mmCIF")
const DSSP_DIR    = joinpath(TOOLS_DIR, "dssp")
const FAILURE_LOG = joinpath(LOGS_DIR, "07_dssp_failures.txt")

# This runs INSIDE the container, once per worker. Paths are relative to work/.
const WORKER_SCRIPT = raw"""
while read id; do
    sub=$(printf %s "$id" | cut -c2-3)
    out="tools/dssp/$sub/$id.dssp.cif.gz"
    [ -s "$out" ] && continue
    mkdir -p "tools/dssp/$sub"
    tmp="/tmp/$id.cif"

    if timeout TIMEOUT mkdssp --output-format mmcif --calculate-accessibility \
           "raw/mmCIF/$sub/$id.cif.gz" "$tmp" >/dev/null 2>&1 && [ -s "$tmp" ]; then
        # Keep only DSSP's own two tables.
        awk '
            /^_dssp_struct_summary\./ || /^_dssp_struct_ladder\./ {
                if (!keep && previous == "loop_") print "loop_"
                keep = 1; print; previous = $0; next
            }
            keep && /^#/ { print "#"; keep = 0 }
            keep         { print }
            { previous = $0 }
        ' "$tmp" | gzip > "$out.part" && mv "$out.part" "$out"
    else
        echo "$id" >> logs/07_dssp_failures.txt
    fi
    rm -f "$tmp"
done < "$LIST"
"""

output_of(pdb_id) = joinpath(DSSP_DIR, pdb_id[2:3], pdb_id * ".dssp.cif.gz")
input_of(pdb_id)  = joinpath(MMCIF_DIR, pdb_id[2:3], pdb_id * ".cif.gz")

function main(arguments)
    limit_at = findfirst(==("--limit"), arguments)
    limit    = limit_at === nothing ? nothing : parse(Int, arguments[limit_at + 1])

    chains   = read_table(CHAINS)
    selected = sort(unique(filter(:selected => identity, chains).pdb_id))

    downloaded = filter(id -> isfile(input_of(id)), selected)
    already_failed = isfile(FAILURE_LOG) ? Set(readlines(FAILURE_LOG)) : Set{String}()
    todo = filter(id -> !isfile(output_of(id)) && !(id in already_failed), downloaded)
    say(length(selected), " selected entries; ", length(downloaded), " downloaded; ",
        length(downloaded) - length(todo), " already done or failed; ", length(todo), " to run")

    limit === nothing || (todo = first(todo, limit))
    isempty(todo) && return

    mkpath(DSSP_DIR); mkpath(LOGS_DIR)
    say("running DSSP on ", length(todo), " entries with ", WORKERS, " workers ...")
    script = replace(WORKER_SCRIPT, "TIMEOUT" => string(TIMEOUT_SECONDS))
    seconds = @elapsed run_in_workers(DSSP_IMAGE, script, todo, joinpath(DSSP_DIR, "_lists");
                                      workers = WORKERS)

    done = count(id -> isfile(output_of(id)), todo)
    say("done in ", round(Int, seconds), " s: ", done, " succeeded, ", length(todo) - done,
        " failed  (", round(done / max(seconds, 1); digits = 1), " entries per second)")
end

main(ARGS)
