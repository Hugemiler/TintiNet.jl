# 05_fetch_structures.jl — download the structure file of every selected entry.
#
# READS    work/index/chains.tsv           (the `selected` column)
#          sources.toml                    ([pdb_structures] base URL)
# WRITES   work/raw/mmCIF/<ab>/<1abc>.cif.gz
#          work/logs/05_fetch_structures_failures.tsv
#
# RUN      julia --project=scripts/data scripts/data/05_fetch_structures.jl --limit 20
#          julia --project=scripts/data scripts/data/05_fetch_structures.jl 1crn 4hhb
#          julia --project=scripts/data scripts/data/05_fetch_structures.jl --tier test_post2023
#          julia --project=scripts/data scripts/data/05_fetch_structures.jl --all
#
# You must say how much you want: a --limit, a list of identifiers, a --tier,
# or --all. Run with no arguments and it only reports how many entries are
# selected and how many are already on disk. The full set is tens of thousands
# of files and tens of gigabytes, so --all is never the default.
#
# Files already on disk are skipped, so it is safe to stop and restart.
#
# For the complete archive, wwPDB recommends rsync over many small downloads:
#   rsync -rlpt -z --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ work/raw/mmCIF/
# That fetches every entry, selected or not.

include("common.jl")
using TOML

# ---- knobs -----------------------------------------------------------------
const SIMULTANEOUS_DOWNLOADS = 8
# ----------------------------------------------------------------------------

const CHAINS      = joinpath(INDEX_DIR, "chains.tsv")
const MMCIF_DIR   = joinpath(RAW_DIR, "mmCIF")
const FAILURE_LOG = joinpath(LOGS_DIR, "05_fetch_structures_failures.tsv")

"Entry 1abc is stored in the folder named after its two middle characters: ab/1abc.cif.gz"
relative_path(pdb_id) = joinpath(pdb_id[2:3], pdb_id * ".cif.gz")

"The value following a flag, e.g. flag_value(args, \"--limit\") for `--limit 20`."
function flag_value(arguments, flag)
    i = findfirst(==(flag), arguments)
    return i === nothing || i == length(arguments) ? nothing : arguments[i + 1]
end

function main(arguments)
    base_url = TOML.parsefile(SOURCES_FILE)["pdb_structures"]["url"]

    chains   = read_table(CHAINS)
    selected = filter(:selected => identity, chains)

    # ---- which entries? ----------------------------------------------------
    limit    = flag_value(arguments, "--limit")
    tier     = flag_value(arguments, "--tier")
    flag_values = filter(!isnothing, [limit, tier])
    explicit = [lowercase(a) for a in arguments if !startswith(a, "--") && !(a in flag_values)]

    if tier !== nothing
        selected = filter(:tier => ==(tier), selected)
    end
    wanted = sort(unique(selected.pdb_id))

    on_disk(id) = isfile(joinpath(MMCIF_DIR, relative_path(id)))
    say(length(wanted), " selected entries", tier === nothing ? "" : " in tier $tier",
        "; ", count(on_disk, wanted), " already on disk")

    if !isempty(explicit)
        wanted = explicit                       # exactly what was asked for
    elseif limit !== nothing
        wanted = first(filter(!on_disk, wanted), parse(Int, limit))
    elseif !("--all" in arguments) && tier === nothing
        say("nothing downloaded. Pass --limit N, identifiers, --tier NAME, or --all.")
        return
    end

    todo = filter(!on_disk, wanted)
    say("downloading ", length(todo), " entries, ", SIMULTANEOUS_DOWNLOADS, " at a time")

    # ---- download ----------------------------------------------------------
    failures = DataFrame(pdb_id = String[], error = String[])
    finished = Threads.Atomic{Int}(0)

    asyncmap(todo; ntasks = SIMULTANEOUS_DOWNLOADS) do pdb_id
        url         = base_url * "/" * pdb_id[2:3] * "/" * pdb_id * ".cif.gz"
        destination = joinpath(MMCIF_DIR, relative_path(pdb_id))
        try
            download_file(url, destination)
        catch err
            push!(failures, (pdb_id, first(sprint(showerror, err), 200)))
        end
        n = Threads.atomic_add!(finished, 1) + 1
        n % 500 == 0 && say(n, " of ", length(todo))
    end

    say("done: ", length(todo) - nrow(failures), " downloaded, ", nrow(failures), " failed")
    if nrow(failures) > 0
        write_table(FAILURE_LOG, sort!(failures, :pdb_id))
    end
end

main(ARGS)
