# 01_fetch_sources.jl — download every source declared in sources.toml.
#
# READS    sources.toml
# WRITES   work/raw/<file>         one file per enabled source
#          work/raw/FETCHED.tsv    what was downloaded, when, and its checksum
#
# RUN      julia --project=scripts/data scripts/data/01_fetch_sources.jl
#          julia --project=scripts/data scripts/data/01_fetch_sources.jl cath_names pdb_seqres
#          julia --project=scripts/data scripts/data/01_fetch_sources.jl --force
#
# With no arguments it fetches every source marked `enabled = true`.
# Name one or more sources to fetch only those.
# A file that is already on disk is skipped, unless you pass --force.

include("common.jl")
using TOML

const FETCH_LOG = joinpath(RAW_DIR, "FETCHED.tsv")

function main(arguments)
    force     = "--force" in arguments
    requested = filter(a -> !startswith(a, "--"), arguments)

    sources = TOML.parsefile(SOURCES_FILE)

    # Decide which sources to fetch.
    names = isempty(requested) ?
        sort([name for (name, source) in sources if get(source, "enabled", false)]) :
        requested

    for name in names
        haskey(sources, name) || error("'$name' is not declared in sources.toml")
    end

    log_rows = isfile(FETCH_LOG) ? read_table(FETCH_LOG) : DataFrame(
        name = String[], file = String[], url = String[],
        bytes = Int[], sha256 = String[], fetched_at = String[],
    )

    for name in names
        source      = sources[name]
        destination = joinpath(RAW_DIR, source["file"])

        if isfile(destination) && !force
            say("have   ", name, "  (", source["file"], ")")
            continue
        end

        say("fetch  ", name, "  <- ", source["url"])
        download_file(source["url"], destination)
        checksum = sha256_of(destination)

        # If sources.toml pins a checksum, the download must match it.
        expected = get(source, "sha256", "")
        if !isempty(expected) && expected != checksum
            rm(destination)
            error("checksum mismatch for '$name': expected $expected, got $checksum")
        end

        # Record what we got. One row per source; a re-fetch replaces the old row.
        filter!(row -> row.name != name, log_rows)
        push!(log_rows, (
            name       = name,
            file       = source["file"],
            url        = source["url"],
            bytes      = filesize(destination),
            sha256     = checksum,
            fetched_at = Dates.format(now(), "yyyy-mm-dd HH:MM"),
        ))
    end

    sort!(log_rows, :name)
    write_table(FETCH_LOG, log_rows)
end

main(ARGS)
