# 03_fetch_release_dates.jl — ask the RCSB when each entry was released.
#
# READS    work/index/entries.tsv
# WRITES   work/index/release_dates.tsv     pdb_id, release_date, rcsb_deposit_date
#
# RUN      julia --project=scripts/data scripts/data/03_fetch_release_dates.jl
#
# Why this step exists
#   The flat index files only give the deposition date. The training cutoffs of
#   AlphaFold 3, Boltz and the others are defined on the *release* date, which
#   can be a year or more later. The RCSB serves release dates through a web
#   query, so we ask for them in batches.
#
# It is safe to stop and restart: entries already in release_dates.tsv are not
# asked for again.

include("common.jl")
using JSON
using TOML

# ---- knobs -----------------------------------------------------------------
const BATCH_SIZE         = 1000     # entries per request
const PAUSE_SECONDS      = 0.3      # wait between requests, to be polite
const ONLY_WITH_PROTEIN  = true     # skip nucleic-acid-only entries
# ----------------------------------------------------------------------------

const ENTRIES = joinpath(INDEX_DIR, "entries.tsv")
const OUTPUT  = joinpath(INDEX_DIR, "release_dates.tsv")

"Ask the RCSB for the release and deposition dates of a batch of entries."
function query_release_dates(endpoint, pdb_ids)
    id_list = join(("\"" * uppercase(id) * "\"" for id in pdb_ids), ",")
    query = "{ entries(entry_ids:[$id_list]) { rcsb_id rcsb_accession_info { initial_release_date deposit_date } } }"
    body  = JSON.json(Dict("query" => query))

    response = IOBuffer()
    Downloads.request(endpoint;
        method  = "POST",
        headers = ["Content-Type" => "application/json"],
        input   = IOBuffer(body),
        output  = response,
    )
    parsed = JSON.parse(String(take!(response)))

    rows = DataFrame(pdb_id = String[], release_date = String[], rcsb_deposit_date = String[])
    for entry in parsed["data"]["entries"]
        entry === nothing && continue                 # unknown or obsolete identifier
        info = entry["rcsb_accession_info"]
        # Dates arrive as "2024-07-31T00:00:00Z"; keep the day.
        day(text) = text === nothing ? "" : first(text, 10)
        push!(rows, (lowercase(entry["rcsb_id"]),
                     day(info["initial_release_date"]),
                     day(info["deposit_date"])))
    end
    return rows
end

function main()
    endpoint = TOML.parsefile(SOURCES_FILE)["rcsb_graphql"]["url"]

    entries = read_table(ENTRIES)
    if ONLY_WITH_PROTEIN
        entries = filter(row -> occursin("prot", row.molecule_type), entries)
    end
    wanted = entries.pdb_id

    done = isfile(OUTPUT) ? read_table(OUTPUT) :
           DataFrame(pdb_id = String[], release_date = String[], rcsb_deposit_date = String[])
    already = Set(done.pdb_id)
    todo    = [id for id in wanted if !(id in already)]
    say(length(wanted), " entries wanted, ", length(already), " already have a date, ",
        length(todo), " to ask for")

    batches = collect(Iterators.partition(todo, BATCH_SIZE))
    for (i, batch) in enumerate(batches)
        append!(done, query_release_dates(endpoint, batch))
        # Save after every batch, so an interruption loses at most one batch.
        CSV.write(OUTPUT, done; delim = '\t')
        say("batch ", i, " of ", length(batches), "  (", nrow(done), " dates so far)")
        sleep(PAUSE_SECONDS)
    end

    sort!(done, :pdb_id)
    write_table(OUTPUT, done)

    missing_dates = length(wanted) - count(in(Set(done.pdb_id)), wanted)
    missing_dates > 0 && say("note: ", missing_dates,
        " entries got no answer (obsolete or not yet released). They will have no tier.")
end

main()
