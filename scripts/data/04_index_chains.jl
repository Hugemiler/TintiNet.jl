# 04_index_chains.jl — one row per protein chain, with its date tier and
# whether we want its structure.
#
# READS    work/raw/pdb/pdb_seqres.txt.gz
#          work/index/entries.tsv
#          work/index/release_dates.tsv
# WRITES   work/index/chains.tsv         one row per protein chain (no sequence)
#          work/index/sequences.fasta    each distinct sequence once
#
# RUN      julia --project=scripts/data scripts/data/04_index_chains.jl
#
# Columns of chains.tsv
#   pdb_id, chain_id     which chain. chain_id is as written in pdb_seqres.
#   length               number of residues in the full sequence
#   unknown_fraction     share of residues written as X
#   sequence_id          chains with exactly the same sequence share this
#                        number; it is also the name used in sequences.fasta
#   method, resolution   copied from entries.tsv
#   deposition_date      copied from entries.tsv
#   release_date         from release_dates.tsv; empty if the RCSB gave none
#   tier                 train_pool | test_2021_2023 | test_post2023 | (empty)
#   selected             true if we want this chain's structure downloaded
#   why_not              if not selected, the first rule it failed; else empty
#
# This step decides what gets DOWNLOADED, so its rules are deliberately looser
# than the rules for what gets TRAINED ON. Tightening a rule later (say, a
# stricter resolution cut) is done in the build scripts and needs no new
# download; loosening one beyond the values here does.

include("common.jl")
using CodecZlib

# ---- knobs -----------------------------------------------------------------

# Date tiers, by RELEASE date.
#   released before TRAIN_BEFORE            -> train_pool
#   TRAIN_BEFORE up to TEST_CLEAN_FROM      -> test_2021_2023
#   TEST_CLEAN_FROM onwards                 -> test_post2023
# TRAIN_BEFORE is the AlphaFold 3 / Boltz-1 training cutoff.
# TEST_CLEAN_FROM is the Boltz-2 cutoff, the latest among the models we compare.
const TRAIN_BEFORE    = Date(2021, 9, 30)
const TEST_CLEAN_FROM = Date(2023, 6, 1)

# Which chains do we want structures for?
const WANTED_METHODS       = ["diffraction"]   # as named in pdb_entry_type.txt
const MAX_RESOLUTION       = 3.0               # ångström
const MIN_LENGTH           = 30                # residues
const MAX_UNKNOWN_FRACTION = 0.10              # share of X in the sequence

# ----------------------------------------------------------------------------

const SEQRES        = joinpath(RAW_DIR, "pdb", "pdb_seqres.txt.gz")
const ENTRIES       = joinpath(INDEX_DIR, "entries.tsv")
const RELEASE_DATES = joinpath(INDEX_DIR, "release_dates.tsv")
const CHAINS_OUT    = joinpath(INDEX_DIR, "chains.tsv")
const FASTA_OUT     = joinpath(INDEX_DIR, "sequences.fasta")

"""
Read every protein chain from pdb_seqres. Records look like

    >101m_A mol:protein length:154  MYOGLOBIN
    MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG

Returns the chain table and the list of distinct sequences.
"""
function read_protein_chains(path)
    chains = DataFrame(pdb_id = String[], chain_id = String[], length = Int[],
                       unknown_fraction = Float64[], sequence_id = Int[])
    sequence_ids = Dict{String, Int}()      # sequence -> its number
    sequences    = String[]                 # number -> sequence

    open(GzipDecompressorStream, path) do io
        while !eof(io)
            header = readline(io)
            startswith(header, '>') || continue
            sequence = readline(io)

            m = match(r"^>(\w+)_(\S+) mol:(\S+) length:(\d+)", header)
            m === nothing && continue
            pdb_id, chain_id, molecule, _ = m.captures
            molecule == "protein" || continue

            id = get!(sequence_ids, sequence) do
                push!(sequences, sequence)
                length(sequences)
            end

            n = length(sequence)
            push!(chains, (lowercase(pdb_id), chain_id, n,
                           round(count(==('X'), sequence) / max(n, 1); digits = 3), id))
        end
    end
    return chains, sequences
end

"Which tier a release date falls in. `missing` when the date is unknown."
function tier_of(release_date)
    ismissing(release_date)          && return missing
    release_date <  TRAIN_BEFORE     && return "train_pool"
    release_date <  TEST_CLEAN_FROM  && return "test_2021_2023"
    return "test_post2023"
end

"The first selection rule a chain fails, or `missing` if it passes them all."
function why_not_selected(row)
    ismissing(row.method)                       && return "method"
    !(row.method in WANTED_METHODS)             && return "method"
    ismissing(row.resolution)                   && return "no resolution"
    row.resolution > MAX_RESOLUTION             && return "resolution"
    row.length < MIN_LENGTH                     && return "too short"
    row.unknown_fraction > MAX_UNKNOWN_FRACTION && return "too many unknown residues"
    ismissing(row.tier)                         && return "no release date"
    return missing
end

function main()
    say("reading ", relpath(SEQRES, REPO_ROOT), " ...")
    chains, sequences = read_protein_chains(SEQRES)
    say(nrow(chains), " protein chains, ", length(sequences), " distinct sequences")

    entries = read_table(ENTRIES)
    dates   = read_table(RELEASE_DATES)

    chains = leftjoin(chains, select(entries, :pdb_id, :method, :resolution, :deposition_date);
                      on = :pdb_id)
    chains = leftjoin(chains, select(dates, :pdb_id, :release_date); on = :pdb_id)

    chains.tier     = tier_of.(chains.release_date)
    chains.why_not  = [why_not_selected(row) for row in eachrow(chains)]
    chains.selected = ismissing.(chains.why_not)

    sort!(chains, [:pdb_id, :chain_id])
    select!(chains, :pdb_id, :chain_id, :length, :unknown_fraction, :sequence_id,
            :method, :resolution, :deposition_date, :release_date,
            :tier, :selected, :why_not)
    write_table(CHAINS_OUT, chains)

    # Each distinct sequence once, named by its sequence_id. This is what the
    # clustering step will read.
    open(FASTA_OUT, "w") do io
        for (id, sequence) in enumerate(sequences)
            println(io, ">", id)
            println(io, sequence)
        end
    end
    say("wrote ", length(sequences), " sequences to ", relpath(FASTA_OUT, REPO_ROOT))

    # ---- summary for the person running this --------------------------------
    selected = filter(:selected => identity, chains)
    say("selected: ", nrow(selected), " chains in ",
        length(unique(selected.pdb_id)), " entries, ",
        length(unique(selected.sequence_id)), " distinct sequences")

    say("selected chains by tier:")
    by_tier = combine(groupby(selected, :tier),
                      nrow => :chains,
                      :pdb_id => (x -> length(unique(x))) => :entries,
                      :sequence_id => (x -> length(unique(x))) => :distinct_sequences)
    show(stdout, by_tier; allrows = true, summary = false, eltypes = false)
    println()

    say("chains not selected, by reason:")
    reasons = sort(combine(groupby(filter(:selected => !, chains), :why_not), nrow => :chains),
                   :chains; rev = true)
    show(stdout, reasons; allrows = true, summary = false, eltypes = false)
    println()
end

main()
