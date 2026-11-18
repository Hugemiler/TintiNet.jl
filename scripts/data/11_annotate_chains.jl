# 11_annotate_chains.jl — one row per selected chain, with everything needed
# to decide which dataset it belongs to.
#
# READS    work/index/chains.tsv                       step 04
#          work/index/sequence_clusters.tsv            step 09
#          work/tools/chains/<ab>.chains.tsv           step 06 (what was observed)
#          work/raw/sifts/pdb_chain_cath_uniprot.csv.gz
#          work/raw/cath/cath-domain-list-v4_4_0.txt
# WRITES   work/index/chains_annotated.tsv
#
# RUN      julia --project=scripts/data scripts/data/11_annotate_chains.jl
#
# Columns added to those of chains.tsv
#   cluster_30 … cluster_90   sequence cluster at each identity level
#   asym_id                   the chain's name inside the mmCIF file
#   observed                  residues with coordinates
#   observed_fraction         observed / length
#   prepared                  true if step 06 produced data for this chain
#   problem                   why not, if it did not ("none" otherwise)
#   uniprot                   UniProt accession(s), ';'-separated
#   cath_domains              number of CATH domains mapped to the chain
#   cath_superfamilies        CATH superfamily codes such as 1.10.490.10,
#                             ';'-separated, sorted; empty if CATH has not
#                             classified this chain
#   cath_classes              CATH class numbers: 1 mainly alpha, 2 mainly
#                             beta, 3 alpha-beta, 4 few secondary structures,
#                             6 special
#
# CATH classifies domains, and a chain can hold several. So a chain can carry
# more than one superfamily.

include("common.jl")
using CodecZlib

const CHAINS        = joinpath(INDEX_DIR, "chains.tsv")
const CLUSTERS      = joinpath(INDEX_DIR, "sequence_clusters.tsv")
const CHAIN_STATUS  = joinpath(TOOLS_DIR, "chains")
const SIFTS         = joinpath(RAW_DIR, "sifts", "pdb_chain_cath_uniprot.csv.gz")
const CATH_DOMAINS  = joinpath(RAW_DIR, "cath", "cath-domain-list-v4_4_0.txt")
const OUTPUT        = joinpath(INDEX_DIR, "chains_annotated.tsv")

"""
CATH domain list: one line per domain, columns separated by spaces.
    1oaiA00     1    10     8    10   ...
    domain      C    A      T    H
Returns domain name => "C.A.T.H".
"""
function read_cath_superfamilies(path)
    superfamily = Dict{String, String}()
    for line in eachline(path)
        startswith(line, '#') && continue
        fields = split(line)
        length(fields) >= 5 || continue
        superfamily[fields[1]] = join(fields[2:5], '.')
    end
    return superfamily
end

"""
SIFTS mapping: PDB,CHAIN,SP_PRIMARY,CATH_ID (first line is a comment).
Returns (pdb_id, chain_id) => (set of UniProt accessions, set of CATH domains).
"""
function read_sifts(path)
    mapping = Dict{Tuple{String, String}, Tuple{Set{String}, Set{String}}}()
    open(GzipDecompressorStream, path) do io
        for line in eachline(io)
            (startswith(line, '#') || startswith(line, "PDB,")) && continue
            fields = split(line, ',')
            length(fields) == 4 || continue
            uniprots, domains = get!(mapping, (String(fields[1]), String(fields[2]))) do
                (Set{String}(), Set{String}())
            end
            push!(uniprots, fields[3])
            push!(domains, fields[4])
        end
    end
    return mapping
end

"Status of every chain step 06 handled, from all its per-folder tables."
function read_chain_status(folder)
    files = filter(f -> endswith(f, ".chains.tsv"), readdir(folder; join = true))
    tables = [CSV.read(f, DataFrame; delim = '\t', stringtype = String,
                       types = Dict(:chain_id => String, :asym_id => String,
                                    :pdb_id => String, :problem => String))
              for f in files]
    status = reduce(vcat, tables)
    return filter(:chain_id => c -> !ismissing(c) && c != "", status)
end

function main()
    chains = filter(:selected => identity, read_table(CHAINS))
    chains.chain_id = string.(chains.chain_id)
    say(nrow(chains), " selected chains")

    # --- sequence clusters ------------------------------------------------------
    chains = leftjoin(chains, read_table(CLUSTERS); on = :sequence_id)

    # --- what step 06 found -----------------------------------------------------
    status = read_chain_status(CHAIN_STATUS)
    say(nrow(status), " chains prepared by step 06")
    status = select(status, :pdb_id, :chain_id, :asym_id, :observed, :problem)
    chains = leftjoin(chains, unique(status, [:pdb_id, :chain_id]); on = [:pdb_id, :chain_id])
    chains.prepared = .!ismissing.(chains.observed) .& (coalesce.(chains.problem, "") .== "none")
    chains.observed = coalesce.(chains.observed, 0)
    chains.observed_fraction = round.(chains.observed ./ chains.length; digits = 3)

    # --- CATH and UniProt --------------------------------------------------------
    superfamily_of = read_cath_superfamilies(CATH_DOMAINS)
    sifts          = read_sifts(SIFTS)
    say(length(superfamily_of), " CATH domains; ", length(sifts), " chains in the SIFTS mapping")

    uniprot = String[]; n_domains = Int[]; superfamilies = String[]; classes = String[]
    for row in eachrow(chains)
        accessions, domains = get(sifts, (row.pdb_id, row.chain_id), (Set{String}(), Set{String}()))
        families = sort(unique(superfamily_of[d] for d in domains if haskey(superfamily_of, d)))
        push!(uniprot, join(sort(collect(accessions)), ';'))
        push!(n_domains, length(domains))
        push!(superfamilies, join(families, ';'))
        push!(classes, join(sort(unique(first(split(f, '.')) for f in families)), ';'))
    end
    chains.uniprot = uniprot
    chains.cath_domains = n_domains
    chains.cath_superfamilies = superfamilies
    chains.cath_classes = classes

    sort!(chains, [:pdb_id, :chain_id])
    write_table(OUTPUT, chains)

    # ---- summary ---------------------------------------------------------------
    say("prepared: ", count(chains.prepared), " of ", nrow(chains))
    say("with a CATH superfamily: ", count(!=(""), chains.cath_superfamilies))
    say("CATH coverage by tier:")
    coverage = combine(groupby(chains, :tier),
        nrow => :chains,
        :cath_superfamilies => (x -> round(100 * count(!=(""), x) / length(x); digits = 1)) => :percent_with_cath)
    show(stdout, coverage; allrows = true, summary = false, eltypes = false)
    println()
end

main()
