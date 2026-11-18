# 09_cluster_sequences.jl — group similar sequences, at several strictness levels.
#
# READS    work/index/sequences.fasta          every distinct sequence once
# WRITES   work/tools/clusters/seq<NN>_cluster.tsv    raw MMseqs2 output per level
#          work/index/sequence_clusters.tsv           one row per sequence_id
#
# RUN      julia --project=scripts/data scripts/data/09_cluster_sequences.jl
#
# NEEDS    Docker image tintinet/mmseqs2 (see dockerfiles/mmseqs2)
#
# Columns of sequence_clusters.tsv
#   sequence_id    as in chains.tsv
#   cluster_30     the sequence_id of this sequence's cluster representative at
#                  30% identity. Two sequences with the same value are in the
#                  same cluster.
#   cluster_50, cluster_70, cluster_90    the same at stricter identity levels
#
# Why several levels
#   30% is the usual convention for "unrelated enough", but it is a convention.
#   Keeping several levels lets us test how much the choice matters without
#   re-running anything.
#
# This step does not need the structure files, only the sequences, so it can
# run while structures are still downloading.

include("common.jl")

# ---- knobs -----------------------------------------------------------------
const IDENTITY_LEVELS = [30, 50, 70, 90]     # percent sequence identity
const MIN_COVERAGE    = 0.8                  # alignment must cover this share of BOTH sequences
# ----------------------------------------------------------------------------

const FASTA       = joinpath(INDEX_DIR, "sequences.fasta")
const CLUSTER_DIR = joinpath(TOOLS_DIR, "clusters")
const OUTPUT      = joinpath(INDEX_DIR, "sequence_clusters.tsv")

"Run MMseqs2 clustering at one identity level, unless its output already exists."
function cluster_at(level)
    prefix = "tools/clusters/seq$(level)"                 # relative to work/
    result = joinpath(WORK_DIR, prefix * "_cluster.tsv")
    if isfile(result)
        say("have   clustering at ", level, "%")
        return result
    end
    say("run    clustering at ", level, "% identity ...")
    run(in_container(MMSEQS_IMAGE,
        "easy-cluster", "index/sequences.fasta", prefix, "tools/clusters/tmp$(level)",
        "--min-seq-id", string(level / 100),
        "-c", string(MIN_COVERAGE), "--cov-mode", "0",
        "-v", "1"))
    rm(joinpath(CLUSTER_DIR, "tmp$(level)"); recursive = true, force = true)
    return result
end

"MMseqs2 writes two columns: representative, member. Read them as integers."
function read_clusters(path, level)
    table = CSV.read(path, DataFrame; delim = '\t', header = [:representative, :sequence_id])
    return rename(select(table, :sequence_id, :representative),
                  :representative => Symbol("cluster_", level))
end

function main()
    mkpath(CLUSTER_DIR)

    combined = nothing
    for level in IDENTITY_LEVELS
        clusters = read_clusters(cluster_at(level), level)
        say("  ", level, "%: ", nrow(clusters), " sequences in ",
            length(unique(clusters[!, 2])), " clusters")
        combined = combined === nothing ? clusters : innerjoin(combined, clusters; on = :sequence_id)
    end

    sort!(combined, :sequence_id)
    write_table(OUTPUT, combined)
end

main()
