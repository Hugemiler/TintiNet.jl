# 20_build_sets.jl — build the pool (with its five hold-out folds) and the
# two test sets.
#
# READS    work/index/chains_annotated.tsv        step 11
#          work/index/sequences.fasta             step 04
#          work/tools/residues/<ab>.arrow         step 10
# WRITES   data/<set>/chains.tsv                  one row per chain
#          data/<set>/residues.arrow              one row per residue
#          data/<set>/provenance.toml             how the set was made
#          data/splits/superfamily_folds.tsv      superfamily -> fold
#          data/splits/unclassified_cluster_folds.tsv
#
#          for <set> in: pool, test_2021_2023, test_post2023
#
# RUN      julia -t 8 --project=scripts/data scripts/data/20_build_sets.jl
#
# Nothing here touches work/. Change a rule below, run again, and the sets are
# rebuilt in minutes. The reasons for this design are in data/README.md.
#
# HOW THE SETS ARE DEFINED
#
#   Every set keeps only chains that pass the quality rules below.
#
#   pool = the tier `train_pool` (released before the AlphaFold 3 training
#   cutoff). It is not split once into train and validation. Instead:
#     1. Every CATH superfamily is given one of N_FOLDS folds: largest first,
#        each to the fold that is lightest so far.
#     2. Sequence clusters with no classified chain are dealt to the folds at
#        random (fixed seed).
#     3. For each fold k, the column `fold<k>` says what the chain is when
#        fold k is held out:
#          train          use for training
#          full           held out; every domain is in a fold-k superfamily
#          partial        held out; some domain is in fold k, some other is not
#          sequence_only  held out; the chain has no CATH label, and its
#                         sequence cluster is held out
#          removed        not held out, but in the same sequence cluster (at
#                         HOLDOUT_IDENTITY) as a held-out chain, so not usable
#                         for training either
#   To train with fold 3 held out, keep the rows where fold3 == "train".
#
#   test_2021_2023 and test_post2023 are the later date tiers, whole.
#
# COLUMNS ADDED TO chains.tsv
#   set                       which set the chain is in
#   partition                 which block of residues.arrow holds its residues
#   representative            true for one chain per sequence cluster (the
#                             best-resolved). Use these for scoring, so that a
#                             much-studied protein does not dominate.
#   fold1 … fold5             (pool) see above
#   pool_cluster_level        (test) the strictest identity level (90, 70, 50,
#                             30) at which the chain shares a sequence cluster
#                             with ANY pool chain; 0 if none even at 30%
#   pool_identical            (test) true if a pool chain has exactly the same
#                             sequence
#
# COLUMNS ADDED TO residues.arrow
#   sequence_id               as in chains.tsv
#   aa                        one-letter residue, from the PDB sequence record
#   ss3                       3-state label from DSSP: H (from H G I P),
#                             E (from E B), C (everything else); "" if DSSP
#                             gave no label for the residue
#   copies                    number of chains IN THIS SET with the same
#                             sequence that have a label at this position
#   share_h, share_e, share_c share of those copies labelled H, E, C. For a
#                             residue that is always helix, share_h is 1.

include("common.jl")
using Arrow
using Random
using TOML

# ---- knobs -----------------------------------------------------------------

# Quality rules, applied to every set.
const MAX_RESOLUTION        = 2.5      # ångström
const MIN_OBSERVED_RESIDUES = 30       # residues with coordinates
const MIN_OBSERVED_FRACTION = 0.5      # of the full sequence

# Hold-out folds.
const N_FOLDS          = 5
const HOLDOUT_IDENTITY = 30            # cluster level used to keep training and held-out chains apart
const SPLIT_SEED       = 20261117      # for dealing unclassified clusters to folds

# Cluster level used to pick one representative chain per cluster.
const REPRESENTATIVE_IDENTITY = 30

# Finished blocks of residues allowed to wait for the disk while writing.
# Lower it if the machine runs out of memory.
const WRITE_AHEAD = 4

# ----------------------------------------------------------------------------

const CHAINS       = joinpath(INDEX_DIR, "chains_annotated.tsv")
const FASTA        = joinpath(INDEX_DIR, "sequences.fasta")
const RESIDUES_DIR = joinpath(TOOLS_DIR, "residues")
const SPLITS_DIR   = joinpath(DATA_DIR, "splits")

const IDENTITY_LEVELS = [90, 70, 50, 30]          # strictest first
cluster_column(level) = Symbol("cluster_", level)

superfamilies_of(row) = ismissing(row.cath_superfamilies) ? String[] :
                        String.(split(row.cath_superfamilies, ';'; keepempty = false))

# ----------------------------------------------------------------------------
# Deciding which chain goes where
# ----------------------------------------------------------------------------

"Why a chain fails the quality rules, or `nothing` if it passes."
function quality_problem(row)
    row.prepared                                  || return "not prepared: " * coalesce(row.problem, "missing")
    row.resolution > MAX_RESOLUTION               && return "resolution"
    row.observed < MIN_OBSERVED_RESIDUES          && return "too few observed residues"
    row.observed_fraction < MIN_OBSERVED_FRACTION && return "too little of the chain observed"
    return nothing
end

"""
Give every superfamily a fold: largest first (by number of sequence clusters),
each to the fold that is lightest so far. Returns superfamily => fold and
superfamily => number of clusters.
"""
function assign_superfamily_folds(pool)
    holdout = cluster_column(HOLDOUT_IDENTITY)
    clusters_of = Dict{String, Set{Int}}()            # superfamily => its clusters
    for row in eachrow(pool), family in superfamilies_of(row)
        push!(get!(clusters_of, family, Set{Int}()), row[holdout])
    end
    size_of  = Dict(family => length(clusters) for (family, clusters) in clusters_of)
    families = sort(collect(keys(size_of)); by = family -> (-size_of[family], family))

    fold_of, load = Dict{String, Int}(), zeros(Int, N_FOLDS)
    for family in families
        fold = argmin(load)
        fold_of[family] = fold
        load[fold] += size_of[family]
    end
    return fold_of, size_of
end

"Sequence clusters with no classified chain, dealt to the folds at random. Returns cluster => fold."
function assign_unclassified_cluster_folds(pool)
    holdout    = cluster_column(HOLDOUT_IDENTITY)
    classified = Set(row[holdout] for row in eachrow(pool) if !isempty(superfamilies_of(row)))
    unclassified = sort(unique(c for c in pool[!, holdout] if !(c in classified)))
    dealt = shuffle(MersenneTwister(SPLIT_SEED), unclassified)
    return Dict(cluster => mod1(position, N_FOLDS) for (position, cluster) in enumerate(dealt))
end

"Add the columns fold1 … foldN: what each chain is when that fold is held out."
function add_fold_columns!(pool, family_fold, cluster_fold)
    families = [superfamilies_of(row) for row in eachrow(pool)]
    clusters = pool[!, cluster_column(HOLDOUT_IDENTITY)]
    for fold in 1:N_FOLDS
        in_fold(family) = family_fold[family] == fold
        has_domain_in_fold = [any(in_fold, f) for f in families]
        cluster_dealt_here = [get(cluster_fold, c, 0) == fold for c in clusters]
        held_out_clusters  = Set(clusters[has_domain_in_fold .| cluster_dealt_here])

        role = Vector{String}(undef, nrow(pool))
        for i in 1:nrow(pool)
            role[i] = if has_domain_in_fold[i]
                all(in_fold, families[i]) ? "full" : "partial"
            elseif !(clusters[i] in held_out_clusters)
                "train"
            elseif isempty(families[i])
                "sequence_only"
            else
                "removed"
            end
        end
        pool[!, Symbol("fold", fold)] = role
    end
    return pool
end

"Mark one chain per cluster as representative: best resolution, then most complete."
function mark_representatives!(chains)
    chains.representative = falses(nrow(chains))
    level = cluster_column(REPRESENTATIVE_IDENTITY)
    ranking = sortperm(chains, [level, :resolution, order(:observed_fraction; rev = true), :pdb_id, :chain_id])
    seen = Set{Int}()
    for i in ranking
        cluster = chains[i, level]
        cluster in seen && continue
        push!(seen, cluster)
        chains.representative[i] = true
    end
    return chains
end

"How close is each chain to the pool, in terms of shared sequence clusters?"
function add_similarity_to_pool!(chains, pool)
    pool_sequences = Set(pool.sequence_id)
    chains.pool_identical = [id in pool_sequences for id in chains.sequence_id]
    chains.pool_cluster_level = zeros(Int, nrow(chains))
    for level in reverse(IDENTITY_LEVELS)             # 30 first, so stricter levels overwrite
        column = cluster_column(level)
        pool_clusters = Set(pool[!, column])
        for i in 1:nrow(chains)
            chains[i, column] in pool_clusters && (chains.pool_cluster_level[i] = level)
        end
    end
    return chains
end

# ----------------------------------------------------------------------------
# Residues
# ----------------------------------------------------------------------------

"DSSP's 8 states reduced to 3. Empty stays empty (no label)."
function ss3_of(dssp_letter)
    dssp_letter == ""               && return ""
    dssp_letter in ("H", "G", "I", "P") && return "H"
    dssp_letter in ("E", "B")       && return "E"
    return "C"
end

"Every distinct sequence, indexed by sequence_id."
function read_sequences(path)
    sequences = String[]
    for line in eachline(path)
        startswith(line, '>') || push!(sequences, line)
    end
    return sequences
end

"The residues of one structure folder that belong to the given chains."
function residues_for(folder, wanted)                 # wanted: (pdb_id, chain_id) => sequence_id
    table = DataFrame(Arrow.Table(joinpath(RESIDUES_DIR, folder * ".arrow")); copycols = true)
    keep  = [haskey(wanted, (table.pdb_id[i], table.chain_id[i])) for i in 1:nrow(table)]
    table = table[keep, :]
    table.sequence_id = [wanted[(table.pdb_id[i], table.chain_id[i])] for i in 1:nrow(table)]
    return table
end

"""
Write one set: chains.tsv, residues.arrow, provenance.toml.

residues.arrow is written as one block ("partition") per structure folder, so
it can be read a block at a time. chains.tsv says which block each chain is in.
"""
function write_set(name, chains, sequences, rules)
    folder_of(pdb_id) = pdb_id[2:3]
    chains  = sort(chains, [:pdb_id, :chain_id])
    folders = sort(unique(folder_of.(chains.pdb_id)))
    folders = filter(f -> isfile(joinpath(RESIDUES_DIR, f * ".arrow")), folders)
    if isempty(folders)
        say("  ", name, ": no chains with residue data; nothing written")
        return
    end
    partition_of = Dict(f => i for (i, f) in enumerate(folders))
    chains.set       = fill(name, nrow(chains))
    chains.partition = [get(partition_of, folder_of(id), 0) for id in chains.pdb_id]

    wanted_by_folder = Dict(f => Dict{Tuple{String, String}, Int}() for f in folders)
    for row in eachrow(chains)
        haskey(wanted_by_folder, folder_of(row.pdb_id)) || continue
        wanted_by_folder[folder_of(row.pdb_id)][(row.pdb_id, row.chain_id)] = row.sequence_id
    end

    # Pass 1: for every sequence, count the labels seen at each position
    # across all its copies in this set.
    say("  ", name, ": counting labels across identical sequences ...")
    counts = Dict{Int, Matrix{UInt16}}()              # sequence_id => positions x (H, E, C)
    column_of = Dict("H" => 1, "E" => 2, "C" => 3)
    for folder in folders
        table = residues_for(folder, wanted_by_folder[folder])
        for i in 1:nrow(table)
            state = ss3_of(table.dssp_ss[i])
            state == "" && continue
            sequence_id = table.sequence_id[i]
            matrix = get!(counts, sequence_id) do
                zeros(UInt16, length(sequences[sequence_id]), 3)
            end
            position = table.seq_index[i]
            position <= size(matrix, 1) && (matrix[position, column_of[state]] += 1)
        end
    end

    # Pass 2: add the new columns and write, one block per folder.
    say("  ", name, ": writing residues ...")
    n_residues = Threads.Atomic{Int}(0)
    function block(folder)
        table = residues_for(folder, wanted_by_folder[folder])
        n = nrow(table)
        aa = fill("X", n); ss3 = fill("", n); copies = zeros(Int32, n)
        share_h = fill(NaN32, n); share_e = fill(NaN32, n); share_c = fill(NaN32, n)
        for i in 1:n
            sequence = sequences[table.sequence_id[i]]
            position = table.seq_index[i]
            position <= length(sequence) && (aa[i] = string(sequence[position]))
            ss3[i] = ss3_of(table.dssp_ss[i])
            matrix = get(counts, table.sequence_id[i], nothing)
            (matrix === nothing || position > size(matrix, 1)) && continue
            total = Int(matrix[position, 1]) + Int(matrix[position, 2]) + Int(matrix[position, 3])
            total == 0 && continue
            copies[i]  = total
            share_h[i] = matrix[position, 1] / total
            share_e[i] = matrix[position, 2] / total
            share_c[i] = matrix[position, 3] / total
        end
        table.aa = aa; table.ss3 = ss3; table.copies = copies
        table.share_h = share_h; table.share_e = share_e; table.share_c = share_c
        Threads.atomic_add!(n_residues, n)
        return table
    end

    folder = joinpath(DATA_DIR, name)
    mkpath(folder)
    # One block at a time, on purpose. Given all the blocks at once, Arrow loads
    # every one of them before writing any, which does not fit in memory for
    # the training set. The file is renamed at the end, so a run that is killed
    # cannot leave half a file that looks finished.
    partial = joinpath(folder, "residues.arrow.part")
    open(Arrow.Writer, partial; compress = :zstd, ntasks = WRITE_AHEAD) do writer
        for f in folders
            Arrow.write(writer, block(f))
        end
    end
    mv(partial, joinpath(folder, "residues.arrow"); force = true)
    write_table(joinpath(folder, "chains.tsv"), chains)

    provenance = Dict(
        "set"      => name,
        "built_at" => Dates.format(now(), "yyyy-mm-dd HH:MM"),
        "built_by" => "scripts/data/20_build_sets.jl",
        "counts"   => Dict(
            "chains"             => nrow(chains),
            "entries"            => length(unique(chains.pdb_id)),
            "distinct_sequences" => length(unique(chains.sequence_id)),
            "clusters_30"        => length(unique(chains.cluster_30)),
            "representatives"    => count(chains.representative),
            "residues"           => n_residues[],
            "partitions"         => length(folders),
        ),
        "rules" => rules,
        "tools" => Dict("dssp" => DSSP_IMAGE, "stride" => STRIDE_IMAGE, "mmseqs2" => MMSEQS_IMAGE),
        "sources" => "see work/raw/FETCHED.tsv for download dates and checksums",
    )
    open(io -> TOML.print(io, provenance; sorted = true), joinpath(folder, "provenance.toml"), "w")
    say("  ", name, ": ", nrow(chains), " chains, ", n_residues[], " residues")
end

# ----------------------------------------------------------------------------

function main()
    chains = read_table(CHAINS)
    chains.chain_id = string.(chains.chain_id)
    say(nrow(chains), " annotated chains")

    # --- quality ---------------------------------------------------------------
    problems = [quality_problem(row) for row in eachrow(chains)]
    good = chains[isnothing.(problems), :]
    say(nrow(good), " pass the quality rules")
    say("left out by the quality rules:")
    reasons = [first(split(p, ':')) for p in problems if p !== nothing]
    tally = sort(combine(groupby(DataFrame(reason = reasons), :reason), nrow => :chains), :chains; rev = true)
    show(stdout, tally; allrows = true, summary = false, eltypes = false); println()

    # --- the pool and its folds -------------------------------------------------
    pool = good[good.tier .== "train_pool", :]
    family_fold, family_size = assign_superfamily_folds(pool)
    cluster_fold             = assign_unclassified_cluster_folds(pool)
    add_fold_columns!(pool, family_fold, cluster_fold)

    mkpath(SPLITS_DIR)
    families = sort(collect(keys(family_fold)))
    write_table(joinpath(SPLITS_DIR, "superfamily_folds.tsv"),
                DataFrame(superfamily = families,
                          fold        = [family_fold[f] for f in families],
                          clusters    = [family_size[f] for f in families]))
    clusters = sort(collect(keys(cluster_fold)))
    write_table(joinpath(SPLITS_DIR, "unclassified_cluster_folds.tsv"),
                DataFrame(cluster_column(HOLDOUT_IDENTITY) => clusters,
                          :fold => [cluster_fold[c] for c in clusters]))

    say(nrow(pool), " pool chains; ", length(family_fold), " superfamilies in ", N_FOLDS, " folds")
    for fold in 1:N_FOLDS
        role = pool[!, Symbol("fold", fold)]
        say("  fold ", fold, ": ", join(["$(count(==(r), role)) $r"
            for r in ("train", "full", "partial", "sequence_only", "removed")], ", "))
    end

    # --- test tiers ----------------------------------------------------------------
    sets = Dict(
        "pool"           => pool,
        "test_2021_2023" => good[good.tier .== "test_2021_2023", :],
        "test_post2023"  => good[good.tier .== "test_post2023", :],
    )
    for name in ("test_2021_2023", "test_post2023")
        add_similarity_to_pool!(sets[name], pool)
    end
    foreach(mark_representatives!, values(sets))

    # --- write -----------------------------------------------------------------------
    rules = Dict(
        "max_resolution"          => MAX_RESOLUTION,
        "min_observed_residues"   => MIN_OBSERVED_RESIDUES,
        "min_observed_fraction"   => MIN_OBSERVED_FRACTION,
        "folds"                   => N_FOLDS,
        "holdout_identity"        => HOLDOUT_IDENTITY,
        "split_seed"              => SPLIT_SEED,
        "representative_identity" => REPRESENTATIVE_IDENTITY,
    )
    say("reading sequences ...")
    sequences = read_sequences(FASTA)
    for name in ("test_2021_2023", "test_post2023", "pool")
        write_set(name, sets[name], sequences, rules)
    end
end

main()
