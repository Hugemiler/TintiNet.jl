# 90_report.jl — count everything, check the sets, write a report a person
# can read in five minutes.
#
# READS    work/index/*.tsv, work/logs/*, work/tools/chains/*.chains.tsv
#          data/<set>/chains.tsv, data/<set>/residues.arrow
# WRITES   data/REPORT.md
#
# RUN      julia --project=scripts/data scripts/data/90_report.jl
#
# The report answers:
#   1. How much made it through each step, and what was lost where?
#   2. How big is each set?
#   3. Do the labels look sane? Do DSSP and STRIDE agree? Do our own backbone
#      angles agree with DSSP's?
#   4. Do the hold-out folds keep training and held-out chains apart? How
#      close are the test sets to the pool?

include("common.jl")
using Arrow
using Printf
using Statistics

const SETS   = ["pool", "test_2021_2023", "test_post2023"]
const REPORT = joinpath(DATA_DIR, "REPORT.md")

percent(part, whole) = whole == 0 ? "n/a" : @sprintf("%.1f%%", 100 * part / whole)
count_lines(path) = isfile(path) ? countlines(path) : 0
has_text(x) = !ismissing(x) && x != ""

"STRIDE's letters reduced to 3 states, the same way as DSSP's."
function stride_ss3(letter)
    letter == ""                  && return ""
    letter in ("H", "G", "I")     && return "H"
    letter in ("E", "B", "b")     && return "E"
    return "C"
end

"Smallest difference between two angles in degrees."
angle_difference(a, b) = (d = abs(a - b); min(d, 360 - d))

"Everything we tally over the residues of one set, one block at a time."
function residue_statistics(path)
    total = observed = labelled = 0
    states = Dict("H" => 0, "E" => 0, "C" => 0)
    both = agree = 0
    angle_n = 0; angle_sum = 0.0; angle_over_5 = 0
    with_altloc = touching_other_chain = 0
    paired = paired_other_chain = 0
    multi = unanimous = 0

    for block in Arrow.Stream(path)
        ss3, stride = block.ss3, block.stride_ss
        for i in eachindex(ss3)
            total += 1
            block.observed[i] || continue
            observed += 1
            with_altloc += block.has_altloc[i]
            touching_other_chain += block.contacts_other[i] > 0

            state = ss3[i]
            state == "" && continue
            labelled += 1
            states[state] += 1

            other = stride_ss3(stride[i])
            if other != ""
                both += 1
                agree += other == state
            end

            ours, theirs = block.phi[i], block.dssp_phi[i]
            if !isnan(ours) && !isnan(theirs)
                d = angle_difference(ours, theirs)
                angle_n += 1; angle_sum += d; angle_over_5 += d > 5
            end

            partner = block.dssp_partner1[i]
            paired += partner != 0
            paired_other_chain += partner == -1

            if block.copies[i] >= 2
                multi += 1
                unanimous += max(block.share_h[i], block.share_e[i], block.share_c[i]) == 1
            end
        end
    end
    return (; total, observed, labelled, states, both, agree, angle_n, angle_sum, angle_over_5,
            with_altloc, touching_other_chain, paired, paired_other_chain, multi, unanimous)
end

function main()
    io = IOBuffer()
    line(args...) = println(io, args...)

    line("# Data report")
    line()
    line("Written by `scripts/data/90_report.jl` on ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ".")
    line()

    # ------------------------------------------------------------------------
    line("## 1. What made it through each step")
    line()
    chains    = read_table(joinpath(INDEX_DIR, "chains.tsv"))
    selected  = filter(:selected => identity, chains)
    entries   = sort(unique(selected.pdb_id))
    mmcif(id) = joinpath(RAW_DIR, "mmCIF", id[2:3], id * ".cif.gz")
    dssp(id)  = joinpath(TOOLS_DIR, "dssp", id[2:3], id * ".dssp.cif.gz")
    downloaded = count(isfile ∘ mmcif, entries)
    with_dssp  = count(isfile ∘ dssp, entries)

    annotated = read_table(joinpath(INDEX_DIR, "chains_annotated.tsv"))
    line("| Step | Count | Of |")
    line("|---|---|---|")
    line("| Protein chains in the PDB | ", nrow(chains), " | |")
    line("| Chains selected for download | ", nrow(selected), " | ", percent(nrow(selected), nrow(chains)), " of all |")
    line("| Entries selected | ", length(entries), " | |")
    line("| Entries downloaded | ", downloaded, " | ", percent(downloaded, length(entries)), " |")
    line("| Entries with DSSP output | ", with_dssp, " | ", percent(with_dssp, downloaded), " of downloaded |")
    line("| Entries DSSP failed on | ", count_lines(joinpath(LOGS_DIR, "07_dssp_failures.txt")), " | |")
    line("| Chains prepared (step 06) | ", count(annotated.prepared), " | ", percent(count(annotated.prepared), nrow(annotated)), " of selected |")
    line("| Chains STRIDE failed on | ", count_lines(joinpath(LOGS_DIR, "08_stride_failures.txt")), " | |")
    line("| Chains with a CATH superfamily | ", count(has_text, annotated.cath_superfamilies), " | ", percent(count(has_text, annotated.cath_superfamilies), nrow(annotated)), " of selected |")
    line()

    not_prepared = filter(:prepared => !, annotated)
    if nrow(not_prepared) > 0
        line("Chains not prepared, by reason:")
        line()
        line("| Reason | Chains |")
        line("|---|---|")
        reasons = [first(coalesce(p, "entry missing or unreadable"), 40) for p in not_prepared.problem]
        tally = sort(combine(groupby(DataFrame(reason = reasons), :reason), nrow => :n), :n; rev = true)
        for row in eachrow(first(tally, 8))
            line("| ", row.reason, " | ", row.n, " |")
        end
        line()
    end

    # ------------------------------------------------------------------------
    line("## 2. The sets")
    line()
    line("| Set | Chains | Entries | Distinct sequences | Clusters at 30% | Residues |")
    line("|---|---|---|---|---|---|")
    set_chains = Dict{String, DataFrame}()
    statistics = Dict{String, Any}()
    for name in SETS
        path = joinpath(DATA_DIR, name, "chains.tsv")
        isfile(path) || continue
        table = read_table(path)
        set_chains[name] = table
        say("reading residues of ", name, " ...")
        statistics[name] = residue_statistics(joinpath(DATA_DIR, name, "residues.arrow"))
        line("| `", name, "` | ", nrow(table), " | ", length(unique(table.pdb_id)), " | ",
             length(unique(table.sequence_id)), " | ", length(unique(table.cluster_30)), " | ",
             statistics[name].total, " |")
    end
    line()
    # ------------------------------------------------------------------------
    if haskey(set_chains, "pool")
        pool  = set_chains["pool"]
        folds = sort([String(c) for c in names(pool) if occursin(r"^fold\d+$", String(c))])
        families(x) = ismissing(x) ? String[] : String.(split(x, ';'; keepempty = false))
        fold_table  = read_table(joinpath(DATA_DIR, "splits", "superfamily_folds.tsv"))
        family_fold = Dict(String(f) => k for (f, k) in zip(fold_table.superfamily, fold_table.fold))

        line("## 2b. The hold-out folds of the pool")
        line()
        line("What each pool chain is when a fold is held out. See `data/README.md`.")
        line()
        line("| Fold | Superfamilies | Train | Clusters in train | Held out: full | partial | sequence only | Removed |")
        line("|---|---|---|---|---|---|---|---|")
        n = nrow(pool)
        for (k, fold) in enumerate(folds)
            role = pool[!, fold]
            line("| ", k, " | ", count(==(k), fold_table.fold), " | ", count(==("train"), role), " (",
                 percent(count(==("train"), role), n), ") | ",
                 length(unique(pool.cluster_30[role .== "train"])), " | ", count(==("full"), role), " | ",
                 count(==("partial"), role), " | ", count(==("sequence_only"), role), " | ",
                 count(==("removed"), role), " |")
        end
        line()

        roles = Matrix{String}(pool[!, folds])
        held(r) = r in ("full", "partial", "sequence_only")
        times_held = [count(held, roles[i, :]) for i in 1:n]
        strongest = [any(==("full"), roles[i, :]) ? "full" : any(==("partial"), roles[i, :]) ? "partial" :
                     any(==("sequence_only"), roles[i, :]) ? "sequence_only" : "never" for i in 1:n]
        line("Strongest hold-out each chain reaches in any fold: ",
             join(["$(count(==(s), strongest)) $s" for s in ("full", "partial", "sequence_only", "never")], ", "), ".")
        line("Held out in one fold: ", count(==(1), times_held), "; in two or more: ", count(>=(2), times_held), ".")
        line()

        line("Checks, which must all be 0:")
        line()
        line("| Fold | Training chains sharing a 30% cluster with a held-out or removed chain | Training chains with a superfamily of the held-out fold |")
        line("|---|---|---|")
        for (k, fold) in enumerate(folds)
            role = pool[!, fold]
            other_clusters = Set(pool.cluster_30[role .!= "train"])
            shared = count(i -> role[i] == "train" && pool.cluster_30[i] in other_clusters, 1:n)
            leaked = count(i -> role[i] == "train" &&
                                any(f -> family_fold[f] == k, families(pool.cath_superfamilies[i])), 1:n)
            line("| ", k, " | ", shared, " | ", leaked, " |")
        end
        line()
    end

    # ------------------------------------------------------------------------
    line("## 3. The labels")
    line()
    line("| Set | Residues observed | Helix | Strand | Coil | DSSP = STRIDE (3 states) | Our phi vs DSSP: mean diff | diff over 5° |")
    line("|---|---|---|---|---|---|---|---|")
    for name in SETS
        haskey(statistics, name) || continue
        s = statistics[name]
        line("| `", name, "` | ", percent(s.observed, s.total), " | ",
             percent(s.states["H"], s.labelled), " | ", percent(s.states["E"], s.labelled), " | ",
             percent(s.states["C"], s.labelled), " | ", percent(s.agree, s.both), " | ",
             s.angle_n == 0 ? "n/a" : @sprintf("%.2f°", s.angle_sum / s.angle_n), " | ",
             percent(s.angle_over_5, s.angle_n), " |")
    end
    line()
    line("| Set | With alternate conformations | Touching another chain | In a strand pair | of which partner in another chain | Same label in every copy of the sequence |")
    line("|---|---|---|---|---|---|")
    for name in SETS
        haskey(statistics, name) || continue
        s = statistics[name]
        line("| `", name, "` | ", percent(s.with_altloc, s.observed), " | ",
             percent(s.touching_other_chain, s.observed), " | ", percent(s.paired, s.labelled), " | ",
             percent(s.paired_other_chain, s.paired), " | ", percent(s.unanimous, s.multi), " |")
    end
    line()
    line("The last column counts only residues whose sequence appears in two or more chains of the set.")
    line()

    # ------------------------------------------------------------------------
    line("## 4. How close the test sets are to the pool")
    line()
    if haskey(set_chains, "pool")
        line("Share of each test set's chains that fall in a sequence cluster also present in the pool.")
        line()
        line("| Set | Chains | Identical sequence in pool | Cluster shared at 90% | 70% | 50% | 30% | None, even at 30% |")
        line("|---|---|---|---|---|---|---|---|")
        for name in SETS[2:end]
            haskey(set_chains, name) || continue
            table = set_chains[name]
            n = nrow(table)
            at(level) = percent(count(>=(level), table.pool_cluster_level), n)
            line("| `", name, "` | ", n, " | ", percent(count(table.pool_identical), n), " | ",
                 at(90), " | ", at(70), " | ", at(50), " | ", at(30), " | ",
                 percent(count(==(0), table.pool_cluster_level), n), " |")
        end
        line()
        line("The test sets are split from the pool by date only, so overlap is expected and is what the `pool_cluster_level` column is for. The comparison is with the whole pool, so it is a cautious one: any single fold's model trained on about three quarters of it.")
        line()
    end

    mkpath(DATA_DIR)
    write(REPORT, String(take!(io)))
    say("wrote ", relpath(REPORT, REPO_ROOT))
end

main()
