# 10_assemble_residues.jl — put everything we know about each residue in one
# table: our own measurements (step 06), DSSP (step 07) and STRIDE (step 08).
#
# READS    work/tools/chains/<ab>.arrow                  from step 06
#          work/tools/dssp/<ab>/<1abc>.dssp.cif.gz       from step 07
#          work/tools/stride/<ab>.stride.gz              from step 08
# WRITES   work/tools/residues/<ab>.arrow                one row per residue
#
# RUN      julia -t 16 --project=scripts/data scripts/data/10_assemble_residues.jl
#          julia -t 16 --project=scripts/data scripts/data/10_assemble_residues.jl cx c7
#          julia -t 16 --project=scripts/data scripts/data/10_assemble_residues.jl --force
#
# The output has every column of the step-06 table, plus:
#   dssp_ss          DSSP's 8-state letter: H G I P (helices), E B (strands),
#                    T S (turn, bend), "-" for none, "" if DSSP has no row
#   dssp_acc         solvent accessibility in Å², chain in the context of the
#                    whole file
#   dssp_phi, dssp_psi
#   dssp_partner1, dssp_partner2
#                    strand partners. A positive number is the seq_index of
#                    the partner in the SAME chain; -1 means the partner is in
#                    another chain; 0 means no partner.
#   stride_ss        STRIDE's letter: H G I (helices), E B b (strands),
#                    T (turn), C (coil), "" if STRIDE has no row
#   stride_phi, stride_psi
#   stride_area      solvent accessibility in Å², chain on its own
#
# Rows are matched on (entry, chain name inside the file, position in the
# full sequence), which all three sources share.
#
# A folder whose output exists is skipped. Name folders to do only those;
# pass --force to redo.

include("common.jl")
include("mmcif.jl")
using Arrow

const CHAINS_DIR   = joinpath(TOOLS_DIR, "chains")
const DSSP_DIR     = joinpath(TOOLS_DIR, "dssp")
const STRIDE_DIR   = joinpath(TOOLS_DIR, "stride")
const RESIDUES_DIR = joinpath(TOOLS_DIR, "residues")

const DSSP_TABLES = Dict(
    "_dssp_struct_summary" => ["label_asym_id", "label_seq_id", "secondary_structure",
                               "accessibility", "phi", "psi"],
    "_dssp_struct_ladder"  => ["type",
                               "beg_1_label_asym_id", "beg_1_label_seq_id", "end_1_label_seq_id",
                               "beg_2_label_asym_id", "beg_2_label_seq_id", "end_2_label_seq_id"],
)

"A number from DSSP or STRIDE text; NaN when it is '.', '?' or absent."
number(text) = Float32(something(tryparse(Float64, text), NaN))

"""
Strand partners from DSSP's ladder table.

A ladder is two paired strands. Strand 1 runs from beg_1 to end_1. Strand 2 is
listed from the residue paired with beg_1 to the one paired with end_1, so for
an anti-parallel ladder its numbers run downwards.

Returns (asym, seq_index) => list of partner (asym, seq_index).
"""
function strand_partners(ladders::CifTable)
    partners = Dict{Tuple{String, Int}, Vector{Tuple{String, Int}}}()
    for row in 1:cif_rows(ladders)
        asym1, asym2 = ladders["beg_1_label_asym_id"][row], ladders["beg_2_label_asym_id"][row]
        beg1 = tryparse(Int, ladders["beg_1_label_seq_id"][row])
        end1 = tryparse(Int, ladders["end_1_label_seq_id"][row])
        beg2 = tryparse(Int, ladders["beg_2_label_seq_id"][row])
        end2 = tryparse(Int, ladders["end_2_label_seq_id"][row])
        (beg1 === nothing || end1 === nothing || beg2 === nothing || end2 === nothing) && continue

        step2 = sign(end2 - beg2)                    # +1 parallel, -1 anti-parallel, 0 single pair
        pairs = min(abs(end1 - beg1), abs(end2 - beg2))
        for k in 0:pairs
            a = (asym1, beg1 + k)
            b = (asym2, beg2 + k * step2)
            push!(get!(partners, a, Tuple{String, Int}[]), b)
            push!(get!(partners, b, Tuple{String, Int}[]), a)
        end
    end
    return partners
end

"Partner as stored in the table: seq_index if same chain, -1 if other chain, 0 if none."
function partner_code(partners, asym, k)
    length(partners) >= k || return Int32(0)
    partner_asym, partner_index = partners[k]
    return partner_asym == asym ? Int32(partner_index) : Int32(-1)
end

"Read a folder's STRIDE file: (pdb_id, asym, seq_index) => (ss, phi, psi, area)."
function read_stride(path)
    result = Dict{Tuple{String, String, Int}, Tuple{String, Float32, Float32, Float32}}()
    isfile(path) || return result
    pdb_id, asym = "", ""
    open(GzipDecompressorStream, path) do io
        for line in eachline(io)
            if startswith(line, "CHAIN ")
                # "CHAIN 1abc_AA": the chain name is everything after the first underscore.
                name = line[7:end]
                pdb_id, asym = name[1:4], name[6:end]
            elseif startswith(line, "ASG")
                fields = split(line)
                length(fields) >= 10 || continue
                seq_index = tryparse(Int, fields[4])
                seq_index === nothing && continue
                result[(pdb_id, asym, seq_index)] =
                    (String(fields[6]), number(fields[8]), number(fields[9]), number(fields[10]))
            end
        end
    end
    return result
end

function assemble_folder(folder)
    residues = DataFrame(Arrow.Table(joinpath(CHAINS_DIR, folder * ".arrow")); copycols = true)
    n = nrow(residues)

    dssp_ss  = fill("", n);       dssp_acc = fill(NaN32, n)
    dssp_phi = fill(NaN32, n);    dssp_psi = fill(NaN32, n)
    partner1 = zeros(Int32, n);   partner2 = zeros(Int32, n)
    stride_ss  = fill("", n);     stride_area = fill(NaN32, n)
    stride_phi = fill(NaN32, n);  stride_psi  = fill(NaN32, n)

    # Where is each residue in the table?
    row_of = Dict{Tuple{String, String, Int}, Int}()
    for i in 1:n
        row_of[(residues.pdb_id[i], residues.asym_id[i], Int(residues.seq_index[i]))] = i
    end

    # --- DSSP: one file per entry --------------------------------------------
    for pdb_id in unique(residues.pdb_id)
        path = joinpath(DSSP_DIR, folder, pdb_id * ".dssp.cif.gz")
        isfile(path) || continue
        tables = try
            read_cif_tables(path, DSSP_TABLES)
        catch
            continue                                  # unreadable output: leave DSSP columns empty
        end
        haskey(tables, "_dssp_struct_summary") || continue
        summary  = tables["_dssp_struct_summary"]
        partners = haskey(tables, "_dssp_struct_ladder") ?
                   strand_partners(tables["_dssp_struct_ladder"]) :
                   Dict{Tuple{String, Int}, Vector{Tuple{String, Int}}}()

        for r in 1:cif_rows(summary)
            asym      = summary["label_asym_id"][r]
            seq_index = tryparse(Int, summary["label_seq_id"][r])
            seq_index === nothing && continue
            i = get(row_of, (pdb_id, asym, seq_index), 0)
            i == 0 && continue                        # a chain we did not select
            letter      = summary["secondary_structure"][r]
            dssp_ss[i]  = letter in (".", "?") ? "-" : letter
            dssp_acc[i] = number(summary["accessibility"][r])
            dssp_phi[i] = number(summary["phi"][r])
            dssp_psi[i] = number(summary["psi"][r])
            mine = get(partners, (asym, seq_index), Tuple{String, Int}[])
            partner1[i] = partner_code(mine, asym, 1)
            partner2[i] = partner_code(mine, asym, 2)
        end
    end

    # --- STRIDE: one file per folder -----------------------------------------
    for ((pdb_id, asym, seq_index), (ss, phi, psi, area)) in
            read_stride(joinpath(STRIDE_DIR, folder * ".stride.gz"))
        i = get(row_of, (pdb_id, asym, seq_index), 0)
        i == 0 && continue
        stride_ss[i], stride_phi[i], stride_psi[i], stride_area[i] = ss, phi, psi, area
    end

    residues.dssp_ss  = dssp_ss;    residues.dssp_acc = dssp_acc
    residues.dssp_phi = dssp_phi;   residues.dssp_psi = dssp_psi
    residues.dssp_partner1 = partner1
    residues.dssp_partner2 = partner2
    residues.stride_ss  = stride_ss;    residues.stride_area = stride_area
    residues.stride_phi = stride_phi;   residues.stride_psi  = stride_psi

    mkpath(RESIDUES_DIR)
    partial = joinpath(RESIDUES_DIR, folder * ".arrow.part")
    Arrow.write(partial, residues; compress = :zstd)
    mv(partial, joinpath(RESIDUES_DIR, folder * ".arrow"); force = true)
    return n
end

function main(arguments)
    force = "--force" in arguments
    named = filter(a -> !startswith(a, "--"), arguments)
    available = sort([replace(f, ".arrow" => "") for f in readdir(CHAINS_DIR) if endswith(f, ".arrow")])
    folders   = isempty(named) ? available : named
    todo = [f for f in folders if force || !isfile(joinpath(RESIDUES_DIR, f * ".arrow"))]
    say(length(folders), " folders, ", length(todo), " to assemble, using ",
        Threads.nthreads(), " threads")

    finished = Threads.Atomic{Int}(0)
    total    = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for folder in todo
        n = try
            assemble_folder(folder)
        catch err
            say("folder ", folder, " failed: ", first(sprint(showerror, err), 200))
            0
        end
        Threads.atomic_add!(total, n)
        k = Threads.atomic_add!(finished, 1) + 1
        k % 50 == 0 && say(k, " of ", length(todo), " folders")
    end
    say("done: ", total[], " residues assembled")
end

main(ARGS)
