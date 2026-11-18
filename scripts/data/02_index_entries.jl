# 02_index_entries.jl — one row per PDB entry: what it is and how it was solved.
#
# READS    work/raw/pdb/pdb_entry_type.txt
#          work/raw/pdb/entries.idx
# WRITES   work/index/entries.tsv
#
# RUN      julia --project=scripts/data scripts/data/02_index_entries.jl
#
# Columns of entries.tsv
#   pdb_id            four-character identifier, lowercase
#   molecule_type     prot | nuc | prot-nuc | other
#   method            diffraction | NMR | EM | other
#   resolution        in ångström; empty when the entry has none (e.g. NMR)
#   deposition_date   yyyy-mm-dd. Note: deposition, not release. The release
#                     date is added by 03_fetch_release_dates.jl.

include("common.jl")

const ENTRY_TYPE_FILE = joinpath(RAW_DIR, "pdb", "pdb_entry_type.txt")
const ENTRIES_IDX     = joinpath(RAW_DIR, "pdb", "entries.idx")
const OUTPUT          = joinpath(INDEX_DIR, "entries.tsv")

# ----------------------------------------------------------------------------
# pdb_entry_type.txt:   101m <tab> prot <tab> diffraction
# ----------------------------------------------------------------------------
function read_entry_types(path)
    table = DataFrame(pdb_id = String[], molecule_type = String[], method = String[])
    for line in eachline(path)
        fields = split(line, '\t')
        length(fields) == 3 || continue
        push!(table, (lowercase(fields[1]), fields[2], fields[3]))
    end
    return table
end

# ----------------------------------------------------------------------------
# entries.idx: two header lines, then tab-separated fields
#   IDCODE, HEADER, ACCESSION DATE, COMPOUND, SOURCE, AUTHOR LIST, RESOLUTION, EXPERIMENT TYPE
# We keep the identifier, the date (mm/dd/yy) and the resolution.
# ----------------------------------------------------------------------------

"Turn 'mm/dd/yy' into a Date. The PDB began in 1971, so 71-99 means 19xx."
function parse_deposition_date(text)
    m = match(r"^(\d\d)/(\d\d)/(\d\d)$", strip(text))
    m === nothing && return missing
    month, day, year2 = parse.(Int, m.captures)
    year = year2 >= 71 ? 1900 + year2 : 2000 + year2
    return Date(year, month, day)
end

"""
Resolution as a number. The field can be 'NOT' (no resolution), or list several
values for multi-method entries ('1.9, 2.3'); we take the first, which is the
one the PDB reports as primary.
"""
function parse_resolution(text)
    first_value = strip(first(split(text, ',')))
    return something(tryparse(Float64, first_value), missing)
end

function read_entries_idx(path)
    table = DataFrame(pdb_id = String[],
                      deposition_date = Union{Date, Missing}[],
                      resolution = Union{Float64, Missing}[])
    for (line_number, line) in enumerate(eachline(path))
        line_number <= 2 && continue            # header and its underline
        fields = split(line, '\t')
        length(fields) >= 7 || continue
        push!(table, (lowercase(fields[1]),
                      parse_deposition_date(fields[3]),
                      parse_resolution(fields[7])))
    end
    return table
end

function main()
    types = read_entry_types(ENTRY_TYPE_FILE)
    idx   = read_entries_idx(ENTRIES_IDX)
    say("entry types: ", nrow(types), " entries;  entries.idx: ", nrow(idx), " entries")

    entries = leftjoin(types, idx; on = :pdb_id)
    sort!(entries, :pdb_id)
    select!(entries, :pdb_id, :molecule_type, :method, :resolution, :deposition_date)

    write_table(OUTPUT, entries)

    # A quick look, so the person running this sees whether it is sane.
    say("by molecule type and method:")
    summary = sort(combine(groupby(entries, [:molecule_type, :method]), nrow => :entries),
                   :entries; rev = true)
    show(stdout, summary; allrows = true, summary = false, eltypes = false)
    println()
end

main()
