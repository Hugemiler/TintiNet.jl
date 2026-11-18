# mmcif.jl — read the tables we need out of an mmCIF structure file.
#
# An mmCIF file is a collection of named tables ("categories"). Most are
# written as a `loop_`: a list of column names, then one row per line.
#
#     loop_
#     _atom_site.group_PDB
#     _atom_site.id
#     ...
#     ATOM 1 N N . MET A 1 1 ? -5.392 -4.777 -7.145 1.00 53.69 ? 1 MET A N 1
#     ...
#     #
#
# This file reads chosen columns of chosen categories and nothing else. It is
# deliberately small: it does not understand every corner of the mmCIF format,
# only what the wwPDB writes for the tables this pipeline uses. Values that
# span several lines (text blocks that start with ';') are not supported, and
# none of the tables we read contain them.

using CodecZlib

"A table read from an mmCIF file: column name => values, all as text."
const CifTable = Dict{String, Vector{String}}

"Split one line of an mmCIF loop into its values. Handles 'quoted values'."
function cif_tokens(line)
    # Fast path: no quotes anywhere, plain whitespace-separated values.
    if !occursin('\'', line) && !occursin('"', line)
        return String.(split(line))
    end

    tokens = String[]
    i, n = firstindex(line), lastindex(line)
    while i <= n
        c = line[i]
        if isspace(c)
            i = nextind(line, i)
        elseif c == '\'' || c == '"'
            # Quoted value: ends at the same quote character followed by a
            # space or the end of the line.
            start = nextind(line, i)
            j = start
            while j <= n
                if line[j] == c && (j == n || isspace(line[nextind(line, j)]))
                    break
                end
                j = nextind(line, j)
            end
            push!(tokens, line[start:prevind(line, j)])
            i = j <= n ? nextind(line, j) : j
        else
            j = i
            while j <= n && !isspace(line[j])
                j = nextind(line, j)
            end
            push!(tokens, line[i:prevind(line, j)])
            i = j
        end
    end
    return tokens
end

"""
    read_cif_tables(path, wanted) -> Dict{String, CifTable}

Read the looped categories named in `wanted` from an mmCIF file (plain or
gzipped). `wanted` maps a category to the columns to keep, for example

    Dict("_atom_site" => ["label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"])

A category that is absent from the file is absent from the result. A wanted
column that is absent from the file comes back filled with "?".
"""
function read_cif_tables(path, wanted)
    tables = Dict{String, CifTable}()
    io = endswith(path, ".gz") ? GzipDecompressorStream(open(path)) : open(path)
    try
        read_cif_tables!(tables, io, wanted)
    finally
        close(io)
    end
    return tables
end

function read_cif_tables!(tables, io, wanted)
    line = readline(io)
    while true
        if strip(line) != "loop_"
            eof(io) && break
            line = readline(io)
            continue
        end

        # --- column names -----------------------------------------------------
        names = String[]
        line = readline(io)
        while startswith(line, '_')
            push!(names, strip(line))
            line = readline(io)
        end
        isempty(names) && continue
        category = String(first(split(names[1], '.')))

        if !haskey(wanted, category)
            continue                      # not interested; keep scanning from `line`
        end

        # Which positions in each row hold the columns we want?
        fields    = [String(last(split(name, '.'; limit = 2))) for name in names]
        keep      = wanted[category]
        positions = [findfirst(==(column), fields) for column in keep]
        columns   = [String[] for _ in keep]
        width     = length(names)

        # --- rows -------------------------------------------------------------
        # A row normally sits on one line. If a line holds fewer values than
        # there are columns, the row continues on the next line.
        pending = String[]
        while !(startswith(line, '#') || startswith(line, "loop_") ||
                startswith(line, "data_") || startswith(line, '_'))
            append!(pending, cif_tokens(line))
            while length(pending) >= width
                for (k, position) in enumerate(positions)
                    push!(columns[k], position === nothing ? "?" : pending[position])
                end
                deleteat!(pending, 1:width)
            end
            eof(io) && break
            line = readline(io)
        end

        tables[category] = CifTable(column => columns[k] for (k, column) in enumerate(keep))
        eof(io) && break
    end
    return tables
end

"Number of rows in a CifTable."
cif_rows(table::CifTable) = isempty(table) ? 0 : length(first(values(table)))
