# common.jl — the few things every data script shares.
#
# Included at the top of each numbered script with `include("common.jl")`.
# Keep this file small. If a helper is used by only one script, it belongs in
# that script.

using CSV
using DataFrames
using Dates
using Downloads
using SHA

# ----------------------------------------------------------------------------
# Where things live
# ----------------------------------------------------------------------------

const REPO_ROOT    = normpath(joinpath(@__DIR__, "..", ".."))
const SOURCES_FILE = joinpath(REPO_ROOT, "sources.toml")

const WORK_DIR  = joinpath(REPO_ROOT, "work")       # downloads and intermediates (gitignored)
const RAW_DIR   = joinpath(WORK_DIR, "raw")         # untouched downloads
const INDEX_DIR = joinpath(WORK_DIR, "index")       # tables describing what we have
const TOOLS_DIR = joinpath(WORK_DIR, "tools")       # outputs of DSSP, STRIDE, clustering...
const LOGS_DIR  = joinpath(WORK_DIR, "logs")

const DATA_DIR  = joinpath(REPO_ROOT, "data")       # constructed sets (gitignored)

# ----------------------------------------------------------------------------
# Talking to the person running the script
# ----------------------------------------------------------------------------

"Print a timestamped progress line."
say(args...) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", args...)

# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
# Every table the pipeline writes is tab-separated text, so it can be opened in
# a spreadsheet, grepped, or read back with `read_table`.

"Read a tab-separated table written by `write_table`."
function read_table(path)
    return CSV.read(path, DataFrame; delim = '\t', missingstring = "", stringtype = String)
end

"Write a DataFrame as a tab-separated table, creating the folder if needed."
function write_table(path, table)
    mkpath(dirname(path))
    CSV.write(path, table; delim = '\t', missingstring = "")
    say("wrote ", nrow(table), " rows to ", relpath(path, REPO_ROOT))
    return path
end

# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------

"""
    download_file(url, destination) -> destination

Download `url` to `destination`. The file appears under its final name only
when the download has finished, so an interrupted run never leaves a partial
file that looks complete.
"""
function download_file(url, destination)
    mkpath(dirname(destination))
    partial = destination * ".part"
    Downloads.download(url, partial)
    mv(partial, destination; force = true)
    return destination
end

"SHA-256 of a file, as a lowercase hex string."
sha256_of(path) = open(io -> bytes2hex(sha256(io)), path)
