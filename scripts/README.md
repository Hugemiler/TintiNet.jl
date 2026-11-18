# scripts

Everything that turns "nothing on disk" into "datasets ready to load".

The idea, in one line: **declare** what you want in `sources.toml`, **fetch and
process** it inside `work/`, then **build** the datasets into `data/`.

```
sources.toml        what we download, declared in one place       (in git)
scripts/data/       one numbered script per step                  (in git)
dockerfiles/        one image per external tool                   (in git)
work/               downloads and intermediates                   (not in git)
data/               finished datasets, read by the model code     (not in git)
```

`work/` can be rebuilt from `sources.toml`. `data/` can be rebuilt from
`work/`. Deleting either loses time, not information.

## Running a step

From the repository root:

```bash
julia --project=scripts/data scripts/data/01_fetch_sources.jl
```

The first time, install the scripts' dependencies:

```bash
julia --project=scripts/data -e 'using Pkg; Pkg.instantiate()'
```

Run the steps in number order. Each one reads only what earlier steps wrote,
and each can be re-run on its own.

## The steps

Every script opens with a comment block saying what it READS, what it WRITES
and how to RUN it. Settings you might want to change are constants in a
"knobs" block near the top.

To run everything in order:

```bash
scripts/data/run_all.sh        # or: scripts/data/run_all.sh 6   to start at step 06
```

| # | Script | What it does | Writes |
|---|---|---|---|
| 01 | `01_fetch_sources.jl` | Downloads every enabled source in `sources.toml` | `work/raw/…`, `work/raw/FETCHED.tsv` |
| 02 | `02_index_entries.jl` | One row per PDB entry: molecule type, method, resolution, deposition date | `work/index/entries.tsv` |
| 03 | `03_fetch_release_dates.jl` | Asks the RCSB for each entry's release date | `work/index/release_dates.tsv` |
| 04 | `04_index_chains.jl` | One row per protein chain; assigns its date tier; marks which chains we want | `work/index/chains.tsv`, `work/index/sequences.fasta` |
| 05 | `05_fetch_structures.jl` | Downloads the structure file of each selected entry | `work/raw/mmCIF/…` |
| 06 | `06_prepare_chains.jl` | Reads each structure once: our own backbone angles, contacts, quality flags; writes each chain as a small file for STRIDE | `work/tools/chains/`, `work/tools/chain_pdb/` |
| 07 | `07_run_dssp.jl` | DSSP on every entry (container `dssp`) | `work/tools/dssp/` |
| 08 | `08_run_stride.jl` | STRIDE on every chain (container `stride`) | `work/tools/stride/` |
| 09 | `09_cluster_sequences.jl` | Sequence clusters at 30, 50, 70 and 90% identity (container `mmseqs2`) | `work/index/sequence_clusters.tsv` |
| 10 | `10_assemble_residues.jl` | Joins steps 06, 07 and 08 into one per-residue table | `work/tools/residues/` |
| 11 | `11_annotate_chains.jl` | Per chain: clusters, CATH superfamilies, how much was observed | `work/index/chains_annotated.tsv` |
| 20 | `20_build_sets.jl` | Builds train, val and the two test tiers | `data/<set>/` |
| 90 | `90_report.jl` | Counts, label checks, separation from train | `data/REPORT.md` |

Two helper files are shared: `common.jl` (paths, tables, downloads, running
containers) and `mmcif.jl` (reading tables out of structure files).

### Not written yet

| Script | What it will do |
|---|---|
| structure search | Search test structures against training with Foldseek, to say whether a fold is new (sequence clusters alone cannot) |
| edge-case sets | Chimeras, fold switchers, isoforms, designed proteins, disorder. Their sources are declared in `sources.toml` but disabled |
| legacy benchmarks | CB513, TS115, CASP |

## What a finished set looks like

Each folder under `data/` holds the same three files:

- `chains.tsv` — one row per chain, with every annotation. Plain text.
- `residues.arrow` — one row per residue. Written in blocks, one per structure
  folder; `chains.tsv` says which block (`partition`) a chain is in.
- `provenance.toml` — the rules, tool versions and counts.

The columns of each are documented at the top of `20_build_sets.jl`,
`10_assemble_residues.jl` and `06_prepare_chains.jl`.

## Where to change things

| I want to… | Edit |
|---|---|
| add or update a download | `sources.toml` |
| move the training cutoff or the test tiers | `TRAIN_BEFORE`, `TEST_CLEAN_FROM` in `04_index_chains.jl` |
| download more or fewer structures | `WANTED_METHODS`, `MAX_RESOLUTION`, `MIN_LENGTH` in `04_index_chains.jl` |
| download faster or slower | `SIMULTANEOUS_DOWNLOADS` in `05_fetch_structures.jl` |
| change which chains are trained on | the quality rules at the top of `20_build_sets.jl` |
| change the validation hold-out | `VALIDATION_SHARE`, `HOLDOUT_IDENTITY`, `SPLIT_SEED` in `20_build_sets.jl` |
| change how alternate conformations or contacts are handled | the rules listed at the top of `06_prepare_chains.jl` |
| use more or fewer processor cores | `WORKERS` in `07_run_dssp.jl` and `08_run_stride.jl`; `THREADS=…` for `run_all.sh` |
| change where things are stored | the paths at the top of `common.jl` |

The rules in step 04 decide what is **downloaded**, so they are looser than the
rules for what is **trained on**. The stricter training rules live in
`20_build_sets.jl`, where changing them costs minutes and no new download.

## What the tables look like

All tables are tab-separated text. Open them in a spreadsheet, `grep` them, or
read them in Julia with `read_table` from `common.jl`.

`work/index/chains.tsv`, the main one:

| Column | Meaning |
|---|---|
| `pdb_id`, `chain_id` | which chain |
| `length` | residues in the full sequence |
| `unknown_fraction` | share of residues written as X |
| `sequence_id` | chains with exactly the same sequence share this number |
| `method`, `resolution` | how the structure was solved |
| `deposition_date`, `release_date` | release date is the one cutoffs are defined on |
| `tier` | `train_pool`, `test_2021_2023` or `test_post2023` |
| `selected` | `true` if we want its structure |
| `why_not` | if not selected, the first rule it failed |

## What a full run of steps 01–04 gave (17 November 2026)

- 262,786 PDB entries; 257,476 contain protein.
- 1,117,340 protein chains; 189,073 distinct sequences.
- Selected (X-ray, 3.0 Å or better, 30 residues or longer):

| Tier | Chains | Entries | Distinct sequences |
|---|---|---|---|
| `train_pool` (released before 30 Sep 2021) | 367,773 | 146,970 | 89,292 |
| `test_2021_2023` | 37,390 | 14,286 | 9,142 |
| `test_post2023` (released from 1 Jun 2023) | 79,802 | 31,337 | 17,465 |
| **Total** | 484,965 | 192,593 | 111,680 |

Steps 01–04 take about five minutes and use about 340 MB.

## What the full run gave (18 November 2026)

| Step | Time | Disk | Result |
|---|---|---|---|
| 05 download | 3 h 20 min | 41 GB | 192,593 of 192,593 entries |
| 06 prepare chains | 28 min | 26 GB | 484,965 chains, 131.7 million residues |
| 07 DSSP | 2 h 30 min | 4 GB | 192,560 entries; 33 failed |
| 08 STRIDE | 6 min | 1.8 GB | 425 chains failed |
| 09 clustering | not timed (run earlier) | 0.3 GB | 189,073 sequences |
| 10 assemble | 3 min | 7.8 GB | 131.7 million residues |
| 11 annotate | 10 s | | 484,965 chains |
| 20 build sets | 7 min | 6.3 GB | see below |
| 90 report | 1 min | | `data/REPORT.md` |

| Set | Chains | Distinct sequences | Clusters at 30% | Residues |
|---|---|---|---|---|
| `train` | 247,178 | 67,802 | 18,486 | 66.0 million |
| `val` | 16,735 | 3,833 | 998 | 5.1 million |
| `test_2021_2023` | 29,252 | 7,232 | 3,234 | 8.0 million |
| `test_post2023` | 59,777 | 14,094 | 5,306 | 16.1 million |

The full numbers and the label checks are in `data/REPORT.md`.

## Things to know

- **`chain_id` comes from `pdb_seqres`.** Step 06 found every one of the
  484,965 selected chains under that name inside its structure file.
- **The 33 entries DSSP failed on are very large complexes** (ribosomes and
  the like). The likely cause is the 15-minute limit, but the log records
  only the entry, not the reason. Their chains are in the sets with
  empty DSSP columns. Raise `TIMEOUT_SECONDS` in `07_run_dssp.jl`, delete
  `work/logs/07_dssp_failures.txt` and re-run step 07 to try them again.
- **The 425 chains STRIDE failed on have only CA atoms** (182 old entries).
  Their STRIDE columns are empty.
- **Step 20 needs about 2 GB of memory**, because it writes one block of
  residues at a time. See `WRITE_AHEAD` there.
- **Release dates come from a web service**, not a file. If it changes its
  format, step 03 is the only script affected.
- **PDB index files change every week.** `work/raw/FETCHED.tsv` records the
  date and checksum of what was actually downloaded.
- **The benchmark and edge-case sets are declared in `sources.toml` but
  disabled**, because their exact download locations are not confirmed yet.
