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

### Written and tested

| # | Script | What it does | Writes |
|---|---|---|---|
| 01 | `01_fetch_sources.jl` | Downloads every enabled source in `sources.toml` | `work/raw/…`, `work/raw/FETCHED.tsv` |
| 02 | `02_index_entries.jl` | One row per PDB entry: molecule type, method, resolution, deposition date | `work/index/entries.tsv` |
| 03 | `03_fetch_release_dates.jl` | Asks the RCSB for each entry's release date | `work/index/release_dates.tsv` |
| 04 | `04_index_chains.jl` | One row per protein chain; assigns its date tier; marks which chains we want | `work/index/chains.tsv`, `work/index/sequences.fasta` |
| 05 | `05_fetch_structures.jl` | Downloads the structure file of each selected entry | `work/raw/mmCIF/…` |

### Not written yet

| # | Script | What it will do |
|---|---|---|
| 06 | `06_run_dssp.jl` | Run DSSP on each structure (container `dssp`) |
| 07 | `07_run_stride.jl` | Run STRIDE on each chain (container `stride`) |
| 08 | `08_features.jl` | Angles from coordinates, accessibility alone and in the assembly, contacts, quality flags |
| 09 | `09_cluster.jl` | Cluster sequences; search test structures against training (containers `mmseqs2`, `foldseek`) |
| 10 | `10_annotate.jl` | Join CATH labels and similarity-to-training onto the chain table |
| 20+ | `20_build_train.jl`, … | One script per dataset, writing into `data/` |
| 90 | `90_report.jl` | Counts, failures, label agreement, leakage checks |

## Where to change things

| I want to… | Edit |
|---|---|
| add or update a download | `sources.toml` |
| move the training cutoff or the test tiers | `TRAIN_BEFORE`, `TEST_CLEAN_FROM` in `04_index_chains.jl` |
| download more or fewer structures | `WANTED_METHODS`, `MAX_RESOLUTION`, `MIN_LENGTH` in `04_index_chains.jl` |
| download faster or slower | `SIMULTANEOUS_DOWNLOADS` in `05_fetch_structures.jl` |
| change where things are stored | the paths at the top of `common.jl` |

The rules in step 04 decide what is **downloaded**, so they are looser than the
rules for what is **trained on**. The stricter training rules will live in the
build scripts, where changing them costs minutes and no new download.

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

Steps 01–04 take about five minutes and use about 340 MB. Step 05 for all
192,593 entries has **not** been run; only a handful of files were fetched to
test it. Expect tens of gigabytes and several hours.

## Things to know

- **`chain_id` comes from `pdb_seqres`.** Whether it matches the chain naming
  inside each mmCIF file has to be checked when step 06 is written.
- **Release dates come from a web service**, not a file. If it changes its
  format, step 03 is the only script affected.
- **PDB index files change every week.** `work/raw/FETCHED.tsv` records the
  date and checksum of what was actually downloaded.
- **The benchmark and edge-case sets are declared in `sources.toml` but
  disabled**, because their exact download locations are not confirmed yet.
