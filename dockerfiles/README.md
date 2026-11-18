# dockerfiles

One folder per tool, each with its own `Dockerfile`. Every external program or
non-Julia runtime the project needs lives here with its version pinned, so
nothing depends on what happens to be installed on the host.

## How to read a Dockerfile here

Each one starts with the same header:

- what the tool is for, in a sentence;
- which version, and where it comes from;
- what it reads and writes;
- the exact `docker build` command;
- an example `docker run` command;
- whether it has been built and run yet.

## Conventions

- **One tool per image.** Tools are not bundled, so each can be versioned,
  rebuilt and replaced on its own.
- **Image name is `tintinet/<tool>:<version>`.** The tag is the tool's version.
- **Everything is pinned:** base image by tag, tool by version or commit.
- **Images hold software only.** Data, model weights and licences are mounted
  at run time, never copied in.
- **The `work/` folder is mounted at `/work`** and is the working directory, so
  paths on the command line are relative to `work/`.
- **Run as yourself** (`-u "$(id -u):$(id -g)"`), so files written into `work/`
  belong to you and not to root.
- **Containers exchange files with the Julia code and nothing else.**
- **Python appears only in the images marked below.**
- GPU images need the NVIDIA container toolkit on the host as well as Docker.

## Environments

| Folder | For | Version | GPU | Python | Status |
|---|---|---|---|---|---|
| `dssp` | secondary structure, accessibility, angles, strand ladders | mkdssp 4.4.10 | no | no | built and tested |
| `stride` | independent secondary-structure assignment | commit 5e0db42 | no | no | built and tested |
| `mmseqs2` | sequence clustering and leakage checks | 18-8cc5c | no | no | built and tested |
| `foldseek` | structure search, to decide whether a fold is new | 10-941cd33 | no | no | built and tested |
| `julia` | reproducible runs of the package | — | yes | no | planned |
| `esm` | baseline probes, context-size runs | — | yes | yes | planned |
| `boltz` | large-predictor comparison | — | yes | yes | planned |
| `chai` | large-predictor comparison | — | yes | yes | planned |
| `alphafold3` | large-predictor comparison (weights mounted) | — | yes | yes | planned |

## Build the four that exist

```bash
docker build -t tintinet/dssp:4.4.10         dockerfiles/dssp
docker build -t tintinet/stride:5e0db42      dockerfiles/stride
docker build -t tintinet/mmseqs2:18-8cc5c    dockerfiles/mmseqs2
docker build -t tintinet/foldseek:10-941cd33 dockerfiles/foldseek
```

## What the test runs showed (17 Nov 2026)

- **dssp** reads gzipped mmCIF directly. It prints "the input file is not
  valid" on files that are fine; that is its dictionary check, and the output
  is still written.
- **dssp 4.2.2 was tried first and dropped.** On about one entry in ten it
  fails with "Duplicate Key violation ... struct_sheet_range" when writing its
  tables in mmCIF format. Version 4.4.10 ran all of those entries.
- **dssp needs `--calculate-accessibility`** to include accessibility in mmCIF
  output.
- **dssp is fast on ordinary entries** (about 0.1 s) and slow on the very
  largest (about 45 s for ribosome-sized ones).
- **stride** works on PDB-format files and refuses mmCIF ("has no
  coordinates"), as expected.
- **mmseqs2** clustered all 189,073 distinct PDB protein sequences at four
  identity levels in under a minute.
- **foldseek** reads gzipped mmCIF directly and finds each structure as its own
  best hit.

## Known wrinkles

- **STRIDE reads PDB-format files only**, not mmCIF. Step 06 of the pipeline
  writes each chain in PDB format for it.
- **DSSP and our own backbone angles differ at residues with alternate
  conformations.** On entry 1c75 DSSP's angles differ from STRIDE's and from
  ours by tens of degrees at a few such residues; ours and STRIDE's agree to
  the second decimal. The pipeline's rule for alternate conformations is
  stated at the top of `scripts/data/06_prepare_chains.jl`.
- **Starting one container per file is slow.** The pipeline starts a few
  containers and has each process a long list of files.
