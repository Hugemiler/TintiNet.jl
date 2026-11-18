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
| `dssp` | secondary structure, accessibility, angles, strand partners | mkdssp 4.2.2 | no | no | built and tested |
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
docker build -t tintinet/dssp:4.2.2          dockerfiles/dssp
docker build -t tintinet/stride:5e0db42      dockerfiles/stride
docker build -t tintinet/mmseqs2:18-8cc5c    dockerfiles/mmseqs2
docker build -t tintinet/foldseek:10-941cd33 dockerfiles/foldseek
```

## What the first test runs showed (17 Nov 2026)

- **dssp** reads gzipped mmCIF directly. It prints "the input file is not
  valid" on files that are fine; that is its dictionary check, and the output
  is still written.
- **dssp has two output formats.** The classic text format includes the strand
  partner columns. The mmCIF format only includes them, and the per-residue
  summary, when run with `--write-experimental`. The classic format has fixed
  column widths that cannot hold very large structures, so the pipeline should
  use the mmCIF format with that flag.
- **stride** works on PDB-format files and refuses mmCIF ("has no
  coordinates"), as expected.
- **mmseqs2** clustered 3,000 sequences at 30% identity into 1,705 clusters.
- **foldseek** reads gzipped mmCIF directly and finds each structure as its own
  best hit.

## Known wrinkles

- **STRIDE reads PDB-format files only**, not mmCIF. The pipeline has to write
  each chain in PDB format first.
- **mkdssp 4.2.2 is the Ubuntu-packaged version**, not the newest upstream.
- **DSSP and STRIDE can report different backbone angles for the same
  residue.** On entry 1c75, which has alternate conformations, they differed by
  tens of degrees at some residues. The two tools were given different file
  formats and handle alternate conformations differently. This is why the
  pipeline also computes angles itself, with an explicit rule for which
  conformation to use.
