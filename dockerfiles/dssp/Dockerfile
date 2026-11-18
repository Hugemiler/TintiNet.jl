# dssp — secondary structure, accessibility, backbone angles and strand
# partners, computed from a structure file.
#
# Tool      mkdssp, from the Ubuntu 24.04 package `dssp` (version 4.2.2).
#           Upstream is at 4.5.x; building that from source is the route to a
#           newer version if we ever need one.
# Input     mmCIF or PDB file
# Output    annotated mmCIF, or the classic DSSP text format
#
# BUILD     docker build -t tintinet/dssp:4.2.2 dockerfiles/dssp
#
# RUN       docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/work:/work" \
#               tintinet/dssp:4.2.2 \
#               --output-format dssp raw/mmCIF/c7/1c75.cif.gz tools/dssp/1c75.dssp
#
# STATUS    Built and run on real files, 17 Nov 2026.

FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends "dssp=4.2.2-2*" \
 && rm -rf /var/lib/apt/lists/*

# Paths given on the command line are relative to the mounted work folder.
WORKDIR /work

ENTRYPOINT ["mkdssp"]
