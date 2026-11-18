# dssp — secondary structure, accessibility, backbone angles and strand
# ladders, computed from a structure file.
#
# Tool      mkdssp 4.4.10, from the Debian 13 ("trixie") package `dssp`.
#           Not 4.2.2 (the Ubuntu 24.04 package): that version fails on about
#           one entry in ten, with "Duplicate Key violation ... struct_sheet_range",
#           when writing its tables in mmCIF format. 4.4.10 does not.
# Input     mmCIF file, plain or gzipped
# Output    the same mmCIF with DSSP's tables added (_dssp_struct_summary,
#           _dssp_struct_ladder, ...), or the classic DSSP text format
#
# BUILD     docker build -t tintinet/dssp:4.4.10 dockerfiles/dssp
#
# RUN       docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/work:/work" \
#               tintinet/dssp:4.4.10 \
#               --output-format mmcif --calculate-accessibility \
#               raw/mmCIF/c7/1c75.cif.gz tools/dssp/1c75.cif
#
#           --calculate-accessibility is needed: in mmCIF output this version
#           leaves accessibility out unless asked.
#
# SPEED     About 0.1 s for an ordinary entry; about 45 s for the very largest
#           (ribosome-sized) ones.
#
# STATUS    Built and run on real files, 17 Nov 2026.

FROM debian:trixie-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends "dssp=4.4.10-1*" \
 && rm -rf /var/lib/apt/lists/*

# Paths given on the command line are relative to the mounted work folder.
WORKDIR /work

ENTRYPOINT ["mkdssp"]
