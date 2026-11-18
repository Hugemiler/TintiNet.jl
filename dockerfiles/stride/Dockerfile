# stride — a second, independent secondary-structure assignment, used to
# measure how much the labels themselves can be trusted.
#
# Tool      STRIDE, built from the authors' repository (github.com/heiniglab/stride)
#           at a fixed commit. STRIDE has no version numbers, so the commit is
#           the version.
# Input     PDB-format file ONLY. STRIDE does not read mmCIF. The pipeline has
#           to write each chain out in PDB format before calling it.
# Output    STRIDE text report on standard output
#
# BUILD     docker build -t tintinet/stride:5e0db42 dockerfiles/stride
#
# RUN       docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/work:/work" \
#               tintinet/stride:5e0db42 tools/pdb/1c75_A.pdb > work/tools/stride/1c75_A.stride
#
# STATUS    Built and run on real files, 17 Nov 2026.

# ---- stage 1: compile ------------------------------------------------------
FROM ubuntu:24.04 AS build

ARG STRIDE_COMMIT=5e0db42cef55b2d8f04e410476efcee106fbedfd

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc make libc6-dev ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/heiniglab/stride/archive/${STRIDE_COMMIT}.tar.gz" \
      | tar -xz -C /tmp \
 && make -C "/tmp/stride-${STRIDE_COMMIT}" stride \
 && cp "/tmp/stride-${STRIDE_COMMIT}/stride" /usr/local/bin/stride

# ---- stage 2: just the program ---------------------------------------------
FROM ubuntu:24.04

COPY --from=build /usr/local/bin/stride /usr/local/bin/stride

WORKDIR /work

ENTRYPOINT ["stride"]
