# mmseqs2 — sequence clustering and search. Used to group similar chains and
# to check that no test protein is a close relative of a training protein.
#
# Tool      MMseqs2 release 18-8cc5c, the authors' prebuilt Linux binary.
#           The AVX2 build runs on any x86-64 processor from roughly 2013 on.
# Input     FASTA files
# Output    cluster tables and search hits as tab-separated text
#
# BUILD     docker build -t tintinet/mmseqs2:18-8cc5c dockerfiles/mmseqs2
#
# RUN       docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/work:/work" \
#               tintinet/mmseqs2:18-8cc5c \
#               easy-cluster index/sequences.fasta tools/clusters/seq30 tools/clusters/tmp \
#               --min-seq-id 0.3 -c 0.8
#
# STATUS    Built and run on real files, 17 Nov 2026.

FROM ubuntu:24.04

ARG MMSEQS_RELEASE=18-8cc5c

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL "https://github.com/soedinglab/MMseqs2/releases/download/${MMSEQS_RELEASE}/mmseqs-linux-avx2.tar.gz" \
      | tar -xz -C /opt \
 && ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs \
 && apt-get purge -y curl && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

ENTRYPOINT ["mmseqs"]
