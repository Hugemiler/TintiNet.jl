# foldseek — structure similarity search. Used to decide whether a test
# protein's fold is new: two proteins can share a fold with no detectable
# sequence similarity, so sequence clustering alone is not enough.
#
# Tool      Foldseek release 10-941cd33, the authors' prebuilt Linux binary.
# Input     structure files (mmCIF or PDB)
# Output    search hits as tab-separated text
#
# BUILD     docker build -t tintinet/foldseek:10-941cd33 dockerfiles/foldseek
#
# RUN       docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/work:/work" \
#               tintinet/foldseek:10-941cd33 \
#               easy-search tools/structures/test tools/structures/train \
#               tools/clusters/foldseek_hits.tsv tools/clusters/tmp
#
# STATUS    Built and run on real files, 17 Nov 2026.

FROM ubuntu:24.04

ARG FOLDSEEK_RELEASE=10-941cd33

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL "https://github.com/steineggerlab/foldseek/releases/download/${FOLDSEEK_RELEASE}/foldseek-linux-avx2.tar.gz" \
      | tar -xz -C /opt \
 && ln -s /opt/foldseek/bin/foldseek /usr/local/bin/foldseek \
 && apt-get purge -y curl && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

ENTRYPOINT ["foldseek"]
