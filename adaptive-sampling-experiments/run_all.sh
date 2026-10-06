#!/usr/bin/env bash
# Full reproduction. Each stage is a separate process so the suite can be resumed after any stage.
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/00_fetch_data.py                      # skips files already in data/raw
python3 scripts/01_build_scenarios.py                 # ~1 min
for s in age black example1 robust;            do python3 scripts/02_tier1.py $s; done   # ~12 min
for s in diag e21a1 e21a2 e21b e22a e22b e24a e24b e25; do python3 scripts/03_tier2.py $s; done  # ~25 min
for s in refs e31 e32 e33 e34;                 do python3 scripts/04_tier3.py $s; done   # ~13 min
for k in alg3 alg3n200 alg3n100 alg2 myopic myopicucb oracle; do python3 scripts/04_tier3.py e35 $k; done
python3 scripts/04_tier3.py e35merge
python3 scripts/06_tables.py
python3 scripts/05_figures.py                         # restyle via configs/style.json alone
python3 scripts/07_paper_figures.py                   # manuscript figures, styled by configs/style-paper.json
python3 tests/test_all.py
