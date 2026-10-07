#!/usr/bin/env bash
# Full reproduction. Each stage is a separate process so the suite can be resumed after any stage.
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/00_fetch_data.py                      # skips files already in data/raw
python3 scripts/01_build_scenarios.py                 # ~1 min
for s in age black example1 robust;            do python3 scripts/02_tier1.py $s; done   # ~12 min
for s in diag e21a1 e21a2 e21b e22a e22b e24a e24b e25; do python3 scripts/03_tier2.py $s; done  # ~25 min
for s in refs e31 e33;                         do python3 scripts/04_tier3.py $s; done
for k in alg3 alg3n800 alg3n400 alg2 myopic oracle oracleall olmincost online uniform; do python3 scripts/04_tier3.py e32 $k; done
python3 scripts/04_tier3.py e32merge
for k in alg3 alg3n800 alg3n400 alg2 myopic oracle; do python3 scripts/04_tier3.py e35 $k; done
python3 scripts/04_tier3.py e35merge
python3 scripts/06_tables.py
python3 scripts/05_figures.py                         # restyle via configs/style.json alone
python3 scripts/07_paper_figures.py                   # manuscript figures, styled by configs/style-paper.json
python3 tests/test_all.py
