# Adaptive sampling from unknown sources — experiments on NHIS 2023

Implementation of the experimental design for *"Sampling from Unknown Sources to Approximate a
Known Target"*. Everything is driven by real data: the US adult **diagnosed-diabetes population**
(NHIS 2023) is the target, and sixteen **real recruitment channels** are reachable subpopulations
of it defined by NHIS variables.

## Reproducing

```bash
python3 scripts/01_build_scenarios.py                   # NHIS -> data/scenarios/*.json + results/E0_scenarios.csv
python3 scripts/02_tier1.py age|black|example1|robust   # sec. 2-3   (E1)
python3 scripts/03_tier2.py diag|e21a1|e21a2|e21b|e22a|e22b|e24a|e24b|e25   # sec. 5 (E2)
python3 scripts/04_tier3.py refs|e31|e32|e33|e34        # sec. 6     (E3)
python3 scripts/05_figures.py                           # figures/F0-F8
python3 scripts/06_tables.py                            # results/E5_*.csv
python3 scripts/07_paper_figures.py                     # figures/paper/*.pdf (manuscript figures)
python3 tests/test_all.py                               # 9 correctness tests
```

Inputs expected at `/mnt/user-data/uploads/adult23.csv` and `/mnt/user-data/uploads/paradata23.csv`
(NHIS 2023 public use, joined on `HHX`). Stages are split so each runs in a few minutes on one
core; results accumulate in `results/` as tidy CSV. Total runtime for the full suite is about 40
minutes; peak memory under 300 MB. Every run is seeded and deterministic.

## Layout

| path | contents |
|---|---|
| `src/geometry.py` | affine-hull margins, min-cost / ℓ∞ allocations, support enumeration |
| `src/theory.py` | Theorem 1/2/3 constants, N(c), both lower bounds |
| `src/env.py` | arms as distributions over label patterns; stratum, multi-label, respondent and drifting modes |
| `src/policies.py` | Algorithms 1/2/3, oracles, ETC, naive rules, myopic cost rule, batching/delay wrappers |
| `src/runner.py` | batched simulation loop, quota-screening baseline, post-stratification stats |
| `data/scenarios/` | 30 scenario JSONs (channel means, target, costs, geometry) + respondent pools |
| `results/` | one tidy CSV per experiment |
| `figures/` | F0 channel compositions, F1 tier-1 rates, F2 tails, F3 tier-2 rates, F4 ladder, F5 cost, F6 support selection, F7 c₀ trade-off, F8 robustness |
| `figures/paper/` | vector PDFs used by `main.tex` (Experiments section and its appendix), made by `scripts/07_paper_figures.py` from `results/*.csv` and styled by `configs/style-paper.json` |
| `FINDINGS.md` | results organised by claim, including the negative ones |

## Data construction in one paragraph

Target population: `DIBEV_A == 1` (diagnosed diabetes), weighted by `WTFA_A`. Channels: online
health-information seekers (`HITLOOK_A`), patient portal (`HITCOMM_A`/`HITTEST_A`), landline and
cell (`AD_PCLASS` from the paradata file), Medicare, Medicaid, VA, emergency department,
inpatient, telehealth, Spanish-language, four regional clinic "sites" (`USUALPL_A`/`LASTDR_A` ×
`REGION`) and rural (`URBRRL`). A channel's arm distribution μ_a is the weighted stratum
composition of its reachable pool; pools with fewer than 150 respondents are dropped. This is a
coverage model — it ignores differential response propensity within a pool, so it *understates*
real channel bias. Costs are the JMIR 2020 meta-analysis medians (\$72 online, \$199 offline per
enrolled participant), swept over the online/offline ratio wherever cost matters.

## Changing the figures

All styling — colours, line widths, dash patterns, fonts, dpi — lives in `configs/style.json`.
Edit that file and re-run `python3 scripts/05_figures.py`; no code changes are needed, and nothing
is re-simulated (the figure script reads `results/*.csv` only). To keep several looks around, copy
the file and point at it: `STYLE_FILE=configs/style-print.json python3 scripts/05_figures.py`.
Keys are policy names exactly as they appear in the `policy` column of the result CSVs; anything
not listed falls back to `default_policy`. `strata_palette` colours the stacked composition bars,
`ladder_palette` the granularity-ladder curves, `reference_lines.color` the theory curves.

## Parameters you can turn

| symbol | where | meaning |
|---|---|---|
| `beta` | `Alg2UCB(beta=…)` | multiplier on the manuscript's confidence radius r_a(n). **β = 1 is Algorithm 2 exactly as written**; β < 1 shrinks the exploration bonus; β = 0 is the greedy plug-in rule. See "What β is" below. |
| `delta` | `Alg2UCB(delta=…)` | confidence level inside r_a(n), default 0.05 (the manuscript's δ) |
| `c0` | `Alg3ThenAlg2(c0=…)` | margin floor in Algorithm 3's support filter (ĉ(I) ≥ 3c₀/4) |
| `n` | `Alg3ThenAlg2(n=…)` | exploration samples per channel before support selection; these samples are retained in the final sample |
| `eps` | `ETC(eps=…)` | fraction of the horizon the explore-then-commit baseline spends exploring |
| `T`, `R` | top of each script | horizon and number of replications |
| `MIN_POOL_N` | `scripts/01_build_scenarios.py` | minimum unweighted respondents for a channel to be used (default 150) |
| `COST_ONLINE`, `COST_OFFLINE` | `scripts/01_build_scenarios.py` | per-recruit costs; the cost ratio is swept separately in `scripts/04_tier3.py` |

### What β is

The manuscript's Algorithm 2 picks `argmin_a ⟨u_t, μ̂_a − p_G⟩ − r_a(N_a(t))` with
`r_a(n) = √( (K/2n)·log(π²mKn²/(3δ)) )`. β multiplies that second term: the code uses
`⟨u_t, μ̂_a − p_G⟩ − β·r_a(N_a(t))`. **β is not in the manuscript** — it is an ablation knob added
here, and β = 1 reproduces the published algorithm exactly.

β is *not* equivalent to re-tuning δ. Since `r ∝ √log(A/δ)` with `A = π²mKn²/3`, matching
`β·r(n;δ)` would need `δ' = A^{1−β²}·δ^{β²}`, which for K = 3, m = 16, n = 10³, δ = 0.05 gives
δ' ≈ 2×10⁷ for β = 0.3 — no valid confidence level. So β < 1 deliberately uses a bonus smaller
than any that Theorem 2's proof admits: the confidence event the proof relies on is no longer
guaranteed, and the guarantee is voided. It is reported as an empirical tuning result, not as a
variant with theory behind it. β = 0 removes the term entirely, leaving the certainty-equivalent
("greedy plug-in") version of the oracle rule of §5.

Measured effect on the headline instance at T = 10⁵ (‖S_T‖): β = 1 → 2.5, β = 0.3 → 1.9,
β = 0.1 → 4.5, β = 0 → 532 (greedy stalls, slope −0.08, because a channel with a badly estimated
mean is never chosen again). The useful reading for the paper: the radius is doing real work — the
greedy version fails — but the theoretical constant is conservative, and the loss from using it is
small once the hull margin is healthy (2.5 vs 1.9) and larger when it is not.
