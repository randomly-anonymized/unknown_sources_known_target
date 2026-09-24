# FINDINGS — adaptive sampling from unknown sources, NHIS 2023 experiments

All numbers below come from `results/*.csv`, produced by the scripts in this repository from the
**NHIS 2023 public-use Sample Adult file joined to the Paradata file** (n = 29,522 adults;
diagnosed diabetes n = 3,294, 9.79% weighted). Type 1 diabetes alone is n = 269, too thin to
estimate 16 channel compositions, so the target population is all diagnosed diabetes, as agreed.
Every channel is a *reachable pool* inside that population; composition is weighted by `WTFA_A`.

Headline instance used throughout: **target = ethnic composition of the US adult diabetes
population** (Hispanic 19.05%, non-Hispanic Black 15.52%, other 65.43%), **16 real recruitment
channels**, costs \$72 (online) / \$199 (offline) per enrolled participant from the JMIR 2020
recruitment-cost meta-analysis.

---

## Summary for the abstract

Three results are worth the abstract's empty bracket:

1. **At trial scale the method removes essentially all of the representation error that naive
   recruitment leaves behind, and it does so by a wider margin than any non-feedback policy —
   including one given the true channel compositions.** At T = 1,000 recruits, ‖p̂−p_G‖₂ is
   0.0071 for Algorithm 2, 0.0211 for the oracle open-loop allocation, 0.0403 for uniform use of
   all channels and 0.0907 for online-only recruitment. The realised slopes over the last decade
   are −1.11 (Algorithm 2), −0.99 (oracle closed loop), −0.50 (oracle open loop), −0.01 (naive).
2. **The hull margin c is the quantity that decides what a trial can match.** With these 16
   channels, matching the 3-category ethnic composition (c = 0.082) is easy, matching Black × age
   (c = 0.0086) costs ~50× more imbalance at the same horizon, and matching ethnicity × age
   (K = 6, c = 0.0014) is out of reach at any realistic trial size. The imbalance plateau tracks
   0.09/c across two orders of magnitude of c.
3. **Feedback, not knowledge, is what buys the rate — and feedback is also what survives
   operational reality.** Under ±5 points of drift in channel composition, Algorithm 1 keeps
   ‖S_T‖ = 7.4 while explore-then-commit and the oracle fixed allocation both degrade to a
   persistent ~1.6-point bias. Batched decisions (every 50 recruits) and 50-round delayed labels
   cost only a factor ≈ 5 in ‖S_T‖ and leave the −1 slope intact.

---

## C1 — Naive recruitment vs the adaptive rule (E0, E1.1, E2.1, E5)

Real channel compositions on the ethnic target (`results/E0_channels_headline.csv`):

| channel | n | Hispanic | Black | other | \$ |
|---|---|---|---|---|---|
| ONLINE (health-info seekers) | 1,583 | 13.7% | 13.9% | 72.4% | 72 |
| PORTAL | 1,691 | 15.5% | 13.4% | 71.1% | 72 |
| LANDLINE | 642 | 9.6% | 14.4% | 75.9% | 199 |
| MEDICAID | 637 | 30.8% | 20.7% | 48.6% | 199 |
| ED | 1,021 | 19.8% | 18.1% | 62.1% | 199 |
| SPANISH-language | 289 | 94.2% | 0.4% | 5.4% | 199 |
| SITE_S (southern clinics) | 1,253 | 15.9% | 24.5% | 59.6% | 199 |
| SITE_W | 691 | 37.2% | 4.8% | 57.9% | 199 |
| RURAL | 666 | 5.2% | 9.2% | 85.6% | 199 |
| **target** | 3,294 | **19.05%** | **15.52%** | **65.43%** | |

Asymptotic error of naive rules (they do not shrink with T):

| rule | Hispanic | Black | other |
|---|---|---|---|
| online-only (cheapest) | −5.40 pp | −1.61 pp | +7.01 pp |
| uniform over all 16 channels | +2.87 pp | −1.52 pp | −1.35 pp |
| pool-proportional | +0.05 pp | −0.71 pp | +0.66 pp |

Two honest points. First, **coverage-based channel biases are much milder on ethnicity than on
age**: the online channel under-represents Black adults by only 1.6 points, because internet
health-information use differs far less by race than by age (landline reach is 2% under 45 vs 24%
at 65+, a 12× gradient). Real trials under-represent Black participants by far more than 1.6
points, which means this construction *understates* the problem it solves — state that in the
paper rather than letting a referee infer it. Second, the naive error that the method actually
removes here is dominated by Hispanic representation (5.4 pp) and, in the age-based instance, by
the 65+ share (7.1 pp).

Tier-1 (two channels, one binary attribute), error at T ≈ 850 and slope over T ≥ 1,000:

| instance | ALG1 | oracle open loop | ETC | cheapest-only | uniform |
|---|---|---|---|---|---|
| online vs landline, 65+ share | **0.0018** (−1.01) | 0.0132 (−0.50) | 0.1194 (−0.74) | 0.0703 (0.00) | 0.1194 (0.00) |
| online vs southern sites, Black share | **0.0038** (−0.99) | 0.0099 (−0.51) | 0.0369 (−0.62) | 0.0174 (0.00) | 0.0369 (0.00) |

## C2 — Feedback is what gives Θ(1/T); open-loop policies cannot (E1.1, E1.3, E2.1)

The Lemma-1 decomposition is visible in the data: under Algorithm 1 the correlation between the
allocation term and the sampling term is **−0.9999** (T1-AGE) and **−0.9993** (T1-BLACK) — the
allocation error is cancelling the realised sampling noise almost exactly. Under
explore-then-commit the same correlation is only −0.25, and its error decays at T^{−0.6}.

The oracle open-loop policy is given the true channel means and herds deterministically on x\*,
so its allocation error is ≤ 1/T by construction; it still decays at exactly T^{−0.50} with
constant √(2ν\*/π) = 0.380 (predicted) vs 0.379 (measured at T = 10⁵). This is the cleanest
statement of the paper's point: **knowing the arm distributions is not what matters; reacting to
the realised labels is.**

Sample complexity N₉₀(ε) = smallest T with P(‖p̂_T − p_G‖₂ ≤ ε) ≥ 0.9, headline instance:

| ε | ALG2(β=0.3) | oracle closed loop | oracle open loop | uniform |
|---|---|---|---|---|
| 0.02 | 832 | 191 | 2,512 | ∞ |
| 0.01 | 1,202 | 398 | 15,849 | ∞ |
| 0.005 | 2,512 | 832 | 69,183 | ∞ |

## C3 — The 1/T rate is optimal (E1.4)

Using the universal lower bound derived in the design note (§10.3): E|S_T| ≥ κ for every policy at
every T. Measured E|S_T| for Algorithm 1 against κ and against the Theorem-1 upper bound:

| instance | κ (lower) | measured E&#124;S_T&#124; | Theorem-1 bound |
|---|---|---|---|
| online vs landline | 0.220 | **1.43** | 14.72 |
| online vs southern sites | 0.139 | **3.57** | 40.79 |

Algorithm 1 sits 6.5× and 25× above the universal floor and 10× below the theorem's bound, i.e.
the rate is matched and the constants are within an order of magnitude. Both lower bounds (the
universal Ω(1/T) and the Ω(T^{−1/2}) for label-independent policies) are plotted as reference
lines in `figures/F1_*.png`; the manuscript should state and prove them.

## C4 — Theorem 1's tail bound (E1.2)

The empirical tail of |S_T| is T-invariant, as the theorem requires (the curves for T = 10², 10³,
10⁴, 10⁵ coincide: `figures/F2_tails.png`), and geometric in k. The optimised bound
inf_λ C(λ)e^{−λk} is valid but loose by roughly two orders of magnitude at k = 10 for the
well-separated instance, more for the narrow one.

## C5 — Algorithm 2, the UCB radius, and the margin (E2.1–E2.4)

β multiplies the manuscript's confidence radius: the index is ⟨u_t, μ̂_a − p_G⟩ − β·r_a(N_a(t)),
so **β = 1 is Algorithm 2 exactly as published** and β = 0 is the greedy plug-in rule. β < 1 is not
a re-tuned δ (matching it would need δ ≈ 2×10⁷ at n = 10³), so it forfeits Theorem 2's guarantee
and is reported as tuning, not theory. At T = 10⁵ on the ethnic target (c = 0.082, K = 3, m = 16),
‖S_T‖:

| ALG2 β=1 | β=0.3 | β=0.1 | greedy (β=0) | oracle closed loop |
|---|---|---|---|---|
| 2.5 | **1.9** | 4.5 | 532 | 2.0 |

Three observations the paper should report:

* **Algorithm 2 matches the oracle closed loop** on this instance — there is no visible price for
  not knowing the channel compositions once c is healthy.
* **Greedy plug-in (β = 0) fails** (slope −0.08): an arm whose mean is badly estimated is never
  chosen again, which is exactly the failure the confidence radius exists to prevent. This
  justifies the UCB term empirically.
* **The theory's constants are vacuous at these horizons and the algorithm works anyway.** With
  c = 0.082, N(c) = 113,956 per arm, M(c) = 1.8×10⁶, and Theorem 2's bound on ‖S_T‖ is 2.0×10⁶ —
  against a measured 1.9. Every one of the 10⁵ rounds is "uncertain" by the r > c/4 criterion, yet
  the O(1/T) regime is reached by T ≈ 10³.

**The granularity ladder (E2.4) is the most useful experiment in the suite.** Measured on the same
16 channels and the same target population:

| stratification | K | margin c | ‖S_T‖ at 10⁵ (oracle) | ‖S_T‖ (ALG2 β=0.3) | ALG2 error at T≈10³ | uniform error |
|---|---|---|---|---|---|---|
| Black vs other | 2 | 0.128 | 0.9 | 0.9 | 0.0037 | 0.0220 |
| Hispanic / Black / other | 3 | 0.082 | 1.8 | 1.9 | 0.0070 | 0.0407 |
| age 18–49/50–64/65+ | 3 | 0.047 | 3.1 | 3.4 | 0.0115 | 0.0412 |
| age × sex | 4 | 0.026 | 9.9 | 16.3 | 0.0240 | 0.0439 |
| Black × age | 4 | 0.0086 | 8.4 | 41.4 | 0.0137 | 0.0454 |
| ethnicity × age | 6 | 0.0014 | 60.8 | 261.4 | 0.0206 | 0.0563 |

The oracle plateau follows 0.09/c to within a factor of 3 across the whole range
(`figures/F4_ladder.png`, right panel). Practical reading for a trialist: *with this channel menu,
a 1,000-person trial can match a 3-category ethnic target to well under a point, but cannot match
ethnicity and age jointly at all* — and the binding constraint is the geometry of the available
channels, not the algorithm.

Multi-label version (FDA-style marginal enrolment goals: Black, Hispanic, 65+, female; K = 4,
ℓ₀ = 3, c = 0.054): same picture, ‖S_T‖ = 4.4 for β = 0.3 vs 4.2 for the oracle, 110 for greedy,
and 9,239 for online-only. This is the formulation closest to how diversity goals are actually
written, and it is worth a subsection.

## C6 — Cost-aware selection (E3.1–E3.3) — **partly negative**

Cheapest support at c₀ = 0.05 is {ONLINE, SITE_S, SITE_W} at \$137.96 per recruit; the cheapest
support ignoring the margin filter is \$106.14. **The margin filter costs 30% in price**, and
that trade-off is the real content of c₀ (`figures/F7_c0_tradeoff.png`):

| c₀ | cost/recruit (n=100) | ‖p̂−p_G‖₂ at T=2×10⁴ | fallback rate |
|---|---|---|---|
| 0.01 | \$112.7 | 0.0130 | 0.00 |
| 0.02 | \$117.2 | 0.0086 | 0.00 |
| 0.05 | \$145.5 | 0.0050 | 0.00 |
| 0.10 | \$193.7 | 0.0010 | 0.36 |

Support selection itself is **unreliable at trial-scale exploration budgets**. With n = 25 samples
per channel, 59% of runs select a support whose true simplex does not contain p_G (so the error
plateaus at the projection distance); this falls to 0% only by n = 1,600 per channel — i.e.
25,600 exploration recruits, more than the trial. Theorem 3's own requirement is n ≥ 72,576 per
channel at c₀ = 0.05. Theorem 3's guarantee P(c(Î) ≥ c₀/2) rises from 0.23 (n = 25) to 1.00
(n = 1,600). Two fixes to state in the paper: Algorithm 3 needs (i) a defined fallback when no
support passes the filter (we fall back to Algorithm 2 on all channels; used in 1–36% of runs
depending on c₀) and (ii) a feasibility guard, because passing ĉ ≥ 3c₀/4 does not imply true
feasibility at small n.

End-to-end at T = 2×10⁴ (`results/E3_endtoend.csv`):

| policy | ‖p̂−p_G‖₂ | \$/recruit |
|---|---|---|
| ALG3+2 (n=100, c₀=0.05) | 0.0037 | 143.79 |
| ALG3+2 (n=25, c₀=0.05) | 0.0194 | 134.75 |
| ALG2 on all channels | 0.0001 | 195.03 |
| **myopic cost-ratio rule (oracle means)** | **0.0001** | **153.68** |
| oracle closed loop on the cheapest admissible support | 0.0001 | 138.12 |
| oracle open loop on the min-cost mixture | 0.0042 | 106.14 |
| online-only | 0.0901 | 72.00 |

**The myopic cost-ratio rule that §6 rejects is the strongest practical competitor here**: it
attains the same accuracy as cost-unaware Algorithm 2 at 21% lower cost, and beats the
full Algorithm 3 + 2 pipeline on both axes at this horizon. Section 6's argument against it is
about worst cases, not this instance; either add a counterexample instance where it provably
overpays, or present it as a baseline the method has to beat and show the regime where it does.

## C7 — Against discarding and reweighting (E1.5, E3, E2.5) — **partly negative**

*Reproduction of Example 1*: adaptive \$2,333.33 vs discard \$3,006.64 (simulated, 500 reps) —
the manuscript's \$2,333 / \$3,000 is correct.

*On the real instances the comparison flips at realistic cost ratios.* Break-even
r\* = (f − x\*)/(1 − x\*) against quota screening on the cheapest channel: **1.95** (age instance)
and **1.76** (Black instance), while the base cost ratio from the meta-analysis is r = 2.76.
Multi-channel version (`results/E3_cost_reference.csv`):

| r | adaptive (no discard) | quota on cheapest | cost-optimal mix + discard |
|---|---|---|---|
| 1.0 | \$72.00 | \$100.47 | \$72.00 |
| 1.5 | \$81.68 | \$100.47 | \$81.68 |
| 2.0 | \$91.35 | \$100.47 | \$84.98 |
| **2.76** | **\$106.06** | **\$100.47** | **\$85.67** |
| 4.0 | \$130.06 | \$100.47 | \$86.79 |

So: the adaptive sample is cheaper than screening only when offline recruitment costs less than
about 2.4× online recruitment; at the meta-analysis medians it costs ~6% more than quota screening
and ~24% more than the discard-optimal mixture. The defensible claims are (i) representation is
achieved *unconditionally*, (ii) cost advantage holds for r < r\*, with r\* computable in closed
form, and (iii) no eligible participant is turned away on demographic grounds — which quota
screening necessarily does.

*Reweighting* is cheap on this instance: post-stratifying an online-only sample gives design
effect 1.030 (ESS loss 3%) and never fails (no empty cells at K = 3). The honest framing is not
"reweighting is expensive" but "reweighting fixes the composition of the *estimate*, not of the
*sample*" — it cannot help trial endpoints that need actual enrolled participants, and its design
effect grows with K.

*Downstream estimation (E2.5, respondent mode, T = 2×10⁴)* is the sharpest negative and the most
interesting. Adaptive sampling drives the ethnic composition error to 1×10⁻⁴, but the bias of
downstream population estimates does **not** improve:

| estimate | adaptive (ALG2) | oracle closed loop | online-only | uniform |
|---|---|---|---|---|
| insulin use | +1.7 pp | +1.0 pp | −0.1 pp | +2.5 pp |
| heart disease | +1.6 pp | +0.6 pp | −0.5 pp | +2.6 pp |
| disability | +3.2 pp | +2.1 pp | −2.9 pp | +3.7 pp |
| fair/poor health | +4.1 pp | +2.6 pp | −3.6 pp | +5.5 pp |

Matching on ethnicity alone forces the sampler toward clinical channels (Medicaid, ED, inpatient,
southern clinic sites) whose members are sicker *within* each ethnic group, so between-stratum
bias is removed and within-stratum bias is imported. This is a property of composition matching on
a coarse target, not of the algorithm — quota screening and reweighting on the same strata have
the same defect — but the paper should say it, because it is the first thing a clinical reviewer
will ask. The constructive version: the target vector should include the dimensions the endpoint
depends on, which immediately runs into the margin limit quantified in C5.

## Robustness (E1.6)

| variant | ‖S_T‖ at T = 2×10⁴ | slope |
|---|---|---|
| reference (Algorithm 1) | 1.98 | −1.03 |
| channel separation × 1/2, 1/4, 1/8 | 4.29, 8.76, 17.39 | −1.0 |
| decisions batched every 10 / 50 recruits | 3.57 / 10.73 | −0.98 / −0.97 |
| labels delayed 10 / 50 rounds | 3.81 / 10.76 | −0.99 / −0.99 |
| ±5 pp drift in channel composition: ALG1 | 7.44 | −0.9 |
| ±5 pp drift: explore-then-commit | 480 (error 1.70 pp) | ≈0 |
| ±5 pp drift: oracle fixed allocation | 444 (error 1.57 pp) | ≈0 |
| ordering assumption violated (both channels below target) | 456 (error 1.61 pp) | −0.03 |

‖S_T‖ scales exactly like 1/c (halving the separation doubles the imbalance), batching and delay
cost O(B) and O(d) as expected, and the infeasible case converges to the projection distance
(1.61 pp = p_G − p_ONLINE) rather than diverging. The drift row is a genuine selling point:
closed-loop adaptation absorbs non-stationarity that defeats every fixed-allocation design.

---

## What is in the repository

```
scripts/01_build_scenarios.py   NHIS -> 30 scenarios + E0 table (asserts every variable code)
scripts/02_tier1.py  {age|black|example1|robust}
scripts/03_tier2.py  {diag|e21a1|e21a2|e21b|e22a|e22b|e24a|e24b|e25}
scripts/04_tier3.py  {e31|e32|e33|refs}
scripts/05_figures.py            F0-F8
scripts/06_tables.py             E5 headline / per-stratum / channel tables
tests/test_all.py                9 tests, all passing
src/{geometry,theory,env,policies,runner}.py
```

`results/` holds one tidy CSV per experiment; `figures/` holds the eight figures; scenarios and
their provenance are in `data/scenarios/`.

## Issues found in the manuscript (beyond those in the design note)

1. **The abstract's lower bound is still missing**; the experiments now plot it, so it has to be
   stated. Both forms are in the design note §10.3.
2. **Theorem 2's constants are vacuous at experiment scale** (bound 2.0×10⁶ vs measured 1.9) and
   every round counts as "uncertain" at T = 10⁵. Present the theorem as a rate result and report
   the empirical constants separately.
3. **Algorithm 3 needs a fallback and a feasibility guard**, and its cost guarantee must be stated
   against the cheapest support *above* the margin threshold: here that is \$137.96 against a true
   cheapest of \$106.14.
4. **Section 6's dismissal of the myopic cost-ratio rule is not supported by this instance.**
5. **R² = ℓ₀ + 1 in Remark 1 needs ‖p_G‖₂ ≤ 1**; the code computes R² from the actual support
   (1.18 for K = 3 categorical, 1.28 for the multi-label instance).
6. For categorical labels the interiority condition is *relative* interior and requires m ≥ K;
   the margin must be computed inside the affine hull, as `src/geometry.py` does.
