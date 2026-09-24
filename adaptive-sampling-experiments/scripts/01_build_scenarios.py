"""Data layer: build real-data scenarios (arm means, target, costs, geometry) from NHIS 2023.

Inputs : adult23.csv (Sample Adult) + paradata23.csv (phone classification), joined on HHX.
Outputs: data/scenarios/*.json, data/scenarios/pools_<target>.npz, results/E0_scenarios.csv
"""
import json, os, sys, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.geometry import margin, feasible_allocation, closest_point_in_hull
from src.theory import N_of_c, lower_bound_kappa, thm2_constants
from src.runner import lp_discard_cost

def _data_dir():
    """NHIS CSVs are looked up in $NHIS_DIR, then ./data/raw, then /mnt/user-data/uploads."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in [os.environ.get("NHIS_DIR"), os.path.join(here, "data", "raw"), "/mnt/user-data/uploads"]:
        if d and os.path.exists(os.path.join(d, "adult23.csv")) and os.path.exists(os.path.join(d, "paradata23.csv")):
            return d
    raise SystemExit("adult23.csv / paradata23.csv not found. Run scripts/00_fetch_data.py "
                     "or set NHIS_DIR to the folder holding them.")


UP = _data_dir()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scenarios")
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
COST_ONLINE, COST_OFFLINE = 72.0, 199.0       # JMIR 2020 meta-analysis medians (per enrolled participant)
MIN_POOL_N = 150

ADULT_COLS = ['HHX','WTFA_A','AGEP_A','AGE65','SEX_A','HISPALLP_A','EDUCP_A','RATCAT_A','REGION','URBRRL',
              'INTV_QRT','DIBEV_A','DIBTYPE_A','DIBINS_A','DIBPILL_A','HYPEV_A','CHDEV_A','BMICAT_A',
              'DISAB3_A','PHSTAT_A','ACCSSINT_A','HITLOOK_A','HITCOMM_A','HITTEST_A','VIRAPP12M_A',
              'MEDICARE_A','MEDICAID_A','PRIVATE_A','MILITARY_A','VAHOSP_A','NOTCOV_A','USUALPL_A',
              'LASTDR_A','EMERG12MTC_A','HOSPONGT_A','LANGSPECR_A']

EXPECTED_CODES = {           # asserted before any recode (hard failure on mismatch)
    'DIBEV_A': {1, 2, 7, 8, 9}, 'SEX_A': {1, 2, 7, 9}, 'HISPALLP_A': set(range(1, 8)) | {97, 98, 99},
    'MEDICARE_A': {1, 2, 3, 7, 8, 9}, 'MEDICAID_A': {1, 2, 3, 7, 8, 9}, 'USUALPL_A': {1, 2, 3, 7, 8, 9},
    'HITLOOK_A': {1, 2, 7, 8, 9}, 'REGION': {1, 2, 3, 4}, 'EMERG12MTC_A': {0, 1, 2, 3, 4, 7, 8, 9},
}
AD_PCLASS_CODES = {1: 'wireless only', 2: 'wireless mostly', 3: 'dual users',
                   4: 'landline mostly', 5: 'landline only', 6: 'phoneless', 7: 'unknown'}


def load():
    a = pd.read_csv(f"{UP}/adult23.csv", usecols=ADULT_COLS, low_memory=False)
    p = pd.read_csv(f"{UP}/paradata23.csv", usecols=['HHX', 'AD_PCLASS', 'SRVY_YR'], low_memory=False)
    for v, codes in EXPECTED_CODES.items():
        seen = set(pd.unique(a[v].dropna()).astype(int).tolist())
        assert seen <= codes, f"unexpected codes in {v}: {sorted(seen - codes)}"
    assert set(pd.unique(p.AD_PCLASS.dropna()).astype(int)) <= set(AD_PCLASS_CODES), "AD_PCLASS codes changed"
    assert (p.SRVY_YR == 2023).all() and a.HHX.is_unique and p.HHX.is_unique
    d = a.merge(p.drop(columns=['SRVY_YR']), on='HHX', how='left', validate='one_to_one')
    age = d.AGEP_A.where(d.AGEP_A <= 85)                      # 97/98/99 = refused / not ascertained / DK
    d['old'] = np.where(age.notna(), (age >= 65).astype(float),
                        np.where(d.AGE65.eq(2), 1.0, np.where(d.AGE65.eq(1), 0.0, np.nan)))
    d['age'] = age
    d['eth'] = np.where(d.HISPALLP_A == 1, 'H', np.where(d.HISPALLP_A == 3, 'B', 'O'))
    d['female'] = (d.SEX_A == 2).astype(float)
    d['w'] = d.WTFA_A.astype(float)
    return d


def arm_masks(d):
    clinic = d.USUALPL_A.eq(1) & d.LASTDR_A.eq(1)
    return {   # channel -> (reachable-pool mask, unit cost)
        'ONLINE':     (d.HITLOOK_A.eq(1), COST_ONLINE),
        'PORTAL':     (d.HITCOMM_A.eq(1) | d.HITTEST_A.eq(1), COST_ONLINE),
        'LANDLINE':   (d.AD_PCLASS.isin([3, 4, 5]), COST_OFFLINE),
        'CELL':       (d.AD_PCLASS.isin([1, 2, 3]), COST_OFFLINE),
        'MEDICARE':   (d.MEDICARE_A.isin([1, 2]), COST_OFFLINE),
        'MEDICAID':   (d.MEDICAID_A.isin([1, 2]), COST_OFFLINE),
        'VA':         (d.VAHOSP_A.eq(1), COST_OFFLINE),
        'ED':         (d.EMERG12MTC_A.isin([1, 2, 3, 4]), COST_OFFLINE),
        'INPATIENT':  (d.HOSPONGT_A.eq(1), COST_OFFLINE),
        'TELEHEALTH': (d.VIRAPP12M_A.eq(1), COST_OFFLINE),
        'SPANISH':    (d.LANGSPECR_A.eq(1), COST_OFFLINE),
        'SITE_NE':    (clinic & d.REGION.eq(1), COST_OFFLINE),
        'SITE_MW':    (clinic & d.REGION.eq(2), COST_OFFLINE),
        'SITE_S':     (clinic & d.REGION.eq(3), COST_OFFLINE),
        'SITE_W':     (clinic & d.REGION.eq(4), COST_OFFLINE),
        'RURAL':      (d.URBRRL.eq(4), COST_OFFLINE),
    }


TARGETS = {
    'S-DIAB':    lambda d: d.DIBEV_A.eq(1),
    'S-DIAB-T2': lambda d: d.DIBEV_A.eq(1) & d.DIBTYPE_A.eq(2),
    'S-DIAB-T1': lambda d: d.DIBEV_A.eq(1) & d.DIBTYPE_A.eq(1),
    'S-HTN':     lambda d: d.HYPEV_A.eq(1),
    'S-ALL':     lambda d: pd.Series(True, index=d.index),
}

LADDERS = {   # id -> (labels, row -> label)
    'L1_K2_black':   (['Black', 'other'],            lambda g: np.where(g.eth == 'B', 'Black', 'other')),
    'L2_K3_eth':     (['Hisp', 'Black', 'other'],    lambda g: np.where(g.eth == 'H', 'Hisp',
                                                       np.where(g.eth == 'B', 'Black', 'other'))),
    'L3_K3_age':     (['18-49', '50-64', '65+'],     lambda g: np.where(g.age < 50, '18-49',
                                                       np.where(g.old == 1, '65+', '50-64'))),
    'L4_K4_blackage':(['B<65', 'B65+', 'X<65', 'X65+'],
                      lambda g: np.where(g.eth == 'B', 'B', 'X') + np.where(g.old == 1, '65+', '<65')),
    'L5_K4_agesex':  (['M<65', 'F<65', 'M65+', 'F65+'],
                      lambda g: np.where(g.female == 1, 'F', 'M') + np.where(g.old == 1, '65+', '<65')),
    'L6_K6_ethage':  (['H<65', 'H65+', 'B<65', 'B65+', 'O<65', 'O65+'],
                      lambda g: g.eth.values + np.where(g.old == 1, '65+', '<65')),
}
# order of the composed labels above must match; fix by construction:
LADDERS['L4_K4_blackage'] = (['B<65', 'B65+', 'X<65', 'X65+'],
                             lambda g: pd.Series(np.where(g.eth == 'B', 'B', 'X'), index=g.index) +
                                       pd.Series(np.where(g.old == 1, '65+', '<65'), index=g.index))
LADDERS['L5_K4_agesex'] = (['M<65', 'M65+', 'F<65', 'F65+'],
                           lambda g: pd.Series(np.where(g.female == 1, 'F', 'M'), index=g.index) +
                                     pd.Series(np.where(g.old == 1, '65+', '<65'), index=g.index))
LADDERS['L6_K6_ethage'] = (['H<65', 'H65+', 'B<65', 'B65+', 'O<65', 'O65+'],
                           lambda g: pd.Series(g.eth.values, index=g.index) +
                                     pd.Series(np.where(g.old == 1, '65+', '<65'), index=g.index))

MULTILABEL = (['Black', 'Hisp', 'age65+', 'female'],
              lambda g: np.stack([(g.eth == 'B').astype(float), (g.eth == 'H').astype(float),
                                  (g.old == 1).astype(float), g.female.values], axis=1))

OUTCOMES = {'insulin': lambda g: g.DIBINS_A.eq(1).astype(float),
            'hypertension': lambda g: g.HYPEV_A.eq(1).astype(float),
            'heart_disease': lambda g: g.CHDEV_A.eq(1).astype(float),
            'obese': lambda g: g.BMICAT_A.eq(4).astype(float),
            'disability': lambda g: g.DISAB3_A.eq(1).astype(float),
            'fair_poor_health': lambda g: g.PHSTAT_A.isin([4, 5]).astype(float)}


def composition(sub, labels, labfn):
    lab = np.asarray(labfn(sub))
    w = sub.w.values
    v = np.array([w[lab == l].sum() for l in labels])
    return v / v.sum(), len(sub)


def build_scenario(sid, sub, arms, labels, labfn, arm_names=None):
    p_G, n_tot = composition(sub, labels, labfn)
    mus, names, ns, costs, pools = [], [], [], [], []
    for a, (mask, cost) in arms.items():
        if arm_names is not None and a not in arm_names:
            continue
        s = sub[mask.reindex(sub.index).fillna(False)]
        if len(s) < MIN_POOL_N:
            continue
        mu, n = composition(s, labels, labfn)
        mus.append(mu); names.append(a); ns.append(n); costs.append(cost)
        pools.append(s.w.sum() / sub.w.sum())
    mu = np.array(mus); cost = np.array(costs); K = len(labels)
    c = margin(mu, p_G)
    x_cost, C_min = feasible_allocation(mu, p_G, cost)
    x_any, _ = feasible_allocation(mu, p_G)
    dist_out, x_proj = closest_point_in_hull(mu, p_G)
    cheapest = int(np.argmin(cost))
    naive = {'cheapest_' + names[cheapest]: float(np.abs(mu[cheapest] - p_G).max()),
             'uniform': float(np.abs(mu.mean(0) - p_G).max()),
             'poolprop': float(np.abs((np.array(pools) / np.sum(pools)) @ mu - p_G).max())}
    kappa = lower_bound_kappa(mu)
    R2 = float(np.max(np.sum((np.eye(K) - p_G) ** 2, axis=1)))
    thm2 = thm2_constants(c, R2, K, len(names)) if c > 0 else {'valid': False}
    lpd, _ = lp_discard_cost(mu, p_G, cost)
    quota_factor = float(np.max(p_G / np.maximum(mu[cheapest], 1e-12)))
    return dict(id=sid, labels=labels, K=K, m=len(names), arms=names, mu=mu.tolist(), p_G=p_G.tolist(),
                cost=cost.tolist(), n_target=int(n_tot), n_arm=ns, pool_share=pools,
                margin_c=float(c), feasible=bool(c > 0), dist_outside=float(dist_out),
                x_star_mincost=None if x_cost is None else x_cost.tolist(),
                x_star_any=None if x_any is None else x_any.tolist(),
                C_min=float(C_min), cost_lp_discard=float(lpd),
                cost_quota_cheapest=float(quota_factor * cost[cheapest]),
                naive_linf=naive, kappa_lb=float(kappa), R2=R2,
                N_c=(None if not thm2.get('valid') else float(thm2['N_c'])),
                bound_S_thm2=(None if not thm2.get('valid') else float(thm2['bound_S'])),
                t_star_approx=(None if c <= 0 else float(0.2 / c ** 2)))


def main():
    d = load()
    os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
    arms = arm_masks(d)
    rows, scenarios = [], {}
    for tname, tfn in TARGETS.items():
        sub = d[tfn(d) & d.old.notna()].copy()
        if len(sub) < 300:
            print(f"[skip] {tname}: n={len(sub)} too small for arm estimation"); continue
        for lid, (labels, labfn) in LADDERS.items():
            sid = f"{tname}__{lid}"
            sc = build_scenario(sid, sub, arms, labels, labfn)
            scenarios[sid] = sc
            rows.append(dict(scenario=sid, target=tname, ladder=lid, K=sc['K'], m=sc['m'],
                             n_target=sc['n_target'], margin_c=sc['margin_c'],
                             t_star=sc['t_star_approx'], C_min=sc['C_min'],
                             cost_lp_discard=sc['cost_lp_discard'], cost_quota=sc['cost_quota_cheapest'],
                             naive_cheapest=list(sc['naive_linf'].values())[0],
                             naive_uniform=sc['naive_linf']['uniform'],
                             naive_poolprop=sc['naive_linf']['poolprop'],
                             kappa=sc['kappa_lb'], N_c=sc['N_c']))
        # multi-label scenario
        labels, labfn = MULTILABEL
        sid = f"{tname}__ML_K4_marginals"
        p_G = np.average(labfn(sub), axis=0, weights=sub.w.values)
        mus, names, ns, costs, pools = [], [], [], [], []
        for a, (mask, cost) in arms.items():
            s = sub[mask.reindex(sub.index).fillna(False)]
            if len(s) < MIN_POOL_N:
                continue
            mus.append(np.average(labfn(s), axis=0, weights=s.w.values))
            names.append(a); ns.append(len(s)); costs.append(cost); pools.append(s.w.sum() / sub.w.sum())
        mu = np.array(mus); cost = np.array(costs)
        c = margin(mu, p_G, categorical=False)
        x_cost, C_min = feasible_allocation(mu, p_G, cost)
        cheapest = int(np.argmin(cost))
        sc = dict(id=sid, labels=labels, K=4, m=len(names), arms=names, mu=mu.tolist(), p_G=p_G.tolist(),
                  cost=cost.tolist(), n_target=int(len(sub)), n_arm=ns, pool_share=pools,
                  margin_c=float(c), feasible=bool(c > 0), categorical=False,
                  x_star_mincost=None if x_cost is None else x_cost.tolist(), C_min=float(C_min),
                  naive_linf={'cheapest_' + names[cheapest]: float(np.abs(mu[cheapest] - p_G).max()),
                              'uniform': float(np.abs(mu.mean(0) - p_G).max()),
                              'poolprop': float(np.abs((np.array(pools)/np.sum(pools)) @ mu - p_G).max())},
                  kappa_lb=None, R2=None, t_star_approx=(None if c <= 0 else float(0.2 / c ** 2)))
        scenarios[sid] = sc
        rows.append(dict(scenario=sid, target=tname, ladder='ML_K4_marginals', K=4, m=sc['m'],
                         n_target=sc['n_target'], margin_c=sc['margin_c'], t_star=sc['t_star_approx'],
                         C_min=sc['C_min'], cost_lp_discard=np.nan, cost_quota=np.nan,
                         naive_cheapest=list(sc['naive_linf'].values())[0],
                         naive_uniform=sc['naive_linf']['uniform'],
                         naive_poolprop=sc['naive_linf']['poolprop'], kappa=np.nan, N_c=np.nan))
        # respondent-level pools (for the realism mode) — primary target only
        if tname == 'S-DIAB':
            keep = [a for a in arms if len(sub[arms[a][0].reindex(sub.index).fillna(False)]) >= MIN_POOL_N]
            mask = np.stack([arms[a][0].reindex(sub.index).fillna(False).values for a in keep])
            lab = {lid: pd.get_dummies(pd.Categorical(np.asarray(LADDERS[lid][1](sub)),
                                                      categories=LADDERS[lid][0])).values.astype(float)
                   for lid in LADDERS}
            out = np.stack([OUTCOMES[k](sub).values for k in OUTCOMES], axis=1)
            np.savez_compressed(os.path.join(OUT, f"pools_{tname}.npz"), w=sub.w.values, arm_mask=mask,
                                arms=np.array(keep), outcomes=out, outcome_names=np.array(list(OUTCOMES)),
                                ml=labfn(sub), **{f"lab_{k}": v for k, v in lab.items()})
    # two-arm Tier-1 scenarios (sec. 2 setting)
    sub = d[TARGETS['S-DIAB'](d) & d.old.notna()].copy()
    for sid, (lid, pair) in {'T1-AGE': ('L3_K3_age', ('ONLINE', 'LANDLINE')),
                             'T1-BLACK': ('L1_K2_black', ('ONLINE', 'SITE_S'))}.items():
        labels, labfn = (['65+', '<65'], lambda g: np.where(g.old == 1, '65+', '<65')) if sid == 'T1-AGE' \
                        else LADDERS['L1_K2_black']
        sc = build_scenario(f"S-DIAB__{sid}", sub, arms, labels, labfn, arm_names=set(pair))
        scenarios[sc['id']] = sc
        rows.append(dict(scenario=sc['id'], target='S-DIAB', ladder=sid, K=sc['K'], m=sc['m'],
                         n_target=sc['n_target'], margin_c=sc['margin_c'], t_star=sc['t_star_approx'],
                         C_min=sc['C_min'], cost_lp_discard=sc['cost_lp_discard'],
                         cost_quota=sc['cost_quota_cheapest'],
                         naive_cheapest=list(sc['naive_linf'].values())[0],
                         naive_uniform=sc['naive_linf']['uniform'],
                         naive_poolprop=sc['naive_linf']['poolprop'], kappa=sc['kappa_lb'], N_c=sc['N_c']))
    for sid, sc in scenarios.items():
        with open(os.path.join(OUT, sid.replace('/', '_') + '.json'), 'w') as f:
            json.dump(sc, f, indent=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'E0_scenarios.csv'), index=False)
    prov = dict(adult_sha256=hashlib.sha256(open(f"{UP}/adult23.csv", 'rb').read(2 ** 22)).hexdigest()[:16],
                n_adults=int(len(d)), cost_online=COST_ONLINE, cost_offline=COST_OFFLINE,
                min_pool_n=MIN_POOL_N)
    json.dump(prov, open(os.path.join(OUT, 'provenance.json'), 'w'), indent=1)
    pd.set_option('display.width', 250)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == '__main__':
    main()
