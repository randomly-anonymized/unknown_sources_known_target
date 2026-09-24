"""Headline tables: E5 (trial-scale comparison), sample complexity, slopes. Writes results/E5_*.csv."""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
RES, SC = f"{ROOT}/results", f"{ROOT}/data/scenarios"


def slope(g, tmin=3000):
    g = g[g["T"] >= tmin].sort_values("T")
    y = g.err_l2_mean if "err_l2_mean" in g else g.err_scalar_mean
    return float(np.polyfit(np.log(g["T"]), np.log(y), 1)[0])


def nearest(df, t):
    return df.iloc[(df["T"] - t).abs().argmin()]


def sample_complexity(df, eps_list=(0.05, 0.02, 0.01, 0.005), col="err_l2_q90"):
    out = {}
    g = df.sort_values("T")
    for e in eps_list:
        hit = g[g[col] <= e]
        out[e] = int(hit["T"].iloc[0]) if len(hit) else np.inf
    return out


def main():
    rows = []
    # ---- Tier 1
    for tag in ["T1-AGE", "T1-BLACK"]:
        d = pd.read_csv(f"{RES}/E1_rates_{tag}.csv")
        for p, g in d.groupby("policy"):
            r = nearest(g, 1000)
            rows.append(dict(block="E1", scenario=tag, policy=p, T=int(r["T"]),
                             err=r.err_scalar_mean, S=r.S_scalar_mean, slope=slope(g, 1000),
                             cost_per_sample=r.cost_per_sample,
                             **{f"N90(eps={e})": v for e, v in
                                sample_complexity(g, col="err_l2_q90").items()}))
    # ---- Tier 2 (categorical headline + multi-label)
    for tag, files in [("ETH", "E2_rates_eth_*.csv"), ("ML", "E2_rates_ml_*.csv"),
                       ("BLACKxAGE", "E2_rates_blackage.csv")]:
        d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/{files}")])
        for p, g in d.groupby("policy"):
            g = g.sort_values("T"); r = nearest(g, 1000)
            rows.append(dict(block="E2", scenario=tag, policy=p, T=int(r["T"]), err=r.err_l2_mean,
                             S=r.S_norm_mean, slope=slope(g), cost_per_sample=r.cost_per_sample,
                             **{f"N90(eps={e})": v for e, v in sample_complexity(g).items()}))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/E5_headline.csv", index=False)
    pd.set_option("display.width", 230)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.5f}"))

    # ---- per-stratum error at trial scale for the headline scenario
    sc = json.load(open(f"{SC}/S-DIAB__L2_K3_eth.json"))
    mu, p_G, names = np.array(sc["mu"]), np.array(sc["p_G"]), sc["arms"]
    cheap = int(np.argmin(sc["cost"]))
    tab = pd.DataFrame({"stratum": sc["labels"], "target": p_G,
                        "online-only (N1)": mu[cheap], "uniform (N2)": mu.mean(0),
                        "pool-proportional (N3)": np.array(sc["pool_share"]) @ mu / np.sum(sc["pool_share"])})
    tab["N1 error (pp)"] = 100 * (tab["online-only (N1)"] - tab.target)
    tab["N2 error (pp)"] = 100 * (tab["uniform (N2)"] - tab.target)
    tab.to_csv(f"{RES}/E5_per_stratum.csv", index=False)
    print("\nper-stratum composition (asymptotic, naive rules):")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\nchannel table (headline scenario):")
    ch = pd.DataFrame(dict(channel=names, n=sc["n_arm"], pool_share=sc["pool_share"], cost=sc["cost"],
                           **{lab: mu[:, k] for k, lab in enumerate(sc["labels"])}))
    ch.to_csv(f"{RES}/E0_channels_headline.csv", index=False)
    print(ch.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()
