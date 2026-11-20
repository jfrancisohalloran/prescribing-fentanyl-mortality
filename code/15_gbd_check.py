"""Global robustness check: the country tests on GBD 2023's modelled opioid-use-disorder death rates, for every country with INCB data.

GBD covers about 204 countries, against 60 with usable WHO data, so this asks whether the country results hold worldwide. GBD rates are
modelled and smoothed, with covariates that may relate to opioid availability, so they check the WHO results rather than replace them.
Outcome: age-standardized opioid-use-disorder death rates per 100,000, both sexes, 2019-21 against 2010-12 (log ratio), and the 2019-21
level. Exposure, supply and GDP as in code/06.

  python3 code/15_gbd_check.py <ihme_gbd date folder> <incb csv> <arrival csv> <worldbank_gdp_per_capita date folder> <out folder>
"""
import importlib.util
import json
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("t6", HERE / "06_tests.py")
t6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t6)
GBD_ALIAS = {"united states of america": "united states of america", "taiwan (province of china)": "taiwan", "turkiye": "turkiye",
             "netherlands": "netherlands (kingdom of the)", "venezuela (bolivarian republic of)": "venezuela (bolivarian rep of)",
             "democratic people's republic of korea": "democratic people's republic of korea", "cote d'ivoire": "cote d'ivoire"}


def main(gbd, incb, arrival, gdp_folder, out):
    z = next(pathlib.Path(gbd).glob("IHME-GBD_2023_DATA-*.zip"))
    with zipfile.ZipFile(z) as f:
        name = next(n for n in f.namelist() if n.endswith(".csv"))
        d = pd.read_csv(f.open(name))
    d = d[(d.cause_name == "Opioid use disorders") & (d.age_name == "Age-standardized") & (d.metric_name == "Rate") & (d.sex_name == "Both")]
    d["key"] = d.location_name.map(t6.key).replace(GBD_ALIAS)

    def win(a, b):
        return d[d.year.between(a, b)].groupby("key").val.mean()

    g = pd.DataFrame({"oud_1012": win(2010, 2012), "oud_1921": win(2019, 2021)})
    x = pd.read_csv(incb)
    x["key"] = x.country.map(t6.key)
    pre = x[x.window.isin(t6.PRE)].groupby("key").agg(rx_pre=("analgesic", "mean"), n=("analgesic", "count"))
    pre = pre[pre.n >= 2]
    a = pd.read_csv(arrival)
    a["key"] = a.country.map(t6.key)
    w = json.loads((pathlib.Path(gdp_folder) / "NY.GDP.PCAP.PP.KD.json").read_text())[1]
    w = pd.DataFrame([(t6.key(r["country"]["value"]), int(r["date"]), r["value"]) for r in w if r["value"] is not None], columns=["key", "year", "gdp"])
    w["key"] = w.key.replace(t6.WB_ALIAS)
    gdp = w[w.year.between(2010, 2012)].groupby("key").gdp.mean()
    m = (pre.join(g, how="inner").join(gdp).join(a.set_index("key")[["country", "status", "established_by_2019", "any_illicit_by_2019", "transit"]])
         .dropna(subset=["oud_1012", "oud_1921"]).reset_index())
    m["any_illicit_by_2019"] = m.any_illicit_by_2019.fillna(False).astype(bool)
    m["established_by_2019"] = m.established_by_2019.fillna(False).astype(bool)
    m["change"] = np.log(m.oud_1921 / m.oud_1012)
    m["lrx"] = np.log1p(m.rx_pre)
    m["lgdp"] = np.log(m.gdp)
    m["anyill"] = m.any_illicit_by_2019.astype(int)
    m.to_csv(pathlib.Path(out) / "gbd_check_countries.csv", index=False)
    unmatched = sorted(set(d.location_name[~d.key.isin(pre.index)]))
    print(f"{len(m)} countries with INCB prescribing and GBD rates; {int(m.established_by_2019.sum())} established, {int(m.anyill.sum())} any illicit; "
          f"GBD locations without INCB data: {len(unmatched)} ({', '.join(unmatched[:10])} ...)")
    res = []
    for oc in ("change", "np.log(oud_1921)"):
        k = m.dropna(subset=["lgdp"])
        r1 = smf.ols(f"{oc} ~ lrx + lgdp", data=k).fit(cov_type="HC3")
        r2 = smf.ols(f"{oc} ~ lrx * anyill + lgdp", data=k).fit(cov_type="HC3")
        res += [dict(outcome=oc, term="lrx", b=r1.params.lrx, se=r1.bse.lrx, p=r1.pvalues.lrx, n=int(r1.nobs)),
                dict(outcome=oc, term="lrx:anyill", b=r2.params["lrx:anyill"], se=r2.bse["lrx:anyill"], p=r2.pvalues["lrx:anyill"], n=int(r2.nobs))]
    rho = spearmanr(m.rx_pre, m.change)
    res.append(dict(outcome="change", term="spearman", b=rho[0], se=np.nan, p=rho[1], n=len(m)))
    q = m.rx_pre.quantile(0.75)
    hi = m[m.rx_pre >= q]
    wi, wo = hi[hi.any_illicit_by_2019], hi[~hi.any_illicit_by_2019]
    mw = mannwhitneyu(wi.change, wo.change)
    res.append(dict(outcome="change", term="top quarter with vs without supply (Mann-Whitney)", b=wi.change.median() - wo.change.median(), se=np.nan, p=mw.pvalue, n=len(hi)))
    mwl = mannwhitneyu(wi.oud_1921, wo.oud_1921)
    res.append(dict(outcome="level 2019-21", term="top quarter with vs without supply (Mann-Whitney)", b=wi.oud_1921.median() - wo.oud_1921.median(), se=np.nan, p=mwl.pvalue, n=len(hi)))
    m["rank"] = m.rx_pre.rank(ascending=False).astype(int)
    r = pd.DataFrame(res)
    r.to_csv(pathlib.Path(out) / "gbd_check_results.csv", index=False)
    print(r.round(3).to_string(index=False))
    print(f"\nTop quarter (S-DDD >= {q:,.0f}, {len(hi)} countries): with supply median change {np.expm1(wi.change.median())*100:+.0f}% "
          f"(n={len(wi)}), level {wi.oud_1921.median():.2f}; without {np.expm1(wo.change.median())*100:+.0f}% (n={len(wo)}), level range "
          f"{wo.oud_1921.min():.2f}-{wo.oud_1921.max():.2f}")
    print("Established markets' prescribing ranks of", len(m), ":", m.loc[m.established_by_2019, ["country", "rank", "oud_1921"]].sort_values("rank").round(2).values.tolist())
    print("Highest GBD opioid death rates 2019-21:", m.sort_values("oud_1921", ascending=False).head(8)[["country", "oud_1921", "status"]].round(2).values.tolist())


if __name__ == "__main__":
    main(*sys.argv[1:6])
