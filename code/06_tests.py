"""The paper's tests, P1-P4.

Exposure: INCB analgesic opioid consumption (total S-DDD less methadone) per million per day, averaged over the 2000-02, 2003-05, 2006-08
and 2009-11 windows (at least two), so it comes before fentanyl everywhere except Estonia. `rx_pre_arrival` keeps only windows that
ended before a country's illicit supply was first seen (Estonia: none, so it's missing there). Oral = oxycodone + hydrocodone; patch =
pharmaceutical fentanyl, mostly transdermal. All enter as log(1 + S-DDD).
Outcome: WHO underlying-cause rates per 100,000, narcotics (X42, X62, Y12) and all drug poisoning, as log ratios of 2019-21 to 2010-12
(each window needing two years; 0.05 added so zeros can be logged).
Supply: code/05's status as of 2019. Wealth: log GDP per capita at PPP, 2010-12.
P1: prescribing predicts the rise. P2: only where illicit supply arrived (interaction). P3: high prescribers without it stay flat.
P4: oral opioids carry the gradient more than patches. OLS with HC3 errors; Spearman beside each.

  python3 code/06_tests.py <incb csv> <who deaths csv> <arrival csv> <worldbank_gdp_per_capita date folder> <out folder>
"""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr

PRE = ["2000-2002", "2003-2005", "2006-2008", "2009-2011"]
WINDOW_END = {"2000-2002": 2002, "2003-2005": 2005, "2006-2008": 2008, "2009-2011": 2011, "2012-2014": 2014}
WHO_ALIAS = {"bolivia": "bolivia (plurinational state of)", "czech republic": "czechia", "hong kong sar": "china, hong kong sar",
             "libyan arab jamahiriya": "libya", "netherlands": "netherlands (kingdom of the)", "turkey": "turkiye",
             "saint vincent and grenadines": "saint vincent and the grenadines", "venezuela": "venezuela (bolivarian rep of)",
             "serbia and montenegro, former": "serbia and montenegro"}
WB_ALIAS = {"bahamas, the": "bahamas", "egypt, arab rep": "egypt", "hong kong sar, china": "china, hong kong sar",
            "iran, islamic rep": "iran (islamic republic of)", "kyrgyz republic": "kyrgyzstan", "micronesia, fed sts": "micronesia (federated states of)",
            "korea, rep": "republic of korea", "moldova": "republic of moldova", "st kitts and nevis": "saint kitts and nevis", "st lucia": "saint lucia",
            "st vincent and the grenadines": "saint vincent and the grenadines", "slovak republic": "slovakia", "united states": "united states of america",
            "venezuela, rb": "venezuela (bolivarian rep of)", "netherlands": "netherlands (kingdom of the)", "gambia, the": "gambia",
            "lao pdr": "lao people's democratic republic", "congo, dem rep": "democratic republic of the congo", "congo, rep": "congo",
            "korea, dem people's rep": "democratic people's republic of korea", "yemen, rep": "yemen", "macao sar, china": "china, macao sar",
            "cote d'ivoire": "cote d'ivoire", "tanzania": "united republic of tanzania", "viet nam": "viet nam", "syrian arab republic": "syrian arab republic"}


def key(s):
    s = str(s).lower().replace("’", "'").replace("ü", "u").replace("ç", "c").replace("é", "e").replace("ô", "o")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ,()']", " ", s)).strip()


def logratio(a, b):
    return np.log((a + 0.05) / (b + 0.05))


def frame(incb, who, arrival, gdp_folder):
    x = pd.read_csv(incb)
    x["key"] = x.country.map(key)
    x["oral"] = x.oxycodone + x.hydrocodone
    arr = pd.read_csv(arrival)
    arr["key"] = arr.country.map(key)
    pre = x[x.window.isin(PRE)].groupby("key").agg(rx_pre=("analgesic", "mean"), n_windows=("analgesic", "count"),
                                                   oral_pre=("oral", "mean"), patch_pre=("fentanyl", "mean"))
    pre = pre[pre.n_windows >= 2]
    first = arr.set_index("key").first_seen
    x["end"] = x.window.map(WINDOW_END)
    ok = x[x.key.map(first).isna() | (x.end < x.key.map(first))]
    pa = ok[ok.window.isin(PRE)].groupby("key").analgesic.mean().rename("rx_pre_arrival")
    w = pd.read_csv(who)
    w["key"] = w.country_name.map(key).replace(WHO_ALIAS)
    w["Year"] = pd.to_numeric(w.Year, errors="coerce")

    def win(col, a, b):
        k = w[w.Year.between(a, b)].groupby("key")[col].agg(["mean", "count"])
        return k["mean"].where(k["count"] >= 2)

    deaths = pd.DataFrame({"narc_1012": win("narcotics_per_100k", 2010, 2012), "narc_1921": win("narcotics_per_100k", 2019, 2021),
                           "drug_1012": win("drug_poisoning_per_100k", 2010, 2012), "drug_1921": win("drug_poisoning_per_100k", 2019, 2021),
                           "pop_1012": win("population", 2010, 2012)})
    g = json.loads((pathlib.Path(gdp_folder) / "NY.GDP.PCAP.PP.KD.json").read_text())[1]
    g = pd.DataFrame([(key(r["country"]["value"]), int(r["date"]), r["value"]) for r in g if r["value"] is not None], columns=["key", "year", "gdp"])
    g["key"] = g.key.replace(WB_ALIAS)
    gdp = g[g.year.between(2010, 2012)].groupby("key").gdp.mean()
    m = (pre.join(pa).join(deaths, how="inner").join(gdp)
         .join(arr.set_index("key")[["country", "status", "status_alt", "first_seen", "at_scale", "in_review_scope", "established_by_2019",
                                     "any_illicit_by_2019", "established_alt", "any_illicit_alt", "any_illicit_or_transit"]]))
    m = m.dropna(subset=["rx_pre", "narc_1012", "narc_1921"]).reset_index()
    m["status"] = m.status.fillna("none known")
    m["established_by_2019"] = m.established_by_2019.fillna(False).astype(bool)
    m["any_illicit_by_2019"] = m.any_illicit_by_2019.fillna(False).astype(bool)
    m["in_review_scope"] = m.in_review_scope.fillna(False).astype(bool)
    for c in ("established_alt", "any_illicit_alt", "any_illicit_or_transit"):
        m[c] = m[c].fillna(False).astype(bool)
    m["est_alt"] = m.established_alt.astype(int)
    m["anyill_alt"] = m.any_illicit_alt.astype(int)
    m["anyill_transit"] = m.any_illicit_or_transit.astype(int)
    m["narc_change"] = logratio(m.narc_1921, m.narc_1012)
    m["drug_change"] = logratio(m.drug_1921, m.drug_1012)
    for c in ("rx_pre", "rx_pre_arrival", "oral_pre", "patch_pre"):
        m["l" + c] = np.log1p(m[c])
    m["lgdp"] = np.log(m.gdp)
    m["est"] = m.established_by_2019.astype(int)
    m["anyill"] = m.any_illicit_by_2019.astype(int)
    m["anyill_nouk"] = (m.any_illicit_by_2019 & (m.country != "United Kingdom")).astype(int)  # the UK's rise isn't known to be fentanyl
    return m


def fit(d, formula, label, out):
    k = d.dropna(subset=[v for v in re.findall(r"[a-z_]+", formula) if v in d.columns])
    r = smf.ols(formula, data=k).fit(cov_type="HC3")
    for term in r.params.index:
        if term != "Intercept":
            out.append(dict(model=label, formula=formula, n=int(r.nobs), term=term, b=r.params[term], se=r.bse[term], p=r.pvalues[term]))
    return r


def spear(d, a, b, label, out):
    k = d.dropna(subset=[a, b])
    s = spearmanr(k[a], k[b])
    out.append(dict(model=label, formula=f"spearman({a}, {b})", n=len(k), term="rho", b=s[0], se=np.nan, p=s[1]))


def main(incb, who, arrival, gdp_folder, outdir):
    outdir = pathlib.Path(outdir)
    m = frame(incb, who, arrival, gdp_folder)
    m.to_csv(outdir / "tests_countries.csv", index=False)
    res = []
    samples = {"all": m, "pop 1m+": m[m.pop_1012 >= 1e6], "review scope": m[m.in_review_scope], "no Estonia": m[m.country != "Estonia"]}
    for oc in ("narc_change", "drug_change"):
        for sname, d in samples.items():
            tag = f"{oc} | {sname}"
            spear(d, "rx_pre", oc, f"P1 | {tag}", res)
            fit(d, f"{oc} ~ lrx_pre + lgdp", f"P1 | {tag}", res)
            fit(d, f"{oc} ~ lrx_pre * est + lgdp", f"P2 established | {tag}", res)
            fit(d, f"{oc} ~ lrx_pre * anyill + lgdp", f"P2 any illicit | {tag}", res)
            spear(d[~d.any_illicit_by_2019], "rx_pre", oc, f"P2 no illicit supply | {tag}", res)
            fit(d, f"{oc} ~ lrx_pre * anyill_nouk + lgdp", f"P2 any illicit, UK coded none | {tag}", res)
            fit(d, f"{oc} ~ lrx_pre * est_alt + lgdp", f"P2 established, alternative coding | {tag}", res)
            fit(d, f"{oc} ~ lrx_pre * anyill_alt + lgdp", f"P2 any illicit, alternative coding | {tag}", res)
            fit(d, f"{oc} ~ lrx_pre * anyill_transit + lgdp", f"P2 any illicit or transit | {tag}", res)
            fit(d, f"{oc} ~ loral_pre + lpatch_pre + lgdp", f"P4 | {tag}", res)
            fit(d, f"{oc} ~ (loral_pre + lpatch_pre) * anyill + lgdp", f"P4 within supply | {tag}", res)
        fit(m, f"{oc} ~ lrx_pre_arrival * est + lgdp", f"P2 pre-arrival exposure | {oc} | all", res)
    r = pd.DataFrame(res)
    r.to_csv(outdir / "tests_results.csv", index=False)

    print(f"Countries: {len(m)} with prescribing before 2012 and both death windows; {int(m.est.sum())} established, "
          f"{int(m.anyill.sum())} with any illicit supply; {int(m.lgdp.notna().sum())} with GDP; {int(m.in_review_scope.sum())} in the reviews' scope")
    show = r[r.term.isin(["rho", "lrx_pre", "lrx_pre:est", "lrx_pre:anyill", "lrx_pre:anyill_nouk", "lrx_pre:est_alt", "lrx_pre:anyill_alt", "loral_pre", "lpatch_pre", "loral_pre:anyill",
                          "lpatch_pre:anyill", "lrx_pre_arrival", "lrx_pre_arrival:est"])]
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(show[["model", "n", "term", "b", "se", "p"]].round(3).to_string(index=False))

    # P3: high prescribers (top quarter of rx_pre) with and without illicit supply
    q = m.rx_pre.quantile(0.75)
    hi = m[m.rx_pre >= q].sort_values("rx_pre", ascending=False)
    cols = ["country", "status", "status_alt", "rx_pre", "oral_pre", "patch_pre", "narc_1012", "narc_1921", "narc_change", "drug_1012", "drug_1921", "drug_change"]
    print(f"\nP3: top quarter of prescribing (analgesic S-DDD >= {q:,.0f}), {len(hi)} countries")
    print(hi[cols].round(2).to_string(index=False))
    # do the 2019-21 levels differ?
    lv_w, lv_wo = hi.loc[hi.any_illicit_by_2019, "narc_1921"], hi.loc[~hi.any_illicit_by_2019, "narc_1921"]
    print(f"P3 levels 2019-21, top quarter: with supply {sorted(lv_w.round(2))}; without {sorted(lv_wo.round(2))}; "
          f"Mann-Whitney p = {mannwhitneyu(lv_w, lv_wo).pvalue:.3f}")
    for flag in ("any_illicit_by_2019", "any_illicit_alt"):
        for oc in ("narc_change", "drug_change"):
            a, b = hi.loc[hi[flag], oc], hi.loc[~hi[flag], oc]
            print(f"{flag} | {oc}: with illicit supply median {a.median():.2f} (n={len(a)}); without {b.median():.2f} (n={len(b)}); "
                  f"without and more than doubled: {(b > np.log(2)).sum()}; fell: {(b < 0).sum()}")
    # Did high prescribers become the fentanyl countries?
    m["rx_rank"] = m.rx_pre.rank(ascending=False).astype(int)
    print("\nPrescribing rank (of", len(m), ") of countries with illicit supply:")
    print(m[m.any_illicit_by_2019].sort_values("rx_rank")[["country", "status", "rx_rank", "rx_pre", "narc_1012", "narc_1921", "drug_1012", "drug_1921"]].round(2).to_string(index=False))
    # Around arrival: deaths in the three years before illicit supply was first seen and in years 2-4 after, where a year is known
    w = pd.read_csv(who)
    w["key"] = w.country_name.map(key).replace(WHO_ALIAS)
    w["Year"] = pd.to_numeric(w.Year, errors="coerce")
    rows = []
    for _, c in m[m.any_illicit_by_2019 & m.first_seen.notna()].iterrows():
        f = int(c.first_seen)
        k = w[w.key == c["key"]]
        pre, post = k[k.Year.between(f - 3, f - 1)], k[k.Year.between(f + 2, f + 4)]
        rows.append(dict(country=c.country, first_seen=f, rx_pre_arrival=c.rx_pre_arrival, narc_before=pre.narcotics_per_100k.mean(),
                         narc_after=post.narcotics_per_100k.mean(), drug_before=pre.drug_poisoning_per_100k.mean(), drug_after=post.drug_poisoning_per_100k.mean()))
    a = pd.DataFrame(rows)
    a["narc_change"] = logratio(a.narc_after, a.narc_before)
    a["drug_change"] = logratio(a.drug_after, a.drug_before)
    a.to_csv(outdir / "tests_around_arrival.csv", index=False)
    print("\nAround arrival (3 years before first seen; years 2-4 after):")
    print(a.round(2).to_string(index=False))
    s = m[m.in_review_scope]
    u = mannwhitneyu(s.loc[s.established_by_2019, "rx_pre"], s.loc[~s.established_by_2019, "rx_pre"])
    print(f"Within the reviews' scope, established vs not, prescribing: Mann-Whitney p = {u.pvalue:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
