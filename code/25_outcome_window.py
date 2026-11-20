"""Illicit fentanyl markets coded for the outcome window, 2019-21, beside the codes "by 2019".

code/05 codes a country's market as it stood by 2019, so a market that ended before the outcome window (Sweden's, dismantled in
2017-18) or whose deaths in 2019-21 did not come from fentanyl (the UK's early-2010s outbreak) counts as supply. This script leaves
code/05 and code/06 as they are and refits their country tests on their own outputs under three further codings:

- `noswuk`: code/06's any-illicit coding with Sweden and the UK coded as having no market, a stricter coding.
- `current`: a market present in 2019-21 by the evidence in appendix A3 and A4: the USA, Canada, Estonia (disrupted in 2017-18 and
  partly restored), Latvia and Lithuania (carfentanil, continuing after May 2019). Bulgaria's and Slovakia's episodes came in the
  early 2010s and count as none.
- `current_noee`: as `current`, with Estonia coded none, since Abel-Ollo et al. 2025 date the end of its fentanyl era to the end
  of 2017.

For each coding it fits code/06's models (change in narcotic and in all drug poisoning on log prescribing, with the interaction and
log GDP, HC3 errors), repeats the heavy-quarter contrasts of change and of 2019-21 levels with their medians and ranks, and repeats
the level contrast among the 13 heaviest prescribers of the 60 (the "13 of the world's 15 heaviest consumers"). With a GBD frame
from code/15 it refits GBD's interaction and its level contrast on the same codings.

  python3 code/25_outcome_window.py <out folder with tests_countries.csv> [gbd_check_countries.csv]

Writes out/outcome_window_results.csv and out/outcome_window_heavy.csv (countries' prescribing ranks, codes and rates: no deaths
counts). `--selftest` runs the code on a simulated frame instead.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

USA, CAN = "United States of America", "Canada"
CURRENT = {USA, CAN, "Estonia", "Latvia", "Lithuania"}
CODINGS = {
    "anyill": None,  # code/06's coding, refitted here as the reference
    "noswuk": lambda c, any_: any_ and c not in {"Sweden", "United Kingdom"},
    "current": lambda c, any_: c in CURRENT,
    "current_noee": lambda c, any_: c in CURRENT - {"Estonia"},
}


def code(m):
    m = m.copy()
    any_ = m.any_illicit_by_2019.astype(bool)
    for name, f in CODINGS.items():
        m[name] = any_.astype(int) if f is None else [int(f(c, a)) for c, a in zip(m.country, any_)]
    return m


def exact_mw(a, b):
    """Two-sided exact Mann-Whitney p (scipy's exact method when there are no ties)."""
    return mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue


def countries(m):
    res = []
    for name in CODINGS:
        for oc in ("narc_change", "drug_change"):
            k = m.dropna(subset=[oc, "lrx_pre", "lgdp"])
            r = smf.ols(f"{oc} ~ lrx_pre * {name} + lgdp", data=k).fit(cov_type="HC3")
            t = f"lrx_pre:{name}"
            lo, hi = r.conf_int().loc[t]
            res.append(dict(source="WHO", coding=name, outcome=oc, model="interaction, adjusted for log GDP", n=int(r.nobs),
                            estimate=r.params[t], lo=lo, hi=hi, p=r.pvalues[t]))
    q = m.rx_pre.quantile(0.75)
    hi_ = m[m.rx_pre >= q].copy()
    top13 = m.nlargest(13, "rx_pre")
    for name in CODINGS:
        for label, d in (("heaviest quarter", hi_), ("13 heaviest", top13)):
            w, wo = d[d[name] == 1], d[d[name] == 0]
            for oc in ("narc_1921", "narc_change"):
                if len(w) == 0 or len(wo) == 0:
                    continue
                res.append(dict(source="WHO", coding=name, outcome=oc, model=f"Mann-Whitney, {label}", n=len(d),
                                n_with=len(w), median_with=w[oc].median(), median_without=wo[oc].median(),
                                min_with=w[oc].min(), max_with=w[oc].max(), min_without=wo[oc].min(), max_without=wo[oc].max(),
                                p=exact_mw(w[oc], wo[oc]),
                                ranks_with=" ".join(str(int(x)) for x in sorted(d[oc].rank(ascending=False)[d[name] == 1]))))
    heavy = hi_.assign(rx_rank=m.rx_pre.rank(ascending=False).loc[hi_.index].astype(int))
    heavy = heavy[["country", "rx_rank", "rx_pre", "status"] + list(CODINGS) + ["narc_1012", "narc_1921", "narc_change"]]
    return res, heavy.sort_values("rx_rank")


def gbd(g):
    g = code(g)
    res = []
    for name in CODINGS:
        k = g.dropna(subset=["change", "lrx", "lgdp"])
        r = smf.ols(f"change ~ lrx * {name} + lgdp", data=k).fit(cov_type="HC3")
        t = f"lrx:{name}"
        lo, hi = r.conf_int().loc[t]
        res.append(dict(source="GBD", coding=name, outcome="change", model="interaction, adjusted for log GDP", n=int(r.nobs),
                        estimate=r.params[t], lo=lo, hi=hi, p=r.pvalues[t]))
        h = g[g.rx_pre >= g.rx_pre.quantile(0.75)]
        w, wo = h[h[name] == 1], h[h[name] == 0]
        res.append(dict(source="GBD", coding=name, outcome="oud_1921", model="Mann-Whitney, heaviest quarter", n=len(h), n_with=len(w),
                        median_with=w.oud_1921.median(), median_without=wo.oud_1921.median(),
                        p=mannwhitneyu(w.oud_1921, wo.oud_1921).pvalue))
    return res


def simulated(seed=8):
    rng = np.random.default_rng(seed)
    names = [USA, CAN, "Estonia", "Latvia", "Lithuania", "Sweden", "United Kingdom", "Bulgaria", "Slovakia"] + [f"C{i}" for i in range(51)]
    rx = np.r_[[30000, 25000, 900, 1100, 1000, 8000, 7500, 700, 2500], rng.lognormal(7, 1.2, 51)]
    any_ = np.r_[[True] * 9, [False] * 51]
    narc1012 = rng.lognormal(0, 0.8, 60)
    narc1921 = narc1012 * np.exp(rng.normal(0, 0.4, 60) + 0.25 * any_ * np.log1p(rx) / 5)
    m = pd.DataFrame(dict(country=names, rx_pre=rx, any_illicit_by_2019=any_, status=np.where(any_, "episodic", "none known"),
                          narc_1012=narc1012, narc_1921=narc1921, lgdp=rng.normal(10, 0.6, 60)))
    m["narc_change"] = np.log((m.narc_1921 + 0.05) / (m.narc_1012 + 0.05))
    m["drug_change"] = m.narc_change + rng.normal(0, 0.2, 60)
    m["lrx_pre"] = np.log1p(m.rx_pre)
    g = m.rename(columns={"narc_1921": "oud_1921", "narc_1012": "oud_1012"}).assign(lrx=m.lrx_pre, change=m.narc_change)
    return m, g


def main(out, gbd_csv=None):
    out = pathlib.Path(out)
    if out.name == "--selftest":
        m, g = simulated()
        m = code(m)
        res, heavy = countries(m)
        res += gbd(g)
        r = pd.DataFrame(res)
        # the coding must leave the reference unchanged and drop only Sweden and the UK
        assert m.loc[m.country.isin(["Sweden", "United Kingdom"]), "noswuk"].eq(0).all()
        assert (m.anyill - m.noswuk).sum() == 2 and m.current.sum() == 5 and m.current_noee.sum() == 4
        # exact test: two with-market countries at ranks 1 and 2 of 15 gives 2 / C(15, 2)
        p = exact_mw(pd.Series([12.0, 7.4]), pd.Series(np.linspace(0.3, 4.6, 13)))
        assert abs(p - 2 / 105) < 1e-9, p
        print(r.round(3).to_string())
        print("selftest passed")
        return
    m = code(pd.read_csv(out / "tests_countries.csv"))
    res, heavy = countries(m)
    if gbd_csv:
        res += gbd(pd.read_csv(gbd_csv))
    r = pd.DataFrame(res)
    r.to_csv(out / "outcome_window_results.csv", index=False)
    heavy.to_csv(out / "outcome_window_heavy.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(heavy.round(2).to_string(index=False))
        print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
