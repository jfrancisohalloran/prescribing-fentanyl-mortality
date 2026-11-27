"""The range of prescribing that closing the access gap involves. Across the 60 countries prescribing and income rise together, and
no lower-income country prescribed heavily, so the comparison cannot say what heavy prescribing does in a poorer country. It can
say what happened across the range such a country would move through. This script gives (a) countries in bands of prescribing,
with their income and narcotic-poisoning mortality, and (b) among countries without an illicit fentanyl market, whether the change
in mortality from 2010-12 to 2019-21 followed prescribing or the growth of prescribing from 2000-02 to 2009-11, for all of them,
for those below the 60 countries' median income, and for those below 5000 S-DDD; and the same question in GBD's modelled estimates
for countries with GDP per person under 15 000 dollars. Set after the paper's results and a review were seen; every variant fitted
is written out.

(c) Poverty under one country's regulation: the paper's county model of overdose deaths (code/33's main model) fitted within
thirds of counties by economic distress, and once with prescribing times distress times the share of the window after arrival.
If heavy prescribing did more harm in poorer places, the ratio should be higher in the most distressed third.

Writes out/dose_range_bands.csv, out/dose_range_tests.csv and out/dose_range_counties.csv.

  python3 code/43_dose_range.py <out> <derived inputs folder> <arcos_county_year.csv> <wonder_pooled date folder>
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

BANDS = [0, 100, 200, 1000, 5000, 1e9]
LABELS = ["100 or less", "101 to 200", "201 to 1000", "1001 to 5000", "over 5000"]


def rho(res, sample, what, a, b):
    k = pd.concat([a, b], axis=1).dropna()
    r, p = spearmanr(k.iloc[:, 0], k.iloc[:, 1])
    res.append(dict(sample=sample, test=what, kind="Spearman", estimate=r, lo=np.nan, hi=np.nan, p=p, n=len(k)))


def counties(out, onea, arcos, wonder):
    spec = importlib.util.spec_from_file_location("c33", pathlib.Path(__file__).resolve().parent / "33_county_arrival.py")
    c33 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c33)
    cb = c33.counties(onea, arcos)
    cb = cb[cb.rx_missing == 0].copy()
    p25, p75 = np.percentile(cb.rx, [25, 75])
    iqr = np.log(p75 / p25)
    cb["third"] = pd.qcut(cb.zx, 3, labels=["least distressed third", "middle third", "most distressed third"])
    cb["lrx_zx"] = cb.lrx * cb.zx
    res = []
    for bound in c33.BOUNDS:
        p, _ = c33.informative(c33.panel(cb, pathlib.Path(wonder), bound))
        c33.run(p, "lrx", [("lrx_x_after", p.lrx * p.after)], "all counties", bound, res, iqr=iqr)
        for third in cb.third.cat.categories:
            k = p[p.third == third].reset_index(drop=True)
            c33.run(k, "lrx", [("lrx_x_after", k.lrx * k.after)], third, bound, res, iqr=iqr)
        c33.run(p, "lrx", [("lrx_x_after", p.lrx * p.after), ("zx_x_after", p.zx * p.after), ("lrx_zx_x_after", p.lrx_zx * p.after)],
                "prescribing times distress", bound, res, extra=("zx", "lrx_zx"), iqr=iqr)
    r = pd.DataFrame(res)
    r = r[r.term.isin(["lrx_x_after", "zx_x_after", "lrx_zx_x_after"])]
    r.to_csv(out / "dose_range_counties.csv", index=False)
    with pd.option_context("display.width", 250):
        print(r[["model", "bound", "term", "b", "lo", "hi", "p", "ratio", "ratio_lo", "ratio_hi", "counties", "states", "converged"]].round(3).to_string(index=False))


def main(out, onea, arcos, wonder):
    out = pathlib.Path(out)
    counties(out, onea, arcos, wonder)
    m = pd.read_csv(out / "tests_countries.csv")
    x = pd.read_csv(out / "incb_sddd_2000_2014.csv")
    import re
    x["key"] = [re.sub(r"\s+", " ", re.sub(r"[^a-z ,()']", " ", str(c).lower().replace("’", "'").replace("ü", "u").replace("ç", "c")
                                              .replace("é", "e").replace("ô", "o"))).strip() for c in x.country]
    w = x.pivot_table(index="key", columns="window", values="analgesic")
    m = m.merge(w[["2000-2002", "2009-2011"]], left_on="key", right_index=True, how="left")
    m["rx_growth"] = np.log((m["2009-2011"] + 1) / (m["2000-2002"] + 1))
    m["band"] = pd.cut(m.rx_pre, BANDS, labels=LABELS)
    rows = []
    for lab in LABELS:
        k = m[m.band == lab]
        q = k[k.anyill == 0]
        rows.append(dict(band=lab, countries=len(k), with_market=int(k.anyill.sum()), median_gdp=k.gdp.median(),
                         no_market_n=len(q), narc_1921_median=q.narc_1921.median(), narc_1921_max=q.narc_1921.max(),
                         change_median=q.narc_change.median()))
    pd.DataFrame(rows).to_csv(out / "dose_range_bands.csv", index=False)

    res = []
    none = m[m.anyill == 0]
    med = m.gdp.median()
    for name, k in ((f"without a market, {len(none)} countries", none), ("without a market, below the median income", none[none.gdp < med]),
                    ("without a market, prescribing under 5000 S-DDD", none[none.rx_pre <= 5000])):
        rho(res, name, "prescribing with change in mortality", k.rx_pre, k.narc_change)
        rho(res, name, "growth of prescribing with change in mortality", k.rx_growth, k.narc_change)
        rho(res, name, "prescribing with mortality in 2019-21", k.rx_pre, k.narc_1921)
        g = k.rx_growth.dropna()
        res.append(dict(sample=name, test="growth of prescribing, 2000-02 to 2009-11, ratio", kind="median and maximum", estimate=float(np.exp(g.median())),
                        lo=np.nan, hi=float(np.exp(g.max())), p=np.nan, n=len(g)))
        res.append(dict(sample=name, test="prescribing 2000-11, S-DDD", kind="minimum and maximum", estimate=np.nan, lo=float(k.rx_pre.min()),
                        hi=float(k.rx_pre.max()), p=np.nan, n=len(k)))
    k = none.dropna(subset=["rx_growth", "narc_change", "lgdp"])
    r = smf.ols("narc_change ~ rx_growth + lgdp", data=k).fit(cov_type="HC3")
    ci = r.conf_int().loc["rx_growth"]
    res.append(dict(sample=f"without a market, {len(none)} countries", test="change in mortality on growth of prescribing, GDP held", kind="linear regression",
                    estimate=r.params.rx_growth, lo=ci[0], hi=ci[1], p=r.pvalues.rx_growth, n=int(r.nobs)))
    g = pd.read_csv(out / "gbd_check_countries.csv").dropna(subset=["gdp"])
    g = g[(g.gdp < 15000) & (g.anyill == 0)]
    name = "GBD's modelled estimates, GDP per person under 15 000 dollars"
    rho(res, name, "prescribing with change in mortality", g.rx_pre, g.change)
    rho(res, name, "prescribing with mortality in 2019-21", g.rx_pre, g.oud_1921)
    res.append(dict(sample=name, test="prescribing 2000-11, S-DDD", kind="minimum and maximum", estimate=np.nan, lo=float(g.rx_pre.min()),
                    hi=float(g.rx_pre.max()), p=np.nan, n=len(g)))
    res = pd.DataFrame(res)
    res.to_csv(out / "dose_range_tests.csv", index=False)
    r, p = spearmanr(m.rx_pre, m.gdp)
    top = m.nlargest(15, "rx_pre")
    print(f"prescribing with GDP per person, 60 countries: Spearman {r:.2f}; lowest GDP among the 15 heaviest {top.gdp.min():.0f}; median GDP {med:.0f}")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
