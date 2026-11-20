"""Alpert et al.'s triplicate comparison, rerun on state overdose deaths with fentanyl's arrival held.

Alpert, Evans, Lieber and Powell (2022; NBER WP 26500, pp. 14, 22) compare the five states with triplicate prescription programs when
OxyContin launched in 1996 (California, Idaho, Illinois, New York, Texas) with the rest: less OxyContin marketing, and slower overdose
growth for twenty years. On the companion county study's clock, three of the five got illicit fentanyl late. This asks how much of their gap in the fentanyl
years is arrival timing.

- Deaths: CDC WONDER county files (`wonder_pooled`), summed to states, with Alpert et al.'s own definitions: ICD-9 E850-E858,
  E950.0-E950.5, E962.0, E980.0-E980.5 for 1985-98 (Compressed Mortality) and ICD-10 X40-X44, X60-X64, X85, Y10-Y14 from 1999. Windows:
  1985-89, 1990-94, 1995-98, 1999-2001, 2002-04, 2005-07, 2008-10, 2011-13, 2014-16, 2017-19; synthetic opioids (T40.4) in 2014-20.
  Suppressed counties enter at 1, 5 or 9. The ICD-9 to ICD-10 break in 1999 shifts every state's level, so comparisons are between states.
- Arrival: the companion county study's half-year clock; distress and the white-powder market flag from the companion county study, as in `code/07`.

Models: (1) the triplicate gap by window, relative to 1995-98, with state and window effects (errors clustered by state); (2) the same with
an indicator for windows after the state's fentanyl arrival; (3) cross-state regressions of growth and of the fentanyl wave on triplicate,
with and without arrival.

  python3 code/09_alpert_rerun.py <inputs/1a_out folder> <arcos_county_year.csv> <wonder_pooled date folder> <out folder>
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("us", HERE / "07_us_states.py")
us = importlib.util.module_from_spec(spec)
spec.loader.exec_module(us)

WINDOWS = {"1985-89": ("cmf_od_1985_1989", 1987), "1990-94": ("cmf_od_1990_1994", 1992), "1995-98": ("cmf_od_1995_1998", 1996.5),
           "1999-01": ("ucd_od_1999_2001", 2000), "2002-04": ("ucd_od_2002_2004", 2003), "2005-07": ("ucd_od_2005_2007", 2006),
           "2008-10": ("ucd_od_2008_2010", 2009), "2011-13": ("ucd_od_2011_2013", 2012), "2014-16": ("ucd_od_2014_2016", 2015),
           "2017-19": ("ucd_od_2017_2019", 2018)}


def fit(d, formula, label, out, bound, cluster=None):
    kw = dict(cov_type="cluster", cov_kwds={"groups": d[cluster]}) if cluster else dict(cov_type="HC3")
    r = smf.ols(formula, data=d).fit(**kw)
    for term in r.params.index:
        if term.startswith("trip") or term in ("za", "post"):
            out.append(dict(model=label, formula=formula, bound=bound, n=int(r.nobs), term=term, b=r.params[term], se=r.bse[term], p=r.pvalues[term], r2=r.rsquared))


def main(onea, arcos, wonder, outdir):
    wonder, outdir, res = pathlib.Path(wonder), pathlib.Path(outdir), []
    for bound in (1, 5, 9):
        s = us.state_frame(onea, arcos, wonder, bound).set_index("st")
        for lab, (f, _) in WINDOWS.items():
            d = us.read_wonder(wonder / f"{f}_county.tsv", bound).groupby("st")[["deaths", "population"]].sum()
            s[f"od_{lab}"] = d.deaths / d.population * 1e5
        s = s.reset_index()
        # (1) and (2): the gap path, base 1995-98
        long = s.melt(id_vars=["st", "triplicate", "arrival"], value_vars=[f"od_{k}" for k in WINDOWS], var_name="window", value_name="od")
        long["window"] = long.window.str[3:]
        long["lod"] = np.log(long.od)
        long["post"] = (long.window.map({k: v[1] for k, v in WINDOWS.items()}) >= long.arrival).astype(int)
        terms = []
        for w in WINDOWS:
            if w != "1995-98":
                name = f"trip_{w[:4]}"
                long[name] = long.triplicate * (long.window == w)
                terms.append(name)
        base = "lod ~ C(st) + C(window) + " + " + ".join(terms)
        fit(long, base, "gap by window, relative to 1995-98", res, bound, cluster="st")
        fit(long, base + " + post", "gap by window, arrival held", res, bound, cluster="st")
        # (3) cross-state
        s["growth_since_launch"] = np.log(s["od_2017-19"] / s["od_1995-98"])
        s["growth_rx_era"] = np.log(s["od_2011-13"] / s["od_1995-98"])
        s["growth_fent_era"] = np.log(s["od_2017-19"] / s["od_2011-13"])
        for oc, lab in (("growth_rx_era", "growth 1995-98 to 2011-13 (before fentanyl)"), ("growth_fent_era", "growth 2011-13 to 2017-19"),
                        ("growth_since_launch", "growth 1995-98 to 2017-19"), ("lt404", "log synthetic-opioid rate 2014-20")):
            fit(s, f"{oc} ~ triplicate", f"{lab} | triplicate", res, bound)
            fit(s, f"{oc} ~ triplicate + za", f"{lab} | triplicate, arrival held", res, bound)
            fit(s, f"{oc} ~ triplicate + za + zx + white", f"{lab} | triplicate, arrival, distress, market", res, bound)
        if bound == 5:
            s.to_csv(outdir / "alpert_rerun_frame.csv", index=False)
    r = pd.DataFrame(res)
    r.to_csv(outdir / "alpert_rerun_results.csv", index=False)
    s = pd.read_csv(outdir / "alpert_rerun_frame.csv")
    print("Triplicate states:", ", ".join(s.loc[s.triplicate == 1, "state"]), "; arrival", s.loc[s.triplicate == 1, "arrival"].tolist())
    k = r[(r.bound == 5)]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(k[["model", "n", "term", "b", "se", "p"]].round(3).to_string(index=False))
        print("\nTriplicate terms at each bound:")
        print(r[r.term.str.startswith("trip")].pivot_table(index=["model", "term"], columns="bound", values="b").round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
