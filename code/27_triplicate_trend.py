"""The triplicate gap with a pre-launch trend, and with 1990-94 as the reference window.

code/09's panel compares the triplicate states with the rest window by window, relative to 1995-98. That window contains OxyContin's
launch in 1996, and the gap in 1990-94 is 0.15 (0.02 to 0.27) with hidden counts at one. This script fits two further panels on
code/09's data, at each bound:

1. A triplicate-specific linear trend in window midyear, identified from the three windows up to 1995-98, with separate gaps for
   each later window: each later gap is then measured against the extended pre-launch trend.
2. code/09's event panel with 1990-94 as the reference window instead of 1995-98.

State and window fixed effects, errors clustered by state, as in code/09. Writes out/triplicate_trend.csv (estimates only).

  python3 code/27_triplicate_trend.py <inputs/1a_out folder> <arcos_county_year.csv> <wonder_pooled date folder> <out folder>
  python3 code/27_triplicate_trend.py --selftest
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = pathlib.Path(__file__).resolve().parent
PRE = ("1985-89", "1990-94", "1995-98")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def models(long, bound):
    mid = {"1985-89": 1987, "1990-94": 1992, "1995-98": 1996.5, "1999-01": 2000, "2002-04": 2003, "2005-07": 2006, "2008-10": 2009,
           "2011-13": 2012, "2014-16": 2015, "2017-19": 2018}
    long = long.copy()
    long["t"] = long.window.map(mid) - 1996.5
    long["trip_trend"] = long.triplicate * long.t
    rows = []
    post = [w for w in mid if w not in PRE]
    for w in post:
        long[f"trip_{w[:4]}"] = long.triplicate * (long.window == w)
    f = "lod ~ C(st) + C(window) + trip_trend + " + " + ".join(f"trip_{w[:4]}" for w in post)
    r = smf.ols(f, data=long).fit(cov_type="cluster", cov_kwds={"groups": long.st})
    ci = r.conf_int()
    for term in ["trip_trend"] + [f"trip_{w[:4]}" for w in post]:
        rows.append(dict(model="gap against the extended pre-launch trend", bound=bound, term=term, b=r.params[term], lo=ci.loc[term, 0],
                         hi=ci.loc[term, 1], p=r.pvalues[term], n=int(r.nobs)))
    other = [w for w in mid if w != "1990-94"]
    for w in other:
        long[f"g_{w[:4]}"] = long.triplicate * (long.window == w)
    f = "lod ~ C(st) + C(window) + " + " + ".join(f"g_{w[:4]}" for w in other)
    r = smf.ols(f, data=long).fit(cov_type="cluster", cov_kwds={"groups": long.st})
    ci = r.conf_int()
    for w in other:
        term = f"g_{w[:4]}"
        rows.append(dict(model="gap by window, relative to 1990-94", bound=bound, term=term, b=r.params[term], lo=ci.loc[term, 0],
                         hi=ci.loc[term, 1], p=r.pvalues[term], n=int(r.nobs)))
    return rows


def simulated(seed=8):
    rng = np.random.default_rng(seed)
    mid = [1987, 1992, 1996.5, 2000, 2003, 2006, 2009, 2012, 2015, 2018]
    names = ["1985-89", "1990-94", "1995-98", "1999-01", "2002-04", "2005-07", "2008-10", "2011-13", "2014-16", "2018-19"]
    names[-1] = "2017-19"
    rows = []
    for st in range(1, 50):
        trip = int(st <= 5)
        a = rng.normal(0, 0.3)
        for w, t in zip(names, mid):
            # a pre-launch trend of +0.01 a year in triplicate states and a gap of -0.4 after 1999, against that trend
            g = trip * (0.01 * (t - 1996.5) + (-0.4 if t >= 2000 else 0))
            rows.append(dict(st=st, triplicate=trip, window=w, lod=a + 0.05 * (t - 1987) + g + rng.normal(0, 0.05)))
    return pd.DataFrame(rows)


def main(onea, arcos=None, wonder=None, outdir=None):
    if onea == "--selftest":
        r = pd.DataFrame(models(simulated(), 0))
        t = r[r.term == "trip_trend"].b.iloc[0]
        g = r[(r.model.str.startswith("gap against")) & (r.term == "trip_2017")].b.iloc[0]
        assert abs(t - 0.01) < 0.01 and abs(g + 0.4) < 0.1, (t, g)
        print(r.round(3).to_string())
        print("selftest passed")
        return
    us = load("us", "07_us_states.py")
    a9 = load("a9", "09_alpert_rerun.py")
    wonder, outdir, rows = pathlib.Path(wonder), pathlib.Path(outdir), []
    for bound in (1, 5, 9):
        s = us.state_frame(onea, arcos, wonder, bound).set_index("st")
        for lab, (f, _) in a9.WINDOWS.items():
            d = us.read_wonder(wonder / f"{f}_county.tsv", bound).groupby("st")[["deaths", "population"]].sum()
            s[f"od_{lab}"] = d.deaths / d.population * 1e5
        s = s.reset_index()
        long = s.melt(id_vars=["st", "triplicate"], value_vars=[f"od_{k}" for k in a9.WINDOWS], var_name="window", value_name="od")
        long["window"] = long.window.str[3:]
        long["lod"] = np.log(long.od)
        rows += models(long, bound)
    r = pd.DataFrame(rows)
    r.to_csv(outdir / "triplicate_trend.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:5])
