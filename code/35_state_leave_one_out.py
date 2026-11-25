"""The state model with each state left out in turn (2026-11-19, after code/31's check found DC's weight).

code/24's main state model: each state's synthetic-opioid death rate over the three calendar years after its arrival year, logged, on
log dispensing 2009-11 and arrival date, OLS with HC3 errors, 49 states. Deaths by state and year from WONDER's final file to 2020
and its counts by state and year for 2021-24, as code/24 reads them, with hidden counts at 1, 5 and 9 (code/24 uses 1 and 9). The
model is refitted without each state, and Cook's distance from the unweighted fit shows which states weigh most.

  python3 -I code/35_state_leave_one_out.py <wonder_state_year_t404 date folder> <out folder>

Writes out/state_leave_one_out.csv: the estimate with all 49 and without each state, at each bound, with Cook's distance. Estimates only.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main(d77, out):
    d77, out = pathlib.Path(d77), pathlib.Path(out)
    a = pd.read_csv(d77 / "wonder_t404_state_year_1999_2020.tsv", sep="\t", dtype=str)
    a = a.assign(st=a.state.str.extract(r"\((\d\d)\)")[0].astype(int), year=a.year.str[:4].astype(int),
                 pop=a.population.str.replace(",", "").astype(float), hidden=a.deaths.eq("Suppressed"),
                 d=pd.to_numeric(a.deaths.str.replace(",", ""), errors="coerce"))
    m = pd.read_csv(d77 / "wonder_t404_state_year_2018_2024.tsv", sep="\t", dtype=str)
    m = pd.DataFrame(dict(st=m.state.str.extract(r"\((\d\d)\)")[0].astype(int), year=m.year.str[:4].astype(int),
                          pop=m.population.str.replace(",", "").astype(float), hidden=m.deaths.eq("Suppressed"),
                          d=pd.to_numeric(m.deaths.str.replace(",", ""), errors="coerce")))
    s = pd.read_csv(out / "alpert_rerun_frame.csv").set_index("st")
    s["Y"] = np.floor(s.arrival).astype(int)
    rows = []
    for bound in (1, 5, 9):
        ya = a.assign(y=np.where(a.hidden, bound, a.d))[["st", "year", "y", "pop"]]
        ym = m.assign(y=np.where(m.hidden, bound, m.d))[["st", "year", "y", "pop"]]
        p = pd.concat([ya, ym[ym.year.between(2021, 2024)]])
        s["ly"] = np.log(pd.Series({st: (lambda k: k.y.sum() / k["pop"].sum() * 1e5)(p[(p.st == st) & p.year.between(r.Y + 1, r.Y + 3)])
                                    for st, r in s.iterrows()}))
        cook = pd.Series(smf.ols("ly ~ lrx + za", data=s).fit().get_influence().cooks_distance[0], index=s.index)
        for left in [None] + list(s.index):
            k = s if left is None else s.drop(left)
            r = smf.ols("ly ~ lrx + za", data=k).fit(cov_type="HC3")
            rows.append(dict(bound=bound, left_out="none" if left is None else s.loc[left, "state"], n=int(r.nobs), b=r.params["lrx"],
                             lo=r.conf_int().loc["lrx", 0], hi=r.conf_int().loc["lrx", 1], p=r.pvalues["lrx"],
                             cooks_distance=np.nan if left is None else cook[left],
                             dispensing=np.nan if left is None else np.exp(s.loc[left, "lrx"]),
                             rate_years_1_3=np.nan if left is None else np.exp(s.loc[left, "ly"])))
    r = pd.DataFrame(rows)
    full = pd.read_csv(out / "fixed_span.csv").query("span == 'years 1-3 after the arrival year' and model == 'dispensing, arrival date'")
    for b in (1, 9):
        assert np.isclose(r.query("bound == @b and left_out == 'none'").b.iloc[0], full[full.hidden == b].b.iloc[0]), "does not reproduce code/24"
    r.to_csv(out / "state_leave_one_out.csv", index=False)
    for b in (1, 5, 9):
        k = r[(r.bound == b) & (r.left_out != "none")].sort_values("cooks_distance", ascending=False)
        print(b, r[(r.bound == b) & (r.left_out == "none")][["b", "lo", "hi"]].round(3).values, "\n", k.head(3).round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
