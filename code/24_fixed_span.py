"""The state models on the same span after each state's own arrival.

A rate pooled over 2014-20 gives a state reached in 2019 two fentanyl years and a state reached in 2014 seven, so the arrival date's
hold on pooled rates is mostly years of exposure. Here each state's synthetic-opioid death rate is taken over the three calendar years
after its arrival year, and over years two to four, and the models of code/07 and code/09 are fitted again beside the pooled rate.

Deaths with a drug-poisoning underlying cause and T40.4 among the multiple causes, by state of residence and year: 1999-2020 from CDC
WONDER's Multiple Cause of Death file, and 2021-2024 from WONDER's counts by state and year in the provisional file, final through
2024, both among the downloaded sources folder `wonder_state_year_t404`. Counts of 1 to 9 that WONDER hides enter at 1 and at 9; none falls in
2021-2024 for these states. Each year's rate uses that year's population. States, prescribing, arrival, distress, market type and triplicate
status are those of code/09's frame. OLS with HC3 errors. Writes out/fixed_span.csv: estimates only, no counts.

With a fourth argument, the file of arrival dates with the standard deviation of each date under resampling of the state's
laboratory reports (columns state, arr_p1_persist_sd), the arrival and dispensing models are fitted again without the states whose
standard deviation exceeds a year, and out/fixed_span_certain.csv is written.

  python3 code/24_fixed_span.py <wonder_state_year_t404 date folder> <out folder> [arrival_dates_uncertainty.csv]
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MODELS = [("arrival date", "ly ~ za", "za"), ("market type", "ly ~ white", "white"), ("dispensing", "ly ~ lrx", "lrx"),
          ("dispensing, arrival date", "ly ~ lrx + za", "lrx"), ("dispensing, arrival date, distress, market type", "ly ~ lrx + za + zx + white", "lrx"),
          ("triplicate", "ly ~ triplicate", "triplicate"), ("triplicate, arrival date", "ly ~ triplicate + za", "triplicate")]


def main(d77, out, uncertainty=None):
    d77, out = pathlib.Path(d77), pathlib.Path(out)
    a = pd.read_csv(d77 / "wonder_t404_state_year_1999_2020.tsv", sep="\t", dtype=str)
    a["st"] = a.state.str.extract(r"\((\d\d)\)")[0].astype(int)
    a["year"] = a.year.str[:4].astype(int)
    a["pop"] = a.population.str.replace(",", "").astype(float)
    a["hidden"] = a.deaths.eq("Suppressed")
    a["d"] = pd.to_numeric(a.deaths.str.replace(",", ""), errors="coerce")
    m = pd.read_csv(d77 / "wonder_t404_state_year_2018_2024.tsv", sep="\t", dtype=str)
    m = pd.DataFrame(dict(st=m.state.str.extract(r"\((\d\d)\)")[0].astype(int), year=m.year.str[:4].astype(int),
                          pop=m.population.str.replace(",", "").astype(float), hidden=m.deaths.eq("Suppressed"),
                          d=pd.to_numeric(m.deaths.str.replace(",", ""), errors="coerce")))
    s = pd.read_csv(out / "alpert_rerun_frame.csv").set_index("st")
    s["Y"] = np.floor(s.arrival).astype(int)
    rows, certain = [], []
    if uncertainty:
        sd = pd.read_csv(uncertainty).set_index("state").arr_p1_persist_sd
        s["date_sd"] = s.state.map(sd)
        assert s.date_sd.notna().all(), "a state has no standard deviation for its date"
    for bound in (1, 9):
        ya = a.assign(y=a.d.fillna(bound).where(~a.hidden, bound))[["st", "year", "y", "pop"]]
        ym = m.assign(y=m.d.where(~m.hidden, bound))[["st", "year", "y", "pop"]]
        p = pd.concat([ya, ym[ym.year.between(2021, 2024)]]).set_index(["st", "year"]).sort_index()

        def rate(st, y0, y1):
            k = p.loc[(st, slice(y0, y1)), :]
            return k.y.sum() / k["pop"].sum() * 1e5

        spans = {"2014-20 pooled": lambda r: rate(r.name, 2014, 2020), "years 1-3 after the arrival year": lambda r: rate(r.name, r.Y + 1, r.Y + 3),
                 "years 2-4 after the arrival year": lambda r: rate(r.name, r.Y + 2, min(r.Y + 4, 2024))}
        for span, fn in spans.items():
            s["ly"] = np.log(s.apply(fn, axis=1).astype(float))
            for label, formula, term in MODELS:
                r = smf.ols(formula, data=s).fit(cov_type="HC3")
                rows.append(dict(span=span, hidden=bound, model=label, term=term, b=r.params[term], se=r.bse[term], p=r.pvalues[term],
                                 n=int(r.nobs), r2=r.rsquared))
                if uncertainty and label in ("arrival date", "dispensing, arrival date") and not span.startswith("years 2"):
                    k = s[s.date_sd <= 1]
                    r = smf.ols(formula, data=k).fit(cov_type="HC3")
                    certain.append(dict(span=span, hidden=bound, model=label, term=term, b=r.params[term], se=r.bse[term], p=r.pvalues[term],
                                        n=int(r.nobs), r2=r.rsquared, median_sd=s.date_sd.median(), states_sd_above_1=int((s.date_sd > 1).sum()),
                                        left_out=" ".join(sorted(s.state[s.date_sd > 1]))))
    r = pd.DataFrame(rows)
    r.to_csv(out / "fixed_span.csv", index=False)
    if uncertainty:
        pd.DataFrame(certain).to_csv(out / "fixed_span_certain.csv", index=False)
        print(pd.DataFrame(certain).round(3).to_string(index=False))
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
