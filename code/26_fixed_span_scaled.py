"""code/24's state models with each window scaled by the national rate in the same calendar years, and what the estimates rule out
().

A state reached in 2014 has its window in 2015-17 and one reached in 2020 in 2021-23, when national synthetic-opioid mortality was
several times higher. code/24's models that hold the arrival date absorb a linear calendar trend, since the window is the arrival
year plus one to three, but not the curved national path. Here each state's window rate is divided by the national rate over the
same three calendar years, the national rate being the deaths of the 49 states summed over their population, and code/24's models
are fitted again on the log of that ratio. Hidden counts of 1 to 9 enter at 1 and at 9, as in code/24; population for 2021-24 is
each state's 2020 population, as in code/24, unless the monthly pull carries yearly population.

For each estimate of log dispensing it also gives what the interval allows across the middle half of states: the 75th percentile
of dispensing over the 25th, raised to the estimate and to each end of its interval, the ratio of mortality that a state at the
75th percentile would have against one at the 25th.

Reads the same inputs as code/24; writes out/fixed_span_scaled.csv (estimates only, no counts).

  python3 code/26_fixed_span_scaled.py <wonder_state_year_t404 date folder> <wonder_state_month date folder> <out folder>
  python3 code/26_fixed_span_scaled.py --selftest
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fs", HERE / "24_fixed_span.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
MODELS = fs.MODELS


def panel(d77, d176, out, bound):
    """Deaths and population by state and year, 1999-2024, exactly as code/24 builds them."""
    a = pd.read_csv(d77 / "wonder_t404_state_year_1999_2020.tsv", sep="\t", dtype=str)
    a["st"] = a.state.str.extract(r"\((\d\d)\)")[0].astype(int)
    a["year"] = a.year.str[:4].astype(int)
    a["pop"] = a.population.str.replace(",", "").astype(float)
    a["hidden"] = a.deaths.eq("Suppressed")
    a["d"] = pd.to_numeric(a.deaths.str.replace(",", ""), errors="coerce")
    m = pd.read_csv(d176 / "wonder_stage1_opioid.tsv", sep="\t", dtype=str)
    m = m[m["Month Code"].notna() & m["Residence State (without Puerto Rico) Code"].notna()]
    m = pd.DataFrame(dict(st=m["Residence State (without Puerto Rico) Code"].astype(int), year=m["Month Code"].str[:4].astype(int),
                          hidden=m.Deaths.eq("Suppressed"), d=pd.to_numeric(m.Deaths, errors="coerce")))
    ya = a.assign(y=a.d.fillna(bound).where(~a.hidden, bound))[["st", "year", "y", "pop"]]
    ym = m.assign(y=m.d.where(~m.hidden, bound)).groupby(["st", "year"], as_index=False).y.sum()
    pop20 = ya[ya.year == 2020].set_index("st")["pop"]
    return pd.concat([ya, ym[ym.year.between(2021, 2024)].assign(pop=lambda t: t.st.map(pop20))])


def fit(s, p, bound):
    p = p[p.st.isin(s.index)].set_index(["st", "year"]).sort_index()
    nat = p.groupby(level="year")[["y", "pop"]].sum()

    def rate(st, y0, y1):
        k = p.loc[(st, slice(y0, y1)), :]
        return k.y.sum() / k["pop"].sum()

    def national(y0, y1):
        k = nat.loc[y0:y1]
        return k.y.sum() / k["pop"].sum()

    q = s.lrx.quantile([0.25, 0.75])
    iqr = float(np.exp(q[0.75] - q[0.25]))
    rows = []
    spans = {"years 1-3 after the arrival year": (1, 3), "years 2-4 after the arrival year": (2, 4)}
    for span, (a0, a1) in spans.items():
        for scaled in (False, True):
            s = s.assign(ly=[np.log(rate(st, y + a0, min(y + a1, 2024)) / (national(y + a0, min(y + a1, 2024)) if scaled else 1e-5))
                             for st, y in zip(s.index, s.Y)])
            for label, formula, term in MODELS:
                r = smf.ols(formula, data=s).fit(cov_type="HC3")
                lo, hi = r.conf_int().loc[term]
                row = dict(span=span, hidden=bound, scaled_by_national_rate=scaled, model=label, term=term, b=r.params[term], lo=lo, hi=hi,
                           p=r.pvalues[term], n=int(r.nobs), r2=r.rsquared)
                if term == "lrx":
                    row.update(iqr_ratio=iqr, wave_ratio=iqr ** r.params[term], wave_ratio_lo=iqr ** lo, wave_ratio_hi=iqr ** hi,
                               excludes_elasticity_1=hi < 1)
                rows.append(row)
    return rows


def simulated(seed=8):
    rng = np.random.default_rng(seed)
    st = np.arange(1, 50)
    arrival = rng.choice(np.arange(2014.25, 2021.5, 0.5), 49)
    s = pd.DataFrame(dict(st=st, state=[f"S{i}" for i in st], arrival=arrival, lrx=np.log(rng.uniform(37, 143, 49)),
                          triplicate=(st <= 5).astype(int), white=(st % 2).astype(int)))
    s["za"] = -(s.arrival - s.arrival.mean()) / s.arrival.std()
    s["zx"] = rng.normal(0, 1, 49)
    s["Y"] = np.floor(s.arrival).astype(int)
    years = np.arange(1999, 2025)
    national = np.interp(years, [1999, 2013, 2016, 2022, 2024], [0.3, 1.0, 6, 22, 15])
    rows = []
    for _, r in s.iterrows():
        for y in years:
            lam = national[y - 1999] * (1 + 0.5 * r.white) * 50 * (0.2 if y <= r.Y else 1)
            rows.append(dict(st=r.st, year=y, y=float(rng.poisson(lam)), pop=5e6))
    return s.set_index("st"), pd.DataFrame(rows)


def main(d77, d176=None, out=None):
    if d77 == "--selftest":
        s, p = simulated()
        r = pd.DataFrame(fit(s, p, 1))
        x = r[(r.term == "white") & r.scaled_by_national_rate & r.span.str.startswith("years 1")].b.iloc[0]
        assert abs(x - np.log(1.5)) < 0.25, x  # the simulated market effect survives scaling
        u = r[(r.term == "za") & ~r.scaled_by_national_rate & (r.model == "arrival date")].b.iloc[0]
        v = r[(r.term == "za") & r.scaled_by_national_rate & (r.model == "arrival date")].b.iloc[0]
        assert abs(v) < abs(u), (u, v)  # scaling removes the calendar effect that the raw window rate carries
        print(r.round(3).to_string())
        print("selftest passed")
        return
    d77, d176, out = pathlib.Path(d77), pathlib.Path(d176), pathlib.Path(out)
    s = pd.read_csv(out / "alpert_rerun_frame.csv").set_index("st")
    s["Y"] = np.floor(s.arrival).astype(int)
    rows = []
    for bound in (1, 9):
        rows += fit(s, panel(d77, d176, out, bound), bound)
    r = pd.DataFrame(rows)
    r.to_csv(out / "fixed_span_scaled.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
