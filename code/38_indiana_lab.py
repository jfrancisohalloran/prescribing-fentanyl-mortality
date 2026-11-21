"""Inside Indiana, did counties that prescribed more before 2012 later get fentanyl sooner, or in a larger share of their cases?

The county test of code/33 reads the question from deaths. This check reads it from the supply, in the one state whose police
laboratories publish every drug identified by county and day: the Indiana State Police file of submissions, 2007 to March 2021. The
models and the rules for reading them were written down before the county figures were computed.
The rows for drugs other than opioids were added afterwards, to see whether dispensing marks every kind of later case.

Data:
- The laboratory file, rows with a submit date whose drug is not "Withdrawn". A case is one masked laboratory number. Marion County is
  left out, since Indianapolis's own laboratory handles most of its cases. Fentanyl is any name containing "fentan".
- Exposures, each logged and scaled to one standard deviation: E1 CDC's county dispensing rate, 2009-11 (`rx` in county_base.csv,
  counties with `rx_missing` = 0); E2 oxycodone and hydrocodone units per person, 2006-12, from the ARCOS county-year file; E3 cases
  holding hydrocodone or oxycodone per 10,000 residents, 2007-11, as log(1 + rate).
- Held in every model: the log of all cases per 10,000 residents in 2007-11, and log population.

Models, each with each exposure alone:
- O1 share: fentanyl cases over heroin-or-fentanyl cases, 2017-20; binomial, robust errors.
- O2 timing: first year from 2014 with two or more fentanyl cases, censored after 2020; Cox model.
- O3 and O4: heroin cases 2012-16 and fentanyl cases 2017-20; Poisson with log population as offset, robust errors.
- O1 and O2 again for counties with 10 or more heroin-or-fentanyl cases in 2017-20.
- Other drugs: the Poisson model for methamphetamine, cocaine, marijuana, alprazolam, all cases, and heroin or fentanyl.

Writes out/indiana_lab_results.csv: coefficients and counts of counties only.

  /opt/homebrew/bin/python3.13 -I code/38_indiana_lab.py <indiana csv> <county_base.csv> <arcos_county_year.csv> <out folder>
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

CASE = "LAB_NUMBER_HASHED"


def rows_of(ind):
    d = pd.read_csv(ind, dtype=str)
    d["date"] = pd.to_datetime(d["SUBMIT_DATE"], format="%m/%d/%Y", errors="coerce")
    d = d[d["date"].notna() & ~d["DRUG"].fillna("").str.contains("Withdrawn")]
    d = d[d["STATE_FIPS"] == "18"].copy()
    d["fips"] = "18" + d["COUNTY_FIPS"].str.zfill(3)
    d = d[d["fips"] != "18097"].copy()
    d["year"] = d["date"].dt.year
    name = d["DRUG"].str.lower()
    d["fent"] = name.str.contains("fentan")
    d["heroin"] = name.eq("heroin")
    d["pill"] = name.str.contains("hydrocodone") | name.str.contains("oxycodone")
    d["hf"] = d["fent"] | d["heroin"]
    d["every"] = True
    for col, pat in (("meth", "methamphetamine"), ("marijuana", "marijuana"), ("alprazolam", "alprazolam")):
        d[col] = name.eq(pat)
    d["cocaine"] = name.str.contains("cocaine")
    return d


def cases(d, col, y0, y1):
    s = d[d[col] & d["year"].between(y0, y1)]
    return s.groupby("fips")[CASE].nunique()


def panel(d, base, arcos):
    c = pd.DataFrame({"all_0711": cases(d, "every", 2007, 2011), "pill_0711": cases(d, "pill", 2007, 2011),
                      "heroin_1216": cases(d, "heroin", 2012, 2016), "fent_1720": cases(d, "fent", 2017, 2020),
                      "hf_1720": cases(d, "hf", 2017, 2020)}).fillna(0)
    c["pop"] = d.groupby("fips")["COUNTY_POP"].first().astype(float)
    fy = d[d["fent"] & d["year"].between(2014, 2020)].groupby(["fips", "year"])[CASE].nunique()
    c["first"] = fy[fy >= 2].reset_index().groupby("fips")["year"].min()
    c["event"] = c["first"].notna().astype(int)
    c["t"] = c["first"].fillna(2021) - 2013
    b = pd.read_csv(base, dtype={"fips": str})
    b["fips"] = b["fips"].str.zfill(5)
    c = c.join(b.set_index("fips")[["rx", "rx_missing"]])
    a = pd.read_csv(arcos, dtype={"fips": str})
    a = a[a["year"].between(2006, 2012)].groupby("fips")[["pills_oxy_hydro", "population_wapo"]].sum()
    c["arcos"] = a["pills_oxy_hydro"] / a["population_wapo"]
    c["E1"] = np.where(c["rx_missing"] == 0, np.log(c["rx"]), np.nan)
    c["E2"] = np.log(c["arcos"])
    c["E3"] = np.log1p(c["pill_0711"] / c["pop"] * 1e4)
    c["flow"] = np.log(c["all_0711"] / c["pop"] * 1e4)
    c["lpop"] = np.log(c["pop"])
    return c


def line(sample, e, outcome, model, n, b, se, note=""):
    return dict(sample=sample, exposure=e, outcome=outcome, model=model, n=n, b=b, lo=b - 1.96 * se, hi=b + 1.96 * se, ratio=np.exp(b),
                ratio_lo=np.exp(b - 1.96 * se), ratio_hi=np.exp(b + 1.96 * se), note=note)


def fits(c, d, tag):
    rows = []
    for e in ("E1", "E2", "E3"):
        s = c[c[e].notna() & np.isfinite(c["flow"])].copy()
        s["z"] = (s[e] - s[e].mean()) / s[e].std()
        X = sm.add_constant(s[["z", "flow", "lpop"]])
        s1 = s[s["hf_1720"] > 0]
        r = sm.GLM(np.column_stack([s1["fent_1720"], s1["hf_1720"] - s1["fent_1720"]]), X.loc[s1.index],
                   family=sm.families.Binomial()).fit(cov_type="HC1")
        rows.append(line(tag, e, "O1 share", "binomial, odds ratio", len(s1), r.params["z"], r.bse["z"]))
        r = sm.PHReg(s["t"], s[["z", "flow", "lpop"]], status=s["event"], ties="efron").fit()
        rho, p = spearmanr(s["z"], s["t"])
        rows.append(line(tag, e, "O2 timing", "Cox, hazard ratio", len(s), r.params[0], r.bse[0],
                         f"events {int(s['event'].sum())}; Spearman with first year {rho:.2f} p {p:.3f}"))
        if tag != "all":
            continue
        for out, col in (("O3 heroin 2012-16", "heroin_1216"), ("O4 fentanyl 2017-20", "fent_1720")):
            r = sm.GLM(s[col], X, family=sm.families.Poisson(), offset=s["lpop"]).fit(cov_type="HC1")
            rows.append(line(tag, e, out, "Poisson, rate ratio", len(s), r.params["z"], r.bse["z"]))
        for g, col in (("methamphetamine", "meth"), ("cocaine", "cocaine"), ("marijuana", "marijuana"), ("alprazolam", "alprazolam"),
                       ("all cases", "every"), ("heroin or fentanyl", "hf")):
            for y0, y1 in ((2012, 2016), (2017, 2020)):
                n = cases(d, col, y0, y1).reindex(s.index).fillna(0)
                r = sm.GLM(n, X, family=sm.families.Poisson(), offset=s["lpop"]).fit(cov_type="HC1")
                rows.append(line(tag, e, f"other: {g} {y0}-{str(y1)[2:]}", "Poisson, rate ratio", len(s), r.params["z"], r.bse["z"]))
    return rows


def main(ind, base, arcos, outdir):
    d = rows_of(ind)
    c = panel(d, base, arcos)
    r = pd.DataFrame(fits(c, d, "all") + fits(c[c["hf_1720"] >= 10], d, "10+ cases"))
    r.to_csv(pathlib.Path(outdir) / "indiana_lab_results.csv", index=False)
    print(r[["sample", "exposure", "outcome", "n", "ratio", "ratio_lo", "ratio_hi"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
