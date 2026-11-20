"""Three checks, each on outputs already built.

1. The country interaction of prescribing with illicit supply, with the USA, Canada or both left out (code/06's frame).
2. The triplicate gap adjusted for region and for heroin market type, alone and with arrival, since the three late-arrival triplicate
   states are western or black-tar markets (code/09's frame, suppressed counts at five).
3. The rebuilt triplicate gap in deaths per 100 000, relative to 1990-94, to set beside Alpert et al.'s published estimates.

Models are OLS with HC3 errors, as in code/06 and code/09. Writes out/audit_checks.csv and out/triplicate_levels.csv.

  python3 code/23_audit_checks.py <out folder>
"""
import pathlib
import sys

import pandas as pd
import statsmodels.formula.api as smf

WEST = {"AZ", "CA", "CO", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"}  # the Census Bureau's West, contiguous states
USA, CANADA = "United States of America", "Canada"


def main(out):
    out, rows = pathlib.Path(out), []

    def fit(check, label, d, formula, term):
        r = smf.ols(formula, data=d).fit(cov_type="HC3")
        rows.append(dict(check=check, model=label, term=term, b=r.params[term], se=r.bse[term], p=r.pvalues[term], n=int(r.nobs)))

    m = pd.read_csv(out / "tests_countries.csv")
    for label, drop in (("all countries", []), ("without the USA", [USA]), ("without Canada", [CANADA]), ("without the USA and Canada", [USA, CANADA])):
        fit("country interaction", label, m[~m.country.isin(drop)], "narc_change ~ lrx_pre * anyill + lgdp", "lrx_pre:anyill")

    t = pd.read_csv(out / "alpert_rerun_frame.csv")
    t["west"] = t.state.isin(WEST).astype(int)
    for oc in ("growth_since_launch", "lt404"):
        for label, extra in (("no adjustment", ""), ("arrival date", " + za"), ("western region", " + west"), ("market type", " + white"),
                             ("arrival date and western region", " + za + west"), ("arrival date and market type", " + za + white")):
            fit(f"triplicate {oc}", label, t, f"{oc} ~ triplicate{extra}", "triplicate")
    r = pd.DataFrame(rows)
    r.to_csv(out / "audit_checks.csv", index=False)

    windows = ["1990-94", "1999-01", "2002-04", "2005-07", "2008-10", "2011-13", "2014-16", "2017-19"]
    g = t.groupby("triplicate")[[f"od_{w}" for w in windows]].mean().T
    gap = g[0] - g[1]
    lv = pd.DataFrame({"window": windows, "others": g[0].values, "triplicate": g[1].values, "gap_rel_1990_94": (gap - gap["od_1990-94"]).values})
    lv.to_csv(out / "triplicate_levels.csv", index=False)
    with pd.option_context("display.width", 200):
        print(r.round(3).to_string(index=False))
        print(lv.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
