"""Check: CDC WONDER's counts by underlying cause leave out drug
poisonings certified under a cardiovascular underlying cause (I00-I99) with the poisoning listed as contributing (Fang et al. 2026). The
manuscript gives the national count. This asks, for code/07's 49 states, how large the missed share is by state, and whether it runs with
prescribing in 2009-11 or with fentanyl's arrival, which is what could bias the state comparisons. It then adds the cardiovascular-coded
synthetic-opioid deaths of 2014-20 back to code/07's synthetic-opioid rate and refits Table 2's two prescribing models.

Inputs: the wonder_cvd_drug tables (D77 by state of residence; see its downloads.tsv) and code/07's bound-5 frame. WONDER lists a
state's cardiovascular-coded count only when it is 10 or more. States it leaves out are dropped from the shares and counted; in the add-back
they enter at 0 and at 9, as bounds. Nothing here writes a count, only shares, correlations and estimates.

  python3 code/20_cardiac_coding_check.py <wonder_cvd_drug date folder> <wonder_pooled date folder> <out folder>
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("us7", HERE / "07_us_states.py")
us7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(us7)

TABLES = {"all drug poisonings, 1999-2020": ("cvd_x4044_1999_2020", "od_1999_2020"),
          "synthetic opioids, 2014-20": ("cvd_t404_2014_2020", "od_t404_2014_2020")}


def read(d, name):
    """A WONDER table saved as state_code and deaths (and sometimes state); suppressed or missing counts become NaN."""
    t = pd.read_csv(d / f"{name}.tsv", sep="\t", dtype={"state_code": str})
    t = t[t.state_code.str.fullmatch(r"\d+", na=False)]   # drops WONDER's Total row
    return pd.to_numeric(t.deaths, errors="coerce").set_axis(t.state_code.astype(int)).rename(name)


def main(cvd_dir, wonder, outdir):
    cvd_dir, wonder, outdir = pathlib.Path(cvd_dir), pathlib.Path(wonder), pathlib.Path(outdir)
    s = pd.read_csv(outdir / "us_states_frame.csv")
    s["st"] = s.st.astype(int)
    s = s.set_index("st")
    rows = []
    for lab, (cname, oname) in TABLES.items():
        c, o = read(cvd_dir, cname).reindex(s.index), read(cvd_dir, oname).reindex(s.index)
        assert o.notna().all(), f"{oname} lacks a state"
        share = (c / (c + o)).dropna()
        for against, x in (("prescribing 2009-11", s.rx), ("arrival date", s.arrival), ("distress", s.distress)):
            rho = spearmanr(share, x.reindex(share.index))
            rows.append(dict(deaths=lab, states_listed=len(share), states_left_out=int(c.isna().sum()),
                             share_listed_states=c.sum() / (c.sum() + o[c.notna()].sum()), share_min=share.min(), share_median=share.median(),
                             share_max=share.max(), against=against, spearman=rho[0], p=rho[1]))
    shares = pd.DataFrame(rows)
    # the add-back: code/07's synthetic-opioid deaths (county files, suppressed counties at 5) plus the cardiovascular-coded ones
    t = us7.read_wonder(wonder / "mcd_t404_2014_2020_county.tsv", 5).groupby("st")[["deaths", "population"]].sum().reindex(s.index)
    assert np.allclose(t.deaths / t.population * 1e5, s.t404_2014_20)
    c = read(cvd_dir, "cvd_t404_2014_2020").reindex(s.index)
    est = []
    for fill in (0, 9):
        k = s.assign(lt404_cv=np.log((t.deaths + c.fillna(fill)) / t.population * 1e5))
        for y in ("lt404", "lt404_cv"):
            for f in (f"{y} ~ lrx + za", f"{y} ~ lrx + za + zx + white"):
                r = smf.ols(f, data=k).fit(cov_type="HC3")
                ci = r.conf_int()
                est.append(dict(left_out_states_at=fill, outcome="published" if y == "lt404" else "with cardiovascular-coded deaths",
                                formula=f, b=r.params["lrx"], lo=ci.loc["lrx", 0], hi=ci.loc["lrx", 1], p=r.pvalues["lrx"]))
        added = (c.fillna(fill) / (t.deaths + c.fillna(fill)))
        est.append(dict(left_out_states_at=fill, outcome="share added back", formula="median and largest across states",
                        b=added.median(), lo=np.nan, hi=added.max(), p=np.nan))
    est = pd.DataFrame(est).drop_duplicates(subset=["left_out_states_at", "outcome", "formula"])
    shares.to_csv(outdir / "cardiac_coding_shares.csv", index=False)
    est.to_csv(outdir / "cardiac_coding_addback.csv", index=False)
    with pd.option_context("display.width", 220):
        print(shares.round(4).to_string(index=False))
        print()
        print(est.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
