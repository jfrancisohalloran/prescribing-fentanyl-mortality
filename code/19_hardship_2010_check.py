"""Check: is an economic-distress index pegged to 2000 too early?

In code/07, distress is the companion county study's county index from the 2000 census, averaged to states with 2008-10 person-year weights. This rebuilds the
state average from the 2008-2012 index, which is the same first principal component of the same six measures, from the American Community
Survey's 2008-2012 summary files and SAIPE 2010 for the same 3,055 counties (folder acs_county_2012). It then refits each of code/07's models
that adjust for distress, at each suppression bound: with the 2000 index as published, with the 2008-2012 index, and with both, which also
holds the change between them. The bound-5 frame is code/07's own `us_states_frame.csv`; bounds 1 and 9 recompute its death rates from the
same WONDER county files.

  python3 code/19_hardship_2010_check.py <inputs/1a_out folder> <wonder_pooled date folder> <index_2010.csv> <out folder>
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("us7", HERE / "07_us_states.py")
us7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(us7)

DISTRESS = {"2000 index": "zx", "2008-2012 index": "zx10", "both": "zx + zx10"}
MODELS = {  # code/07's models with distress, by label: formula with {d} for the distress terms, and the sample
    "arrival, distress, market": ("{oc} ~ lrx + za + {d} + white", "all"),
    "arrival, distress | east only": ("{oc} ~ lrx + za + {d}", "east"),
    "pills beside all prescribing": ("{oc} ~ lrx + lpills + za + {d} + white", "all"),
    "oral oxycodone and hydrocodone against fentanyl": ("{oc} ~ loral_g + lfent_g + za + {d} + white", "all"),
}
OUTCOMES = {"lt404": "log synthetic-opioid rate 2014-20", "rise": "overdose rise 2011-13 to 2017-19",
            "pre_rise": "overdose rise 2008-10 to 2011-13 (before fentanyl)"}


def state_index(onea, wonder, index_csv):
    """The 2000 and 2008-2012 indexes averaged to states as code/07 averages distress: 2008-10 person-years, counties with dispensing data."""
    cb = pd.read_csv(onea / "county_base.csv", dtype={"fips": str})
    cb["fips"] = cb.fips.str.zfill(5)
    cb["w"] = cb.fips.map(us7.read_wonder(wonder / f"{us7.OD['2008-10']}_county.tsv", 5).set_index("fips").population)
    cb = cb[(cb.rx_missing == 0) & cb.w.notna() & cb.rx.notna()]
    ix = pd.read_csv(index_csv, dtype={"fips": str}).set_index("fips")
    cb = cb.join(ix[["index_2000", "index_2010"]], on="fips")
    r_county, r_rebuilt = pearsonr(cb.econ_distress, cb.index_2010)[0], pearsonr(cb.econ_distress, cb.index_2000)[0]
    print(f"counties: {len(cb)}; without the 2008-2012 index: {cb.index_2010.isna().sum()}; county r, 2000 against 2008-2012: "
          f"{r_county:.3f}; rebuilt 2000 index against the companion county study's: {r_rebuilt:.4f}")
    avg = lambda t, c: np.average(t[c], weights=t.w)
    s = cb.groupby("st").apply(lambda t: pd.Series(dict(distress=avg(t, "econ_distress"), d10=avg(t, "index_2010"))))
    return s, dict(counties=len(cb), r_county=r_county, r_rebuilt_2000=r_rebuilt)


def frame(s5, wonder, bound):
    """code/07's frame at a suppression bound: its bound-5 frame with the death rates recomputed from the WONDER county files."""
    s = s5.copy()
    if bound != 5:
        for lab, f in us7.OD.items():
            d = us7.read_wonder(wonder / f"{f}_county.tsv", bound).groupby("st")[["deaths", "population"]].sum()
            s[f"od_{lab}"] = s.st.map(d.deaths / d.population * 1e5)
        t = us7.read_wonder(wonder / "mcd_t404_2014_2020_county.tsv", bound).groupby("st")[["deaths", "population"]].sum()
        s["t404_2014_20"] = s.st.map(t.deaths / t.population * 1e5)
        s["rise"] = np.log(s["od_2017-19"] / s["od_2011-13"])
        s["pre_rise"] = np.log(s["od_2011-13"] / s["od_2008-10"])
        s["lt404"] = np.log(s.t404_2014_20)
    return s


def main(onea, wonder, index_csv, outdir):
    onea, wonder, outdir = pathlib.Path(onea), pathlib.Path(wonder), pathlib.Path(outdir)
    s5 = pd.read_csv(outdir / "us_states_frame.csv")
    s5["st"] = s5.st.astype(int)
    si, summary = state_index(onea, wonder, index_csv)
    s5 = s5.join(si.d10, on="st")
    gap = (s5.st.map(si.distress) - s5.distress).abs().max()
    assert gap < 1e-9, f"the 2000 index does not reproduce code/07's state distress (largest gap {gap})"
    s5["zx10"] = (s5.d10 - s5.d10.mean()) / s5.d10.std()
    summary.update(states=len(s5), r_state=pearsonr(s5.distress, s5.d10)[0])
    pd.DataFrame([summary]).to_csv(outdir / "hardship_2010_summary.csv", index=False)
    print(f"{len(s5)} states; state r, 2000 against 2008-2012 index: {summary['r_state']:.3f}")
    rows = []
    for bound in (1, 5, 9):
        s = frame(s5, wonder, bound)
        if bound == 5:
            assert np.allclose(s.lt404, np.log(s.t404_2014_20))
        for oc, olab in OUTCOMES.items():
            for mlab, (fml, sample) in MODELS.items():
                for dlab, d in DISTRESS.items():
                    data = s[s.white == 1] if sample == "east" else s
                    f = fml.format(oc=oc, d=d)
                    r = smf.ols(f, data=data).fit(cov_type="HC3")
                    ci = r.conf_int()
                    for term in [t for t in r.params.index if t != "Intercept"]:
                        rows.append(dict(bound=bound, outcome=olab, model=mlab, distress=dlab, term=term, b=r.params[term], lo=ci.loc[term, 0],
                                         hi=ci.loc[term, 1], p=r.pvalues[term], n=int(r.nobs), r2=r.rsquared, formula=f))
    r = pd.DataFrame(rows)
    r.to_csv(outdir / "hardship_2010_check.csv", index=False)
    key = r[r.term.isin(["lrx", "loral_g", "lfent_g", "lpills", "zx", "zx10"])]
    with pd.option_context("display.width", 240, "display.max_rows", 400):
        print(key[key.bound == 5][["outcome", "model", "distress", "term", "b", "lo", "hi", "p"]].round(3).to_string(index=False))
        print("\nThe prescribing terms at each bound:")
        print(key[key.term.isin(["lrx", "loral_g", "lfent_g"])].pivot_table(index=["outcome", "model", "term", "distress"], columns="bound",
                                                                          values="b").round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
