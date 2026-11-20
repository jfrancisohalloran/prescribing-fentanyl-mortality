"""The paper's tests inside the United States, with states as markets.

Between countries, only four markets had illicit fentanyl by 2019. Every US state had it by 2021, at staggered dates and on very different
prescribing, so the states test P2 (does prescribing set the size of the wave where supply arrived?) and a weak form of P4. They can't test
P3, since no state went without.

- Prescribing: CDC's county opioid dispensing rate per 100 persons, 2009-11 (the companion county study's `county_base.csv`), averaged to states weighted by
  2008-10 person-years. That is before illicit fentanyl in every state (the companion county study's clock puts the earliest arrival in 2014).
- Pills: oxycodone and hydrocodone dosage units shipped per person per year, 2006-12 (DEA ARCOS as released by the Washington Post). It has no patches, so P4 here is pills against all prescribing, not pills
  against patches.
- Deaths: CDC WONDER county files (`wonder_pooled`), summed to states. Overdose (underlying cause D1-D4) in 2008-10, 2011-13,
  2014-16 and 2017-19; synthetic opioids (T40.4, multiple cause) in 2014-20, the companion state study's outcome. Counties WONDER suppresses (1-9 deaths)
  enter at a bound, 1, 5 or 9 (`--bound`); results are shown for each. Crude rates per 100,000 person-years.
- Arrival: the companion county study's half-year clock (`arr_p1_persist`), as the companion state study used it. Distress: the companion county study's economic distress index, weighted the same
  way. Heroin market: the companion county study's white-powder flag (east of the Mississippi).

Models are OLS with HC3 errors on 49 states (48 and DC; the companion county study's counties exclude Alaska and Hawaii), and a state-by-window panel with
state and window effects and errors clustered by state, which asks whether the prescribing gradient steepened once fentanyl arrived.

  python3 code/07_us_states.py <inputs/1a_out folder> <arcos_county_year.csv> <wonder_pooled date folder> <out folder> [code/08's ARCOS retail csv]
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

FIPS = {"ALABAMA": 1, "ALASKA": 2, "ARIZONA": 4, "ARKANSAS": 5, "CALIFORNIA": 6, "COLORADO": 8, "CONNECTICUT": 9, "DELAWARE": 10,
        "DISTRICT OF COLUMBIA": 11, "FLORIDA": 12, "GEORGIA": 13, "HAWAII": 15, "IDAHO": 16, "ILLINOIS": 17, "INDIANA": 18, "IOWA": 19, "KANSAS": 20,
        "KENTUCKY": 21, "LOUISIANA": 22, "MAINE": 23, "MARYLAND": 24, "MASSACHUSETTS": 25, "MICHIGAN": 26, "MINNESOTA": 27, "MISSISSIPPI": 28,
        "MISSOURI": 29, "MONTANA": 30, "NEBRASKA": 31, "NEVADA": 32, "NEW HAMPSHIRE": 33, "NEW JERSEY": 34, "NEW MEXICO": 35, "NEW YORK": 36,
        "NORTH CAROLINA": 37, "NORTH DAKOTA": 38, "OHIO": 39, "OKLAHOMA": 40, "OREGON": 41, "PENNSYLVANIA": 42, "RHODE ISLAND": 44,
        "SOUTH CAROLINA": 45, "SOUTH DAKOTA": 46, "TENNESSEE": 47, "TEXAS": 48, "UTAH": 49, "VERMONT": 50, "VIRGINIA": 51, "WASHINGTON": 53,
        "WEST VIRGINIA": 54, "WISCONSIN": 55, "WYOMING": 56}
TRIPLICATE = {6, 16, 17, 36, 48}  # California, Idaho, Illinois, New York, Texas (Alpert et al., NBER WP 26500, p. 14)
OD = {"2008-10": "ucd_od_2008_2010", "2011-13": "ucd_od_2011_2013", "2014-16": "ucd_od_2014_2016", "2017-19": "ucd_od_2017_2019"}
MID = {"2008-10": 2009, "2011-13": 2012, "2014-16": 2015, "2017-19": 2018}


def read_wonder(path, bound):
    d = pd.read_csv(path, sep="\t", dtype={"fips": str})
    d["fips"] = d.fips.str.zfill(5)
    d["deaths"] = pd.to_numeric(d.deaths.replace("Suppressed", bound), errors="coerce")
    d["population"] = pd.to_numeric(d.population, errors="coerce")
    d["st"] = d.fips.str[:2].astype(int)
    return d


def retail(path):
    """DEA ARCOS Report 4 (code/08): grams per 100,000 per year, 2009-11, for oral oxycodone plus hydrocodone and for fentanyl base."""
    r = pd.read_csv(path)
    r = r[r.year.between(2009, 2011)]
    r["st"] = r.state.map(FIPS)
    r = r.dropna(subset=["st"])
    pick = lambda pat: r[r.drug.str.contains(pat, regex=True)].groupby(["st", "year"]).grams_per_100k.sum().groupby("st").mean()
    return pd.DataFrame({"oral_g": pick(r"^OXYCODONE|^HYDROCODONE"), "fent_g": pick(r"^FENTANYL")})


def state_frame(onea, arcos, wonder, bound, retail_csv=None):
    onea, wonder = pathlib.Path(onea), pathlib.Path(wonder)
    cb = pd.read_csv(onea / "county_base.csv", dtype={"fips": str})
    cb["fips"] = cb.fips.str.zfill(5)
    w0810 = read_wonder(wonder / f"{OD['2008-10']}_county.tsv", bound).set_index("fips").population
    cb["w"] = cb.fips.map(w0810)
    cb = cb[(cb.rx_missing == 0) & cb.w.notna() & cb.rx.notna()]
    s = cb.groupby("st")[["rx", "econ_distress", "w"]].apply(lambda t: pd.Series(dict(rx=np.average(t.rx, weights=t.w), distress=np.average(t.econ_distress, weights=t.w),
                                                         n_counties=len(t))))
    a = pd.read_csv(arcos, dtype={"fips": str})
    a = a[a.year.between(2006, 2012) & a.population_wapo.notna()]
    a["st"] = a.fips.str[:2].astype(int)
    p = a.groupby("st")[["pills_oxy_hydro", "population_wapo"]].sum()
    s["pills"] = p.pills_oxy_hydro / p.population_wapo
    for lab, f in OD.items():
        d = read_wonder(wonder / f"{f}_county.tsv", bound).groupby("st")[["deaths", "population"]].sum()
        s[f"od_{lab}"] = d.deaths / d.population * 1e5
    t = read_wonder(wonder / "mcd_t404_2014_2020_county.tsv", bound).groupby("st")[["deaths", "population"]].sum()
    s["t404_2014_20"] = t.deaths / t.population * 1e5
    clock = pd.read_csv(onea / "halfyear/analogue_state_timing.csv").set_index("st_fips")
    s["arrival"] = clock.arr_p1_persist
    s["state"] = clock.state
    s = s.join(pd.read_csv(onea / "heroin_market_state.csv").set_index("st")[["white"]])
    if retail_csv:
        s = s.join(retail(retail_csv))
    s["triplicate"] = s.index.isin(TRIPLICATE).astype(int)
    s = s.dropna(subset=["rx", "arrival", "od_2011-13", "od_2017-19"]).reset_index()
    s["lrx"] = np.log(s.rx)
    s["lpills"] = np.log(s.pills)
    s["za"] = -(s.arrival - s.arrival.mean()) / s.arrival.std()  # higher = earlier, as in the companion state study
    s["zx"] = (s.distress - s.distress.mean()) / s.distress.std()
    s["rise"] = np.log(s["od_2017-19"] / s["od_2011-13"])
    s["pre_rise"] = np.log(s["od_2011-13"] / s["od_2008-10"])
    s["lt404"] = np.log(s.t404_2014_20)
    if retail_csv:
        s["loral_g"] = np.log(s.oral_g)
        s["lfent_g"] = np.log(s.fent_g)
    return s


def ols(d, formula, label, out, bound):
    r = smf.ols(formula, data=d).fit(cov_type="HC3")
    for term in r.params.index.drop("Intercept"):
        out.append(dict(model=label, formula=formula, bound=bound, n=int(r.nobs), term=term, b=r.params[term], se=r.bse[term], p=r.pvalues[term], r2=r.rsquared))


def main(onea, arcos, wonder, outdir, retail_csv=None):
    outdir, res = pathlib.Path(outdir), []
    for bound in (1, 5, 9):
        s = state_frame(onea, arcos, wonder, bound, retail_csv)
        if bound == 5:
            s.to_csv(outdir / "us_states_frame.csv", index=False)
        east = s[s.white == 1]
        for oc, lab in (("rise", "overdose rise 2011-13 to 2017-19"), ("lt404", "log synthetic-opioid rate 2014-20"), ("pre_rise", "overdose rise 2008-10 to 2011-13 (before fentanyl)")):
            rho = spearmanr(s.rx, s[oc])
            res.append(dict(model=f"P2 | {lab}", formula=f"spearman(rx, {oc})", bound=bound, n=len(s), term="rho", b=rho[0], se=np.nan, p=rho[1], r2=np.nan))
            ols(s, f"{oc} ~ lrx", f"P2 | {lab}", res, bound)
            ols(s, f"{oc} ~ lrx + za", f"P2 + arrival | {lab}", res, bound)
            ols(s, f"{oc} ~ lrx + za + zx + white", f"P2 + arrival, distress, market | {lab}", res, bound)
            ols(east, f"{oc} ~ lrx + za + zx", f"P2 + arrival, distress | east only | {lab}", res, bound)
            ols(s, f"{oc} ~ lrx + lpills + za + zx + white", f"P4 pills beside all prescribing | {lab}", res, bound)
            if retail_csv:
                ols(s, f"{oc} ~ loral_g + lfent_g + za + zx + white", f"P4 oral oxycodone and hydrocodone against fentanyl (ARCOS grams) | {lab}", res, bound)
                ols(s, f"{oc} ~ loral_g + za + zx + white", f"P4 oral alone | {lab}", res, bound)
                ols(s, f"{oc} ~ lfent_g + za + zx + white", f"P4 fentanyl alone | {lab}", res, bound)
        # does the prescribing gradient steepen once fentanyl has arrived? First the cross-state gradient in each window, then a
        # state-by-window panel. With state effects only changes in the gradient are identified, so 2008-10 is the base.
        for lab in OD:
            k = s.assign(lod=np.log(s[f"od_{lab}"]))
            ols(k, "lod ~ lrx", f"P2 gradient across states | overdose rate {lab}", res, bound)
        long = s.melt(id_vars=["st", "lrx", "arrival"], value_vars=[f"od_{k}" for k in OD], var_name="window", value_name="od")
        long["window"] = long.window.str[3:]
        long["post"] = (long.window.map(MID) >= long.arrival).astype(int)
        long["lod"] = np.log(long.od)
        for w in ("2011-13", "2014-16", "2017-19"):
            long[f"lrx_{w[:4]}"] = long.lrx * (long.window == w)
        trend = "lrx_2011 + lrx_2014 + lrx_2017"
        for formula, lab in (("lod ~ C(st) + C(window) + " + trend, "change in the gradient from 2008-10"),
                             ("lod ~ C(st) + C(window) + post + lrx:post", "gradient after the state's arrival"),
                             ("lod ~ C(st) + C(window) + " + trend + " + post + lrx:post", "after arrival, beyond the calendar change in the gradient")):
            r = smf.ols(formula, data=long).fit(cov_type="cluster", cov_kwds={"groups": long.st})
            for term in [t for t in r.params.index if t.startswith("lrx") or t == "post"]:
                res.append(dict(model=f"P2 steepening | {lab}", formula=formula, bound=bound, n=int(r.nobs), term=term, b=r.params[term], se=r.bse[term], p=r.pvalues[term], r2=r.rsquared))
    r = pd.DataFrame(res)
    r.to_csv(outdir / "us_states_results.csv", index=False)
    s = pd.read_csv(outdir / "us_states_frame.csv")
    print(f"{len(s)} states; prescribing {s.rx.min():.0f}-{s.rx.max():.0f} per 100; arrival {s.arrival.min()}-{s.arrival.max()}; "
          f"rx against arrival, Spearman {spearmanr(s.rx, s.arrival)[0]:.2f}; rx against pills {spearmanr(s.rx, s.pills)[0]:.2f}")
    if "oral_g" in s:
        print(f"ARCOS 2009-11: oral against fentanyl grams, Spearman {spearmanr(s.oral_g, s.fent_g)[0]:.2f}; oral against rx {spearmanr(s.oral_g, s.rx)[0]:.2f}; "
              f"fentanyl against rx {spearmanr(s.fent_g, s.rx)[0]:.2f}; states with both {int(s[['oral_g', 'fent_g']].notna().all(axis=1).sum())}")
    keep = r[r.term.isin(["rho", "lrx", "lpills", "za", "lrx:post", "post", "lrx_2011", "lrx_2014", "lrx_2017", "loral_g", "lfent_g"])]
    with pd.option_context("display.width", 230, "display.max_rows", 500):
        print(keep[keep.bound == 5][["model", "n", "term", "b", "se", "p", "r2"]].round(3).to_string(index=False))
        b = keep[keep.term.isin(["lrx", "lrx:post", "lrx_2017"])].pivot_table(index=["model", "term"], columns="bound", values="b").round(3)
        print("\nThe prescribing coefficient at each suppression bound:")
        print(b.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:6])
