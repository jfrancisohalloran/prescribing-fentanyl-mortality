"""Country tests with mental and behavioural codes added to the outcome.

code/02's narcotic-poisoning outcome (X42, X62, Y12) misses deaths that some countries, Germany, Austria and the UK among them,
code to the F codes for drug use as the underlying cause. This script runs code/02 with two further cause groups, leaving code/02
as it is, and refits code/06's country tests on each:

- `narcotics_f`: X42, X62 and Y12 with F11-F12, F14-F16 and F19, the underlying causes of EUDA's Selection B. Selection B also
  takes X41, X44, X61, X64, X85, Y11 and Y14 only when an opioid, cocaine, cannabis, hallucinogen or amphetamine T-code is among
  the multiple causes, which WHO's file, underlying cause only, cannot apply.
- `drug_f`: all drug poisoning (X40-X44, X60-X64, X85, Y10-Y14) with F11-F16 and F18-F19, the UK's definition of a drug
  poisoning death.

It writes the WHO file with the extra groups to out/who_drug_deaths_fcodes.csv, runs code/06's frame on it with each new group in
place of narcotics, and writes out/fcodes_results.csv: the interaction model, the heavy-quarter contrasts and the rates of the
heaviest prescribers, under each outcome.

  python3 code/29_who_fcodes.py <who_mortality_db date folder> <worldbank_population date folder> <incb csv> <arrival csv> <worldbank_gdp date folder> <out folder>
  python3 code/29_who_fcodes.py --selftest
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

HERE = pathlib.Path(__file__).resolve().parent
F_SELB = ("F11", "F12", "F14", "F15", "F16", "F19")
F_ONS = ("F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def groups(base):
    g = dict(base)
    g["narcotics_f"] = tuple(base["narcotics"]) + F_SELB
    g["drug_f"] = tuple(base["drug_poisoning"]) + F_ONS
    return g


def tests(m, label):
    rows = []
    for oc in ("narc_change",):
        r = smf.ols(f"{oc} ~ lrx_pre * anyill + lgdp", data=m.dropna(subset=[oc, "lgdp"])).fit(cov_type="HC3")
        lo, hi = r.conf_int().loc["lrx_pre:anyill"]
        rows.append(dict(outcome=label, model="interaction, adjusted for log GDP", estimate=r.params["lrx_pre:anyill"], lo=lo, hi=hi,
                         p=r.pvalues["lrx_pre:anyill"], n=int(r.nobs)))
    h = m[m.rx_pre >= m.rx_pre.quantile(0.75)]
    for oc in ("narc_1921", "narc_change"):
        w, wo = h.loc[h.anyill == 1, oc], h.loc[h.anyill == 0, oc]
        rows.append(dict(outcome=label, model=f"Mann-Whitney, heaviest quarter, {oc}", median_with=w.median(), median_without=wo.median(),
                         max_without=wo.max(), p=mannwhitneyu(w, wo).pvalue, n=len(h)))
    return rows


def main(who, wb=None, incb=None, arrival=None, gdp=None, out=None):
    if who == "--selftest":
        base = {"narcotics": ("X42", "X62", "Y12"), "drug_poisoning": ("X40", "X44")}
        g = groups(base)
        assert set(g["narcotics_f"]) == {"X42", "X62", "Y12", "F11", "F12", "F14", "F15", "F16", "F19"}
        assert "F13" not in g["narcotics_f"] and "F13" in g["drug_f"] and "F17" not in g["drug_f"]
        rng = np.random.default_rng(8)
        m = pd.DataFrame(dict(rx_pre=rng.lognormal(7, 1, 60), anyill=(np.arange(60) < 9).astype(int), lgdp=rng.normal(10, 0.5, 60),
                              narc_1921=rng.lognormal(0, 1, 60), narc_change=rng.normal(0, 0.5, 60)))
        m["lrx_pre"] = np.log1p(m.rx_pre)
        print(pd.DataFrame(tests(m, "simulated")).round(3).to_string())
        print("selftest passed")
        return
    w2, t6 = load("w2", "02_who_drug_deaths.py"), load("t6", "06_tests.py")
    out = pathlib.Path(out)
    w2.GROUPS = groups(w2.GROUPS)
    path = out / "who_drug_deaths_fcodes.csv"
    w2.main(who, path, wb)
    w = pd.read_csv(path)
    rows, base = [], None
    for label, col in (("narcotic poisoning (code/02)", "narcotics_per_100k"), ("narcotics with Selection B's F codes", "narcotics_f_per_100k"),
                       ("all drug poisoning with the UK's F codes", "drug_f_per_100k")):
        k = w.assign(narcotics_per_100k=w[col])
        tmp = out / "_who_tmp.csv"
        k.to_csv(tmp, index=False)
        m = t6.frame(incb, tmp, arrival, gdp)
        tmp.unlink()
        if base is None:
            base = set(m.country)
        else:
            # The F codes let a few more countries pass code/06's frame; the like-for-like comparison keeps the main analysis's 60.
            rows += tests(m, label + ", all countries passing")
            m = m[m.country.isin(base)]
        rows += tests(m, label)
        h = m[m.rx_pre >= m.rx_pre.quantile(0.75)].sort_values("rx_pre", ascending=False)
        print(label, len(m), "countries")
        print(h[["country", "status", "narc_1012", "narc_1921", "narc_change"]].round(2).to_string(index=False))
    r = pd.DataFrame(rows)
    r.to_csv(out / "fcodes_results.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:7])
