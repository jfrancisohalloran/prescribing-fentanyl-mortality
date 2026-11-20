"""External causes of death by intent, and the narcotic interaction net of each negative control (2026-11-19, beside code/30).

code/30 found that external causes other than drug poisoning, which fentanyl cannot cause, show an interaction of their own with
prescribing and illicit supply. This script splits them by intent to see where it lies, and refits the main narcotic interaction with
each control's change subtracted from narcotic change, and with it added as a covariate. Its own WHO extraction: underlying cause,
detailed lists 103 and 104, national rows, both sexes, codes matched on their first three characters; population from code/02's
output; window means of yearly rates, each needing two years, as code/06 takes them; change as the log ratio of 2019-21 to 2010-12
with 0.05 added. Each country is matched to code/06's frame (out/tests_countries.csv) by its narcotic window means, so the 60
countries, prescribing, GDP and supply codes are code/06's. OLS with HC3 errors.

  python3 -I code/34_external_by_intent.py <who_mortality_db date folder> <out folder>

Writes out/external_by_intent.csv (each outcome's interaction under two supply codings), out/external_net.csv (the narcotic
interaction net of, and adjusted for, each control) and out/external_market_countries.csv (each market country's change by outcome,
beside the median of countries without a market). Rates and estimates only.
"""
import io
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DRUG = {f"X4{i}" for i in range(5)} | {f"X6{i}" for i in range(5)} | {"X85"} | {f"Y1{i}" for i in range(5)}
CURRENT = {"United States of America", "Canada", "Estonia", "Latvia", "Lithuania"}  # markets present in 2019-21, as code/25 codes them


def num(c):
    return c[0], int(c[1:3])


def between(c, lo, hi):
    return num(lo) <= num(c) <= num(hi)


GROUPS = {  # label, test on a three-character code
    "narcotics": ("Narcotic poisoning, X42, X62, Y12", lambda c: c in {"X42", "X62", "Y12"}),
    "alcohol": ("Alcohol poisoning, X45, X65, Y15", lambda c: c in {"X45", "X65", "Y15"}),
    "road": ("Road traffic, V01-V89", lambda c: between(c, "V01", "V89")),
    "external": ("External causes other than drug poisoning", lambda c: between(c, "V01", "Y89") and c not in DRUG),
    "accident": ("Accidents, V01-X59, drug poisoning excluded", lambda c: between(c, "V01", "X59") and c not in DRUG),
    "selfharm": ("Intentional self-harm, X60-X84, drug poisoning excluded", lambda c: between(c, "X60", "X84") and c not in DRUG),
    "assault": ("Assault, X85-Y09, drug poisoning excluded", lambda c: between(c, "X85", "Y09") and c not in DRUG),
    "undetermined": ("Undetermined intent, Y10-Y34, drug poisoning excluded", lambda c: between(c, "Y10", "Y34") and c not in DRUG),
    "falls": ("Falls, W00-W19", lambda c: between(c, "W00", "W19")),
    "other": ("Other external causes, Y35-Y89", lambda c: between(c, "Y35", "Y89")),
}


def read(path):
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype=str, low_memory=False)


def main(who, out):
    who, out = pathlib.Path(who), pathlib.Path(out)
    rows = []
    for f in sorted(who.glob("morticd10_part*.zip")):
        d = read(f)
        d = d[d.Admin1.isna() & d.SubDiv.isna() & d.Sex.isin(["1", "2"]) & d.List.isin(["103", "104"])]
        d = d[d.Cause.str.match(r"^[A-Z]\d\d")]
        agg = d.assign(c3=d.Cause.str[:3], deaths=pd.to_numeric(d.Deaths1, errors="coerce")).groupby(["Country", "Year", "c3"], as_index=False).deaths.sum()
        for g, (_, fn) in GROUPS.items():
            rows.append(agg[agg.c3.map(fn)].groupby(["Country", "Year"], as_index=False).deaths.sum().assign(group=g))
    wide = pd.concat(rows).pivot_table(index=["Country", "Year"], columns="group", values="deaths", aggfunc="sum").reset_index()
    wide["Year"] = wide.Year.astype(int)
    pop = pd.read_csv(out / "who_drug_deaths_country_year.csv", dtype={"Country": str})[["Country", "Year", "population"]]
    wide = wide.merge(pop, on=["Country", "Year"], how="left")

    def win(col, a, b):
        k = wide[wide.Year.between(a, b)].assign(r=lambda x: x[col] / x.population * 1e5).dropna(subset=["r"])
        g = k.groupby("Country").r.agg(["mean", "count"])
        return g["mean"].where(g["count"] >= 2)

    rates = pd.DataFrame({f"{g}_{w}": win(g, a, b) for g in GROUPS for w, (a, b) in (("1012", (2010, 2012)), ("1921", (2019, 2021)))})
    frame = pd.read_csv(out / "tests_countries.csv")
    cand = rates[["narcotics_1012", "narcotics_1921"]].dropna()
    who_code = []
    for r in frame.itertuples():
        hit = cand[np.isclose(cand.narcotics_1012, r.narc_1012, rtol=1e-9) & np.isclose(cand.narcotics_1921, r.narc_1921, rtol=1e-9)]
        who_code.append(hit.index[0] if len(hit) == 1 else None)
    frame["who"] = who_code
    assert frame.who.notna().all(), list(frame.loc[frame.who.isna(), "country"])
    m = frame.merge(rates, left_on="who", right_index=True, how="left")
    assert np.allclose(m.narcotics_1012, m.narc_1012) and np.allclose(m.narcotics_1921, m.narc_1921)
    m["current"] = m.country.isin(CURRENT).astype(int)
    for g in GROUPS:
        m[f"{g}_chg"] = np.log((m[f"{g}_1921"] + 0.05) / (m[f"{g}_1012"] + 0.05))
    assert np.allclose(m.narcotics_chg, m.narc_change)

    res = []
    for g, (label, _) in GROUPS.items():
        d = m.dropna(subset=[f"{g}_chg", "lgdp"])
        for code, coding in (("anyill", "any market by 2019"), ("current", "markets present in 2019-21")):
            r = smf.ols(f"{g}_chg ~ lrx_pre * {code} + lgdp", data=d).fit(cov_type="HC3")
            t = f"lrx_pre:{code}"
            res.append(dict(outcome=label, key=g, coding=coding, n=int(r.nobs), b=r.params[t], lo=r.conf_int().loc[t, 0],
                            hi=r.conf_int().loc[t, 1], p=r.pvalues[t]))
    res = pd.DataFrame(res)
    res.to_csv(out / "external_by_intent.csv", index=False)

    net = []
    for g in ("alcohol", "road", "external", "selfharm", "assault"):
        d = m.dropna(subset=[f"{g}_chg", "lgdp"]).assign(net=lambda x: x.narcotics_chg - x[f"{g}_chg"])
        for how, formula in (("narcotic change less the control's change", "net ~ lrx_pre * anyill + lgdp"),
                             ("control's change held", f"narcotics_chg ~ lrx_pre * anyill + lgdp + {g}_chg")):
            r = smf.ols(formula, data=d).fit(cov_type="HC3")
            t = "lrx_pre:anyill"
            net.append(dict(control=GROUPS[g][0], key=g, how=how, n=int(r.nobs), b=r.params[t], lo=r.conf_int().loc[t, 0],
                            hi=r.conf_int().loc[t, 1], p=r.pvalues[t]))
    net = pd.DataFrame(net)
    net.to_csv(out / "external_net.csv", index=False)
    mk = m[(m.anyill == 1) | (m.current == 1)][["country", "anyill", "current", "rx_pre"] + [f"{g}_chg" for g in GROUPS]]
    mk = pd.concat([mk, m[m.anyill == 0][[f"{g}_chg" for g in GROUPS]].median().to_frame().T.assign(country="Median, countries without a market by 2019")])
    mk.to_csv(out / "external_market_countries.csv", index=False)
    with pd.option_context("display.width", 220):
        print(res.round(3).to_string(index=False))
        print(net.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
