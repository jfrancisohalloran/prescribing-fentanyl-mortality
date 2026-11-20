"""Does WHO's underlying-cause coding miss drug deaths that national monitoring counts? A check against EUDA's DRD-33.

DRD-33 (Statistical Bulletin 2023) gives each country's drug-induced deaths under its own national definition, 1985-2021, usually from
special or forensic registers as well as general mortality data. WHO's counts come from code/02: narcotics (X42, X62, Y12) and all drug
poisoning (X40-X44, X60-X64, X85, Y10-Y14), underlying cause only. Where national counts rise and WHO's don't, WHO's coding is missing
the deaths, which matters most for Latvia, whose fentanyl-analogue market from 2015 doesn't show in WHO's figures.

  python3 code/10_euda_drd_check.py <euda_drd date folder> <who deaths csv from code/02> <out csv>
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ALIAS = {"Czechia": "Czech Republic", "Türkiye": "Turkey", "Netherlands": "Netherlands"}
SHOW = ["Latvia", "Estonia", "Lithuania", "Sweden", "Finland", "Germany", "Austria", "United Kingdom", "Bulgaria", "Slovakia"]


def euda(folder):
    d = pd.read_excel(pathlib.Path(folder) / "DRD-33.xlsx", header=None)
    head = d.index[d.iloc[:, 0].astype(str).str.strip().eq("Country")][0]
    years = [int(y) for y in d.iloc[head, 1:].dropna()]
    body = d.iloc[head + 1:, :len(years) + 1]
    body = body[body.iloc[:, 0].notna() & body.iloc[:, 1:].notna().any(axis=1)]
    body.columns = ["country"] + years
    body = body[~body.country.astype(str).str.match(r"^\(\d+\)")]  # footnotes
    t = body.melt(id_vars="country", var_name="year", value_name="euda_national")
    t["euda_national"] = pd.to_numeric(t.euda_national, errors="coerce")
    return t.dropna(subset=["euda_national"])


def main(folder, who, out):
    e = euda(folder)
    w = pd.read_csv(who)
    w["Year"] = pd.to_numeric(w.Year, errors="coerce")
    w["country"] = w.country_name.replace({v: k for k, v in ALIAS.items()})
    m = e.merge(w[["country", "Year", "narcotics", "drug_poisoning", "population"]].rename(columns={"Year": "year"}), on=["country", "year"], how="left")
    m.to_csv(out, index=False)
    print(f"{e.country.nunique()} countries in DRD-33, {e.year.min()}-{e.year.max()}; {m.narcotics.notna().sum()} country-years matched to WHO")
    for c in SHOW:
        k = m[(m.country == c) & m.year.between(2008, 2021)].sort_values("year")
        if k.empty:
            print(f"{c}: not in DRD-33")
            continue
        print(f"\n{c} (EUDA national / WHO narcotics / WHO all drug poisoning):")
        print("  " + "  ".join(f"{int(r.year)}: {r.euda_national:.0f}/{r.narcotics:.0f}/{r.drug_poisoning:.0f}" if pd.notna(r.narcotics)
                              else f"{int(r.year)}: {r.euda_national:.0f}/-/-" for r in k.itertuples()))
    # does the national count move with WHO's? change 2010-12 to 2019-21, log ratio, where both windows have data
    rows = []
    for c, k in m.groupby("country"):
        a, b = k[k.year.between(2010, 2012)], k[k.year.between(2019, 2021)]
        if len(a) >= 2 and len(b) >= 2:
            rows.append(dict(country=c, euda_change=np.log(b.euda_national.mean() / a.euda_national.mean()),
                             who_narc_change=np.log((b.narcotics.mean() + 0.5) / (a.narcotics.mean() + 0.5)) if b.narcotics.notna().sum() >= 2 and a.narcotics.notna().sum() >= 2 else np.nan))
    r = pd.DataFrame(rows).dropna()
    print(f"\nChange 2010-12 to 2019-21, EUDA national against WHO narcotics: Spearman {r.euda_change.corr(r.who_narc_change, method='spearman'):.2f} (n = {len(r)})")
    print(r.assign(gap=r.euda_change - r.who_narc_change).sort_values("gap").round(2).to_string(index=False))
    # P3 on national counts: the European top-quarter prescribers (code/06), rates per 100,000 from WHO's or the World Bank's population
    p3 = {"Germany": "diverted only", "Belgium": "unclear", "Austria": "diverted only", "Denmark": "none known", "Luxembourg": "none known",
          "Finland": "none known", "Spain": "diverted only", "Netherlands": "none known", "Sweden": "episodic", "United Kingdom": "episodic"}
    rows = []
    for c, status in p3.items():
        k = m[m.country == c]
        a, b = k[k.year.between(2010, 2012)], k[k.year.between(2019, 2021)]
        ra = (a.euda_national / a.population * 1e5).mean() if a.population.notna().any() else np.nan
        rb = (b.euda_national / b.population * 1e5).mean() if b.population.notna().any() else np.nan
        rows.append(dict(country=c, status=status, years_2010_12=len(a), years_2019_21=len(b), rate_2010_12=ra, rate_2019_21=rb,
                         change=np.log(b.euda_national.mean() / a.euda_national.mean()) if len(a) and len(b) else np.nan))
    t = pd.DataFrame(rows)
    print("\nP3 on national counts, European top-quarter prescribers (rates per 100,000):")
    print(t.round(2).to_string(index=False))
    without = t[t.status.isin(["none known", "diverted only", "unclear"]) & (t.years_2010_12 >= 2) & (t.years_2019_21 >= 2)]
    print(f"Without illicit supply, both windows: median change {without.change.median():.2f} (x{np.exp(without.change.median()):.2f}), "
          f"n = {len(without)}; highest 2019-21 rate {without.rate_2019_21.max():.1f} per 100,000")


if __name__ == "__main__":
    main(*sys.argv[1:4])
