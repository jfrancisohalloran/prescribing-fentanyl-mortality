"""Europe's fentanyl markets after China's class-wide control of fentanyl-related substances on 1 May 2019.

The Discussion sets that control beside the US rise in 2019-21. This check asks whether the European markets, which our coding marks as
established or episodic by 2019, receded after it: fentanyl-type seizures by year from UNODC's World Drug Report 2026 annex (table 7.1,
fentanyl, carfentanil and 3-methylfentanyl, the only fentanyl-type substances it lists), and narcotic-poisoning deaths per 100 000 by
year from the WHO Mortality Database, as built by code/02. Both run past the study's 2019-21 window, to 2024 where countries reported.
A country-year with no row in the annex is a year with no seizure reported, which is kept apart from a reported zero.

  python3 code/21_europe_after_2019.py <unodc_wdr_seizures date folder> <out folder>
"""
import importlib.util
import pathlib
import sys

import pandas as pd

YEARS = range(2015, 2025)


def unodc_key():
    """code/11's name matching, so countries join the same way in both checks."""
    spec = importlib.util.spec_from_file_location("unodc11", pathlib.Path(__file__).with_name("11_unodc_fentanyl.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.key


def main(unodc, out):
    out, key = pathlib.Path(out), unodc_key()
    u = pd.read_excel(pathlib.Path(unodc) / "7.1_Drug_seizures_2015-2024.xlsx", header=1)
    u["key"] = u.Country.map(key)
    europe = set(u.loc[u.Region == "Europe", "key"])
    a = pd.read_csv(out / "synthetic_arrival.csv")
    a["key"] = a.country.map(key)
    markets = a[a.status.isin(["established", "episodic"]) & a.key.isin(europe)][["country", "key", "status"]]

    f = u[u.DrugName.astype(str).str.contains("fentan", case=False) & u["Reference year"].isin(YEARS)].copy()
    f["kg"] = pd.to_numeric(f.Kilograms, errors="coerce")
    f["carf"] = f.kg.where(f.DrugName.astype(str).str.contains("carfentanil", case=False))
    kg = f.groupby(["key", "Reference year"]).agg(kg_reported=("kg", "sum"), carfentanil_kg=("carf", "sum")).reset_index()
    kg = kg.rename(columns={"Reference year": "year"})

    w = pd.read_csv(out / "who_drug_deaths_country_year.csv")
    w["key"] = w.country_name.map(key)
    w = w[w.Year.isin(YEARS)].rename(columns={"Year": "year"})[["key", "year", "narcotics", "population", "narcotics_per_100k"]]

    grid = markets.merge(pd.DataFrame({"year": list(YEARS)}), how="cross")
    d = grid.merge(kg, on=["key", "year"], how="left").merge(w, on=["key", "year"], how="left").drop(columns="key")
    d.to_csv(out / "europe_after_2019.csv", index=False)

    print(f"{len(markets)} European markets: {', '.join(markets.country)}")
    pd.set_option("display.width", 200)
    print("\nFentanyl-type seizures reported to UNODC, kg (blank: none reported):")
    print(d.pivot(index="country", columns="year", values="kg_reported").round(3).to_string(na_rep=""))
    print("\nNarcotic-poisoning deaths per 100 000, WHO:")
    print(d.pivot(index="country", columns="year", values="narcotics_per_100k").round(2).to_string(na_rep=""))
    r = d.pivot(index="country", columns="year", values="narcotics_per_100k")
    s = pd.DataFrame({"2016-17": r[[2016, 2017]].mean(axis=1), "2018": r[2018], "2020-21": r[[2020, 2021]].mean(axis=1),
                      "2022-24": r[[2022, 2023, 2024]].mean(axis=1)})
    s["change 2016-17 to 2018, %"] = 100 * (s["2018"] / s["2016-17"] - 1)
    s["change 2018 to 2020-21, %"] = 100 * (s["2020-21"] / s["2018"] - 1)
    print("\nMean rates by period:")
    print(s.round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
