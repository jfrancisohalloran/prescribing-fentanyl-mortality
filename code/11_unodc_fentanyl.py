"""A global check on the supply measure: fentanyl-type seizures by country, 2015-2021, from UNODC's World Drug Report 2026 annex (table 7.1).

Reads only rows whose drug is fentanyl, carfentanil or 3-methylfentanyl (nitazene and other new-opioid rows are not read). Seizures reflect enforcement and transit as well as use, and UNODC files fentanyl under both pharmaceutical and illicit
opioids, so this checks the review-based coding rather than replacing it: does any country outside the established and episodic markets
seize fentanyl on their scale?

  python3 code/11_unodc_fentanyl.py <unodc_wdr_seizures date folder> <worldbank_population date folder> <arrival csv from code/05> <out csv>
"""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ALIAS = {"united kingdom of great britain and northern ireland": "united kingdom", "venezuela (bolivarian republic of)": "venezuela (bolivarian rep of)",
         "netherlands": "netherlands (kingdom of the)", "china, hong kong special administrative region": "china, hong kong sar",
         "china, macao special administrative region": "china, macao sar", "turkey": "turkiye", "czech republic": "czechia", "united states": "united states of america"}


def key(s):
    s = str(s).lower().replace("’", "'").replace("ü", "u").replace("é", "e").replace("ç", "c")
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z ,()']", " ", s)).strip()
    return ALIAS.get(s, s)


def main(unodc, wb, arrival, out):
    d = pd.read_excel(pathlib.Path(unodc) / "7.1_Drug_seizures_2015-2024.xlsx", header=1)
    d = d[d.DrugName.astype(str).str.contains("fentan", case=False) & d["Reference year"].between(2015, 2021)]
    d["kg"] = pd.to_numeric(d.Kilograms, errors="coerce")
    g = d.groupby(["Iso3_code", "Country", "Reference year"]).kg.sum().reset_index()
    pop = json.loads((pathlib.Path(wb) / "SP.POP.TOTL.json").read_text())[1]
    pop = pd.DataFrame([(r["countryiso3code"], int(r["date"]), r["value"]) for r in pop if r["value"]], columns=["Iso3_code", "Reference year", "pop"])
    g = g.merge(pop, on=["Iso3_code", "Reference year"], how="left")
    c = g.groupby(["Iso3_code", "Country"]).agg(years_reported=("Reference year", "nunique"), first_year=("Reference year", "min"),
                                                 kg_2015_21=("kg", "sum"), pop=("pop", "mean")).reset_index()
    c["g_per_million_per_year"] = c.kg_2015_21 * 1000 / (c["pop"] / 1e6) / 7
    a = pd.read_csv(arrival)
    a["key"] = a.country.map(key)
    c["key"] = c.Country.map(key)
    c = c.merge(a[["key", "country", "status", "in_review_scope"]], on="key", how="left")
    c["status"] = c.status.fillna("not in INCB list")
    c = c.sort_values("g_per_million_per_year", ascending=False)
    c.drop(columns="key").to_csv(out, index=False)
    print(f"{len(c)} countries report fentanyl-type seizures in 2015-2021; unmatched to INCB names: {sorted(c.loc[c.country.isna(), 'Country'])}")
    print(c[["Country", "status", "years_reported", "first_year", "kg_2015_21", "g_per_million_per_year"]].round(2).to_string(index=False))
    s = c.groupby("status").g_per_million_per_year.median().round(2)
    print("\nMedian grams per million per year, by our coding:", s.to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:5])
