"""First look: how many countries have both prescribing and deaths, and how the two line up.

Prescribing: INCB's annual consumption of opioids in S-DDD per million inhabitants per day, summed over substances, without methadone
(mostly treatment), averaged over 2015-2017. This stands in for prescribing before fentanyl until the 2000-2014 tables (Table XIV.1 of the
2003-2015 editions) are extracted; national levels move slowly, but for North America 2015-2017 is not "before".
Deaths: WHO underlying-cause rates per 100,000, narcotics and psychodysleptics (X42, X62, Y12) and all drug poisoning, averaged over 2010-2012
and 2019-2021, each needing at least two years.

  python3 code/03_first_look.py <incb_narcotic_drugs date folder> <who deaths csv> <out csv>
"""
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ALIAS = {"united states of america": "united states", "united kingdom of great britain and northern ireland": "united kingdom",
         "czech republic": "czechia", "turkey": "turkiye", "republic of korea": "korea", "korea, republic of": "korea", "russian federation": "russia",
         "iran (islamic republic of)": "iran", "venezuela (bolivarian republic of)": "venezuela", "bolivia (plurinational state of)": "bolivia",
         "republic of moldova": "moldova", "the former yugoslav republic of macedonia": "north macedonia", "netherlands (kingdom of the)": "netherlands",
         "china, hong kong sar": "hong kong", "hong kong sar": "hong kong"}


def key(s):
    s = re.sub(r"[^a-z ,()]", "", str(s).lower().replace("ü", "u")).strip()
    return ALIAS.get(s, s)


def window(d, col, a, b):
    k = d[d.Year.between(a, b)].groupby("key")[col].agg(["mean", "count"])
    return k["mean"].where(k["count"] >= 2)


def main(incb, who, out):
    x = pd.read_excel(pathlib.Path(incb) / "2025_2_Consumption_in_S-DDD_2015-2024.xlsx", sheet_name="Data", header=0).iloc[:, :7]
    x.columns = ["year", "country", "pop_1000s", "substance", "dose_mg", "kg", "sddd"]
    x = x[pd.to_numeric(x.year, errors="coerce").notna()].astype({"year": int})
    x = x[~x.substance.str.contains("methadone", case=False, na=False)]
    s = x[x.year.between(2015, 2017)].groupby(["country", "year"]).sddd.sum().groupby("country").mean().rename("sddd_2015_17")
    fent = x[x.year.between(2015, 2017) & x.substance.str.contains("fentanyl", case=False, na=False)].groupby(["country", "year"]).sddd.sum().groupby("country").mean().rename("fentanyl_sddd_2015_17")
    s = pd.concat([s, fent], axis=1).reset_index()
    s["key"] = s.country.map(key)
    w = pd.read_csv(who)
    w["key"] = w.country_name.map(key)
    w["Year"] = pd.to_numeric(w.Year, errors="coerce")
    deaths = pd.DataFrame({"narc_2010_12": window(w, "narcotics_per_100k", 2010, 2012), "narc_2019_21": window(w, "narcotics_per_100k", 2019, 2021),
                           "drug_2010_12": window(w, "drug_poisoning_per_100k", 2010, 2012), "drug_2019_21": window(w, "drug_poisoning_per_100k", 2019, 2021)})
    m = s.merge(deaths.reset_index(), on="key", how="inner")
    m = m.dropna(subset=["sddd_2015_17", "narc_2019_21"])
    m["narc_change"] = np.log((m.narc_2019_21 + 0.05) / (m.narc_2010_12 + 0.05))
    m.sort_values(["sddd_2015_17", "country"], ascending=[False, True]).round(3).to_csv(out, index=False)  # country breaks ties, so the order is the same on every machine
    print(f"countries with prescribing and later deaths: {len(m)}; with both death windows: {m.narc_2010_12.notna().sum()}")
    for col in ("narc_2019_21", "narc_change", "drug_2019_21"):
        k = m.dropna(subset=[col])
        r = spearmanr(k.sddd_2015_17, k[col])
        print(f"Spearman, opioid S-DDD 2015-17 against {col}: {r[0]:.2f} (p = {r[1]:.3f}, n = {len(k)})")
    top = m.sort_values("sddd_2015_17", ascending=False).head(15)
    print(top[["country", "sddd_2015_17", "fentanyl_sddd_2015_17", "narc_2010_12", "narc_2019_21", "drug_2019_21"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
