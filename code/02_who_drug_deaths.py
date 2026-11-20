"""Drug deaths by country and year from the WHO Mortality Database (underlying cause, all ages, both sexes), read straight from the zips.

Groups: drug poisoning of any intent (X40-X44, X60-X64, X85, Y10-Y14); narcotics and psychodysleptics (X42, X62, Y12), the closest the
underlying cause comes to opioid poisoning; and opioid use disorders (F11). Countries that report only condensed lists (101, 10M) can't be
split this way and are dropped, with their names printed.

Population comes from WHO's own file; where that has no figure (the US and Canada in recent years, among others), the World Bank's total
population fills it, matched on country name (column population_source says which).

  python3 code/02_who_drug_deaths.py <who_mortality_db date folder> <out csv> [worldbank_population date folder]
"""
import io
import json
import pathlib
import re
import sys
import zipfile

import pandas as pd

ALIAS = {"united states of america": "united states", "korea, rep.": "korea", "republic of korea": "korea", "iran, islamic rep.": "iran",
         "iran (islamic republic of)": "iran", "venezuela, rb": "venezuela", "venezuela (bolivarian republic of)": "venezuela", "egypt, arab rep.": "egypt",
         "slovak republic": "slovakia", "kyrgyz republic": "kyrgyzstan", "turkey": "turkiye", "czech republic": "czechia", "republic of moldova": "moldova",
         "hong kong sar, china": "hong kong", "china, hong kong sar": "hong kong", "bahamas, the": "bahamas", "gambia, the": "gambia",
         "netherlands (kingdom of the)": "netherlands", "russian federation": "russia", "the former yugoslav republic of macedonia": "north macedonia",
         "saint vincent and the grenadines": "st vincent", "st. vincent and the grenadines": "st vincent", "saint lucia": "st lucia", "st. lucia": "st lucia",
         "saint kitts and nevis": "st kitts", "st. kitts and nevis": "st kitts", "micronesia, fed. sts.": "micronesia", "micronesia (federated states of)": "micronesia"}


def key(s):
    s = re.sub(r"\s+", " ", str(s).lower().replace("ü", "u")).strip()
    return ALIAS.get(s, s)


def worldbank(folder):
    d = json.loads((pathlib.Path(folder) / "SP.POP.TOTL.json").read_text())[1]
    w = pd.DataFrame([(key(x["country"]["value"]), int(x["date"]), x["value"]) for x in d if x["value"] is not None], columns=["key", "Year", "wb_population"])
    return w.drop_duplicates(["key", "Year"]).set_index(["key", "Year"]).wb_population

GROUPS = {
    "drug_poisoning": ("X40", "X41", "X42", "X43", "X44", "X60", "X61", "X62", "X63", "X64", "X85", "Y10", "Y11", "Y12", "Y13", "Y14"),
    "narcotics": ("X42", "X62", "Y12"),
    "opioid_use_disorder": ("F11",),
}
DETAILED = {"103", "104"}  # three- and four-character ICD-10 lists


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        return pd.read_csv(io.BytesIO(z.read(name)), dtype=str, low_memory=False)


def main(folder, out, wb_folder=None):
    folder = pathlib.Path(folder)
    codes = read_zip(folder / "mort_country_codes.zip").rename(columns=str.lower)
    names = dict(zip(codes.iloc[:, 0], codes.iloc[:, 1]))
    pop = read_zip(folder / "mort_pop.zip")
    pop = pop[(pop.Admin1.isna()) & (pop.SubDiv.isna()) & (pop.Sex.isin(["1", "2"]))]
    pop = pop.assign(Pop1=pd.to_numeric(pop.Pop1, errors="coerce")).groupby(["Country", "Year"]).Pop1.sum().rename("population")
    parts = []
    for f in sorted(folder.glob("morticd10_part*.zip")):
        d = read_zip(f)
        d = d[d.Admin1.isna() & d.SubDiv.isna() & d.Sex.isin(["1", "2"])]
        cause3 = d.Cause.str[:3]
        d = d.assign(cause3=cause3, deaths=pd.to_numeric(d.Deaths1, errors="coerce"))
        for g, cs in GROUPS.items():
            k = d[d.cause3.isin(cs)]
            parts.append(k.groupby(["Country", "Year", "List"], as_index=False).deaths.sum().assign(group=g))
        print(f.name, len(d), "rows")
    long = pd.concat(parts, ignore_index=True)
    r = long.pivot_table(index=["Country", "Year", "List"], columns="group", values="deaths", aggfunc="sum").reset_index()
    r.columns.name = None
    dropped = sorted({names.get(c, c) for c in r.loc[~r.List.isin(DETAILED), "Country"]})
    r = r[r.List.isin(DETAILED)].drop(columns="List").groupby(["Country", "Year"]).sum(min_count=1).reset_index()
    r = r.join(pop, on=["Country", "Year"])
    r.insert(1, "country_name", r.Country.map(names))
    r["population_source"] = r.population.notna().map({True: "WHO", False: ""})
    if wb_folder:
        r["Year"] = pd.to_numeric(r.Year)
        wb = worldbank(wb_folder)
        fill = pd.Series([wb.get((key(n), y)) for n, y in zip(r.country_name, r.Year)], index=r.index, dtype=float)
        use = r.population.isna() & fill.notna()
        r.loc[use, "population"] = fill[use]
        r.loc[use, "population_source"] = "World Bank"
    for g in GROUPS:
        r[f"{g}_per_100k"] = r[g] / r.population * 1e5
    r.to_csv(out, index=False)
    print(f"{r.Country.nunique()} countries, years {r.Year.min()}-{r.Year.max()}; condensed lists only, dropped: {', '.join(dropped[:12])}{' ...' if len(dropped) > 12 else ''}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
