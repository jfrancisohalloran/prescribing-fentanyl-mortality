"""Norway on national counts, beside code/10's European heavy prescribers.

Norway is one of the world's 15 heaviest consumers of opioid analgesics but lacks the WHO mortality data the main tests use, so it
is not among the 60 countries. EUDA's DRD-33 has its national count of drug-induced deaths. code/10 turns national counts into rates
with WHO's population, which Norway lacks, so this script uses the World Bank's total population for every country it shows,
all ages, the basis of the paper's "one to five per 100 000". It prints each country's 2010-12 and 2019-21 rates and the change,
and the summary code/10 prints, with Norway added. It also says why Norway is missing from the WHO file: no rows, or only
condensed cause lists (code/02's dropped list).

  python3 code/28_norway_national.py <euda_drd date folder> <worldbank_population date folder> <out folder> [WHO mortality date folder]
  python3 code/28_norway_national.py --selftest
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
P3 = {"Germany": "diverted only", "Belgium": "unclear", "Austria": "diverted only", "Denmark": "none known", "Luxembourg": "none known",
      "Finland": "none known", "Spain": "diverted only", "Netherlands": "none known", "Sweden": "episodic", "United Kingdom": "episodic",
      "Norway": "none known"}
WB_NAMES = {"Netherlands": "Netherlands"}


def wb_population(folder):
    d = json.loads((pathlib.Path(folder) / "SP.POP.TOTL.json").read_text())[1]
    return pd.DataFrame([(x["country"]["value"], int(x["date"]), x["value"]) for x in d if x["value"] is not None],
                        columns=["country", "year", "population"])


def table(e, pop):
    m = e.merge(pop, on=["country", "year"], how="left")
    rows = []
    for c, status in P3.items():
        k = m[m.country == c]
        a, b = k[k.year.between(2010, 2012)], k[k.year.between(2019, 2021)]
        rows.append(dict(country=c, status=status, years_2010_12=len(a), years_2019_21=len(b),
                         rate_2010_12=(a.euda_national / a.population * 1e5).mean(), rate_2019_21=(b.euda_national / b.population * 1e5).mean(),
                         change=np.log(b.euda_national.mean() / a.euda_national.mean()) if len(a) and len(b) else np.nan))
    t = pd.DataFrame(rows)
    without = t[t.status.isin(["none known", "diverted only", "unclear"]) & (t.years_2010_12 >= 2) & (t.years_2019_21 >= 2)]
    return t, without


def main(drd, wb=None, out=None, who=None):
    if drd == "--selftest":
        years = range(2008, 2022)
        e = pd.DataFrame([(c, y, 100 + 10 * i) for i, c in enumerate(P3) for y in years], columns=["country", "year", "euda_national"])
        pop = pd.DataFrame([(c, y, 5e6) for c in P3 for y in years], columns=["country", "year", "population"])
        t, without = table(e, pop)
        assert np.isclose(t.set_index("country").loc["Norway", "rate_2019_21"], (100 + 10 * 10) / 5e6 * 1e5)
        assert "Norway" in set(without.country)
        print(t.round(2).to_string(index=False))
        print("selftest passed")
        return
    spec = importlib.util.spec_from_file_location("e10", HERE / "10_euda_drd_check.py")
    e10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e10)
    e = e10.euda(drd)
    e["year"] = pd.to_numeric(e.year)
    t, without = table(e, wb_population(wb))
    t.to_csv(pathlib.Path(out) / "norway_national.csv", index=False)
    print(t.round(2).to_string(index=False))
    print(f"Without illicit supply, both windows: median change {without.change.median():.2f} (x{np.exp(without.change.median()):.2f}), "
          f"n = {len(without)}; 2019-21 rates {without.rate_2019_21.min():.1f} to {without.rate_2019_21.max():.1f} per 100,000")
    if who:
        spec = importlib.util.spec_from_file_location("w2", HERE / "02_who_drug_deaths.py")
        w2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(w2)
        folder = pathlib.Path(who)
        codes = w2.read_zip(folder / "mort_country_codes.zip").rename(columns=str.lower)
        names = dict(zip(codes.iloc[:, 1], codes.iloc[:, 0]))
        no = names.get("Norway")
        for f in sorted(folder.glob("morticd10_part*.zip")):
            d = w2.read_zip(f)
            k = d[d.Country == no]
            if len(k):
                print(f.name, "Norway rows by list and year:", k.groupby(["List", "Year"]).size().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:5])
