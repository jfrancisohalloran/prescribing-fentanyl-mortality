"""Two further checks, set after the paper's results and a review were seen.

  A   Scotland beside the USA, Canada, England and Wales and the UK in the WHO Mortality Database, on the paper's definitions: narcotic
      poisoning (X42, X62, Y12), all drug poisoning (X40-X44, X60-X64, X85, Y10-Y14), and the share of drug poisonings that the
      narcotic codes hold. The share differs between countries because a death naming several drugs is coded to X44 in some and to
      X42 in others, so the comparison is given on both definitions.
  B   What the heaviest prescribers consumed in morphine milligram equivalents. INCB's defined daily doses for statistical purposes
      (S-DDD, in milligrams, from the technical report for 2012, part four) are turned into milligrams and then into morphine
      equivalents with the conversion factors of CDC's 2022 guideline (MMWR Recomm Rep 71(3), table). Fentanyl's S-DDD, 0.6 mg, is
      a 25 microgram-an-hour patch for a day, 60 morphine milligram equivalents at CDC's 2.4 per microgram an hour. Pethidine,
      tilidine and INCB's "others" have no CDC factor and are left out; their share of S-DDD is reported.

Inputs: this paper's out/ folder and the WHO Mortality Database folder. Writes out/scotland_who.csv (rates per 100 000 only),
out/composition_mme.csv and out/composition_contrasts.csv, the contrasts with countries ranked on each measure.

  python3 code/40_scotland_composition.py <out> <who_mortality_db date folder>
"""
import importlib.util
import io
import pathlib
import sys
import zipfile

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
NARC = ("X42", "X62", "Y12")
DRUG = ("X40", "X41", "X42", "X43", "X44", "X60", "X61", "X62", "X63", "X64", "X85", "Y10", "Y11", "Y12", "Y13", "Y14")
PLACES = {"4330": "Scotland", "4310": "England and Wales", "4308": "United Kingdom", "2450": "USA", "2090": "Canada"}
SDDD_MG = {"codeine": 240, "fentanyl": 0.6, "hydrocodone": 15, "hydromorphone": 20, "morphine": 100, "oxycodone": 75}
MME_PER_MG = {"codeine": 0.15, "fentanyl": 60 / 0.6, "hydrocodone": 1.0, "hydromorphone": 5.0, "morphine": 1.0, "oxycodone": 1.5}
LEFT_OUT = ["pethidine", "tilidine", "others"]


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype=str, low_memory=False)


def scotland(folder, out):
    pop = read_zip(folder / "mort_pop.zip")
    pop = pop[pop.Country.isin(PLACES) & pop.Sex.isin(["1", "2"])]
    pop = pop.assign(p=pd.to_numeric(pop.Pop1, errors="coerce")).groupby(["Country", "Year"]).p.sum()
    # the WHO population file lacks some countries and years; the paper's own country file carries the population it used
    own = pd.read_csv(out / "who_drug_deaths_country_year.csv", dtype={"Country": str, "Year": str})
    own = own[own.Country.isin(PLACES)].set_index(["Country", "Year"]).population
    pop = pop.combine_first(own)
    parts = []
    for part in sorted(folder.glob("morticd10_part*.zip")):
        d = read_zip(part)
        d = d[d.Country.isin(PLACES) & d.Sex.isin(["1", "2"]) & d.List.isin(["103", "104"])]
        d = d.assign(c3=d.Cause.str[:3], n=pd.to_numeric(d.Deaths1, errors="coerce"))
        parts.append(d[d.c3.isin(DRUG)][["Country", "Year", "c3", "n"]])
    d = pd.concat(parts)
    rows = []
    for (c, y), g in d.groupby(["Country", "Year"]):
        p = pop.get((c, y))
        if p is None or not 2010 <= int(y) <= 2021:
            continue
        rows.append(dict(place=PLACES[c], year=int(y), narcotics=g.n[g.c3.isin(NARC)].sum() / p * 1e5, drug_poisoning=g.n.sum() / p * 1e5,
                         unspecified_x44_y14=g.n[g.c3.isin(("X44", "X64", "Y14"))].sum() / p * 1e5))
    t = pd.DataFrame(rows)
    res = []
    for win, (a, b) in (("2010-12", (2010, 2012)), ("2019-20", (2019, 2020)), ("2019-21", (2019, 2021))):
        k = t[t.year.between(a, b)].groupby("place")[["narcotics", "drug_poisoning", "unspecified_x44_y14"]].agg(["mean", "count"])
        for place in k.index:
            r = k.loc[place]
            res.append(dict(place=place, window=win, years=int(r[("narcotics", "count")]), narcotics_per_100k=r[("narcotics", "mean")],
                            drug_poisoning_per_100k=r[("drug_poisoning", "mean")], unspecified_per_100k=r[("unspecified_x44_y14", "mean")]))
    res = pd.DataFrame(res)
    res["narcotic_share"] = res.narcotics_per_100k / res.drug_poisoning_per_100k
    res["unspecified_share"] = res.unspecified_per_100k / res.drug_poisoning_per_100k
    res.to_csv(out / "scotland_who.csv", index=False)
    return res


def composition(out):
    spec = importlib.util.spec_from_file_location("c39", HERE / "39_further_checks.py")
    c39 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c39)
    res = []
    m, _ = c39.item7(out, res)
    m["mme"] = sum(m[s] * SDDD_MG[s] * MME_PER_MG[s] for s in SDDD_MG)
    m["left_out_share"] = m[LEFT_OUT].sum(axis=1) / m.rx_pre
    for s in ("oxycodone", "hydrocodone", "fentanyl", "morphine"):
        m[s + "_mme_share"] = m[s] * SDDD_MG[s] * MME_PER_MG[s] / m.mme
    m["sddd_share_oxy_hydro"], m["sddd_share_fentanyl"] = m.oxy_hydro / m.rx_pre, m.fentanyl / m.rx_pre
    m["rank_sddd"] = m.rx_pre.rank(ascending=False)
    m["rank_mme"] = m.mme.rank(ascending=False)
    us, ca = m.set_index("country").loc[c39.USA], m.set_index("country").loc[c39.CAN]
    m["sddd_of_usa"], m["mme_of_usa"] = m.rx_pre / us.rx_pre, m.mme / us.mme
    m["sddd_of_canada"], m["mme_of_canada"] = m.rx_pre / ca.rx_pre, m.mme / ca.mme
    keep = m[(m.rank_sddd <= 15) | (m.rank_mme <= 15)].sort_values("rank_sddd")
    cols = ["country", "status", "rank_sddd", "rank_mme", "rx_pre", "mme", "sddd_of_usa", "mme_of_usa", "sddd_of_canada", "mme_of_canada",
            "sddd_share_oxy_hydro", "sddd_share_fentanyl", "oxycodone_mme_share", "hydrocodone_mme_share", "fentanyl_mme_share", "morphine_mme_share", "left_out_share", "narc_1921"]
    keep[cols].to_csv(out / "composition_mme.csv", index=False)
    c39.contrasts(m, "mme", "morphine milligram equivalents", res, "7")
    con = pd.DataFrame(res)
    con = con[con.check.str.startswith("15 heaviest on")]
    con.to_csv(out / "composition_contrasts.csv", index=False)
    return keep[cols], con


def main(out, who):
    out, who = pathlib.Path(out), pathlib.Path(who)
    with pd.option_context("display.width", 300, "display.max_colwidth", 200):
        print(scotland(who, out).round(3).to_string(index=False))
        comp, con = composition(out)
        print(comp.round(3).to_string(index=False))
        print(con[["check", "sample", "b", "lo", "hi", "p", "n", "n_with", "note"]].round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
