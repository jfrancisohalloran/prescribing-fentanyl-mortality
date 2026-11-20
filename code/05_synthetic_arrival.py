"""When illicitly made fentanyls arrived, by country, to 2021: the supply measure for P2 and P3.

The measure is built from supply evidence (seizures, forensic samples, market reports), never from deaths, so it isn't the outcome
under another name. Dates come from the reviews in the literature library, read before use: RAND's international chapter (Pardo et al.
2019, Table 4.4, p. 103, and pp. 73-104), Mounteney et al. 2015 and Gladden et al. 2016. Canada's dates are checked against Health
Canada's Drug Analysis Service samples, which are among the downloaded sources. National reports among them (Australia, Austria, Spain, Italy, Greece,
France, Belgium) and the EU Drug Markets Report 2019 were checked on 2026-11-06 for the heavy prescribers. `status_alt` is a sensitivity
coding: dismantled laboratories count as episodes (Germany, France), and Lithuania counts as established like Latvia.

Status, as of 2019 (the last year before the outcome window ends):
- established: illicitly made fentanyls became a large, lasting share of the illicit opioid market.
- episodic: an outbreak or a separate market that was brief, local or later dismantled.
- diverted only: misuse of fentanyl came from pharmaceutical products, not illicit manufacture.
- unclear: fentanyl use reported, but the sources can't say whether it was illicit.
- none known: no market reported in the reviewed sources or the national reports checked. That's a weaker claim outside their scope (Europe, North America and
  Australia), so `in_review_scope` says which countries the reviews covered.

Fentanyl and its analogues only. Nitazenes are left out,
and they weren't a large share of any market before 2021 in these sources.

  python3 code/05_synthetic_arrival.py <incb csv from code/04> <health_canada_das date folder> <out csv>
"""
import sys

import pandas as pd

P = "Pardo2019"
OS = "other_settings"  # national reports, downloaded (other_settings/20261101), checked 2026-11-06
EU = "EDMR2019"  # EU Drug Markets Report 2019, downloaded (eu_network_sources/20261101/edmr_2019.pdf)
ALT = {"Germany": "episodic", "France": "episodic", "Lithuania": "established"}
TRANSIT = {"Mexico", "China, Hong Kong SAR"}  # large fentanyl seizures in UNODC's 2015-21 data (code/11), no known consumer market  # labs count as episodes; Lithuania as Latvia's twin
EVIDENCE = [  # country (INCB 2025 spelling), status, first year seen, first year at scale, year it ended, product, sources, note
    ("Estonia", "established", 2001, 2002, None, "fentanyl; 3-methylfentanyl from 2002; other analogues from 2015",
     f"{P} pp. 76-77, 103; Ojanpera2008; Mounteney2015 p. 626; Mounteney2012 p. 4",
     "First seizures 2001-2002, dominant since 2002 after the Taliban ban cut heroin; supply severely disrupted in 2017-2018"),
    ("Latvia", "established", 2012, 2015, None, "carfentanil, then other analogues", f"{P} pp. 92-94, 103; LatviaNR2007 p. 92",
     "Carfentanil first found in 2012 but scarce until 2015; analogues still growing at the expense of heroin in 2019. The first year dates the market, not earlier isolated seizures: 43 g of 3-methylfentanyl was seized in 2006"),
    ("Canada", "established", 2012, 2015, None, "fentanyl and analogues, first as counterfeit oxycodone", f"{P} pp. 96-97, 103",
     "British Columbia first; Alberta and Ontario follow. Checked against the Drug Analysis Service below"),
    ("United States of America", "established", 2013, 2016, None, "fentanyl and analogues", f"{P} p. 103; Gladden2016 p. 840; CDC2008 p. 793",
     "Eastern states first. National law-enforcement fentanyl submissions rose from 1,015 in 2013 to 5,343 in 2014. The first year dates the lasting market, not an earlier episode: 1,013 deaths in six jurisdictions in April 2005 to March 2007 were attributed to non-pharmaceutical fentanyl, some of it believed to come from clandestine production in Mexico, where a facility in Toluca was closed in May 2006"),
    ("Sweden", "episodic", 2014, 2014, 2018, "analogues sold online as nasal sprays; diverted patches since 2006", f"{P} pp. 86-88, 92, 103; {EU} p. 115",
     "A separate online market, dismantled by prosecutions in 2017-2018; no reports after early 2018. Brief episodes in the mid-1990s and early 2000s. Two small fentanyl laboratories dismantled in 2016"),
    ("Lithuania", "episodic", None, None, None, "carfentanil", f"{P} p. 74 (citing EMCDDA 2018e); {EU} p. 122",
     "A recent increase in carfentanil trafficking cases, seizures and toxicology findings, reported in 2018; scale and dates not given"),
    ("Bulgaria", "episodic", None, None, None, "fentanyl and 3-methylfentanyl sold as heroin", f"{P} p. 75; Mounteney2015 pp. 626-627",
     "A brief outbreak in the early 2010s, during a heroin shortage. Production facilities have also been seized in Bulgaria, at dates not given"),
    ("Slovakia", "episodic", None, None, None, "fentanyl and 3-methylfentanyl sold as heroin", f"{P} p. 75; Mounteney2015 pp. 626-627",
     "A brief outbreak in the early 2010s, during a heroin shortage; an illicit laboratory dismantled in 2011"),
    ("United Kingdom", "episodic", None, None, None, "fentanyls", f"Mounteney2015 pp. 626-627; {EU} pp. 67, 115",
     "An outbreak of fentanyl-related deaths, in the same period as the Bulgarian and Slovak outbreaks (early 2010s), with no earlier history of misuse. A fentanyl processing centre dismantled and a wholesale online vendor arrested in April 2017"),
    ("Germany", "diverted only", None, None, None, "diverted transdermal patches", f"Mounteney2015 pp. 627-628; {P} p. 75; {EU} p. 115",
     "Misuse linked with diverted pharmaceuticals, Bavaria especially; an outbreak of deaths in the same period as the Bulgarian and Slovak ones. Small fentanyl laboratories dismantled in 2007 and 2015 (episodic in the alternative coding)"),
    ("Finland", "none known", None, None, None, "", f"{P} pp. 74, 103; Mounteney2015 p. 627; Mounteney2012 p. 4",
     "RAND's case of a conspicuous absence: buprenorphine replaced heroin from 2002. Mounteney reports an outbreak of fentanyl deaths. 3-methylfentanyl first seized in 2001, with later detections isolated"),
    ("Australia", "unclear", None, None, None, "fentanyl", f"{P} p. 74 (citing AIHW 2018); {OS} acic_iddr_2017-18 p. 133, idrs_national_2023 pp. 80-82",
     "Fentanyl use noted, illicit or diverted unclear. Non-prescribed fentanyl among people who inject drugs, 5-9% a year nationally in 2018-23; fentanyl among clandestine-laboratory substances in 2017-18, with no count; no illicit market reported"),
    ("Austria", "diverted only", None, None, None, "pharmaceutical fentanyl", f"{OS} at_epi_2022 pp. 41, 81; at_epi_2025 pp. 17, 34",
     "Treatment data count non-prescribed fentanyl (12 mentions in 2021); fentanyl in 5% of drug-death toxicology in 2021; high-potency painkillers like fentanyl called isolated cases"),
    ("Spain", "diverted only", None, None, None, "pharmaceutical fentanyl", f"{OS} oeda_informe_2024 pp. 160, 177; oeda_mortalidad_2025 pp. 11-12",
     "OEDA counts fentanyl among therapeutic opioid painkillers; 12 emergency episodes in 2022; no illicit manufacture reported"),
    ("Belgium", "unclear", None, None, None, "fentanyl, illicit and diverted not separated", f"{OS} be_tdi_2022 pp. 29, 75",
     "The treatment register merges illicit and diverted fentanyl; under 0.1% of treatment episodes in 2022"),
    ("Italy", "none known", None, None, None, "", f"{OS} it_relazione_2022 pp. 213, 358; it_dcsa_2019",
     "Sporadic analogue detections only (first analogue deaths notified in 2018); non-prescribed fentanyl the main drug for 6 people in treatment in 2021; the 2021 wastewater survey tested for analogues"),
    ("France", "none known", None, None, None, "", f"{EU} p. 115; {OS} fr_trend_bordeaux_2023",
     "One small fentanyl laboratory dismantled in 2018 (episodic in the alternative coding); the saved French reports don't mention fentanyl"),
    ("Greece", "none known", None, None, None, "", f"Mounteney2015 p. 627; {OS} gr_ektepn_2020", "An illicit production site seized; no market reported; the 2020 national report doesn't mention fentanyl"),
    ("Portugal", "none known", None, None, None, "", "Mounteney2015 p. 627", "An illicit production site seized; no market reported"),
]
# Countries the reviews covered: the EU and its neighbours in Mounteney's survey, North America, and Australia (RAND).
SCOPE = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
         "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands (Kingdom of the)", "Poland", "Portugal",
         "Romania", "Slovakia", "Slovenia", "Spain", "Sweden", "United Kingdom", "Norway", "Türkiye", "Canada", "United States of America",
         "Australia"}


def canada_check(folder):
    """Fentanyls' share of fentanyl and heroin samples analysed by Health Canada, by year (named fentanyl analogues only)."""
    d = pd.read_csv(f"{folder}/YearlyCounts_20260105.csv")
    c = d[d.LOCATION == "C"].set_index("YEAR")
    named = ["FENTANYL", "CARFENTANIL", "FURANYL FENTANYL", "ACETYL FENTANYL", "CYCLOCPROPYL FENTANYL", "MeOAc FENTANYL",
             "para-FLUOROFENTANYL", "BROMOFENTANYL"]
    f = c[named].sum(axis=1)
    return (100 * f / (f + c.HEROIN)).round(1).loc[2012:2021]


def main(incb, das, out):
    countries = sorted(set(pd.read_csv(incb).country))
    ev = pd.DataFrame(EVIDENCE, columns=["country", "status", "first_seen", "at_scale", "ended", "product", "sources", "note"])
    missing = set(ev.country) - set(countries)
    assert not missing, f"not in the INCB names: {missing}"
    r = pd.DataFrame({"country": countries}).merge(ev, on="country", how="left")
    r["status"] = r.status.fillna("none known")
    r["in_review_scope"] = r.country.isin(SCOPE)
    r["established_by_2019"] = r.status.eq("established")
    r["any_illicit_by_2019"] = r.status.isin(["established", "episodic"])
    r["status_alt"] = r.country.map(ALT).fillna(r.status)
    r["established_alt"] = r.status_alt.eq("established")
    r["any_illicit_alt"] = r.status_alt.isin(["established", "episodic"])
    r["transit"] = r.country.isin(TRANSIT)
    r["any_illicit_or_transit"] = r.any_illicit_by_2019 | r.transit
    r.to_csv(out, index=False)
    share = canada_check(das)
    print("Canada, fentanyls' share of fentanyl and heroin samples (%):", share.to_dict())
    print(f"first year above a third: {share[share > 100 / 3].index.min()}; above half: {share[share > 50].index.min()}")
    print(r.status.value_counts().to_string())
    print(f"{r.in_review_scope.sum()} countries in the reviews' scope; established: {', '.join(r.country[r.established_by_2019])}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
