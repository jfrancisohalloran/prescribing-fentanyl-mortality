"""The appendix (paper/APPENDIX.md): the data sources, the check on the INCB extraction, illicit fentanyl supply by country with its
evidence, the seizure and national-count checks, the check on GBD's modelled estimates, every model fitted with its estimate, the STROBE
checklist, and the sources cited. Built from the outputs and the data catalogue, so no number is retyped.

Written to these rules: plain section titles, an opening paragraph, full sentences, no italics and no asides
in parentheses, and tables that give each model in the same plain words row to row, never by internal labels or code variables. The
coding notes are kept in plain sentences here (NOTES); if code/05 changes a note, this script stops until its plain version is updated.

  python3 code/16_appendix.py <out folder> <paper folder> <data_catalog/sources.csv>
"""
import pathlib
import re
import sys

import pandas as pd

DATASETS = ["incb_narcotic_drugs", "who_mortality_db", "worldbank_population", "worldbank_gdp_per_capita", "euda_drd", "unodc_wdr_seizures",
            "euda_edr_pages", "reitox_national_reports", "ihme_gbd", "health_canada_das", "wonder_pooled", "wonder_cvd_drug", "wonder_state_year_t404", "wonder_state_month", "dea_arcos_retail", "acs_county_2012", "other_settings",
            "eu_network_sources", "arcos_county", "wonder_county_year", "teds_a", "euda_statsbulletin_2023", "statcan_deaths_cause",
            "statcan_population", "canada_province_prescribing"]
SHORT = {"United States of America": "USA", "United Kingdom": "UK", "China, Hong Kong SAR": "Hong Kong", "Netherlands (Kingdom of the)": "Netherlands"}


def md(df):
    """A pipe table; column widths follow the longest entries, capped, so LaTeX shares the page sensibly."""
    d = df.fillna("").astype(str)
    widths = [min(40, max(4, len(col), *(len(v) for v in d[col].head(200)))) for col in d.columns]
    head = "| " + " | ".join(d.columns) + " |\n|" + "|".join("-" * w for w in widths) + "|\n"
    return head + "\n".join("| " + " | ".join(v.replace("|", "/") for v in row) + " |" for row in d.values) + "\n"


def joinx(parts):
    """Terms of an interaction, 'Log prescribing × illicit supply': lower case after the first, as in running text."""
    return " × ".join([parts[0]] + [p[0].lower() + p[1:] if not p[:2].isupper() else p for p in parts[1:]])


def period(w):
    """'2006-2008' -> '2006–08'."""
    return f"{w[:4]}–{w[-2:]}"


PLAIN_SOURCES = {  # A1: what each dataset is, in plain words; the retrieval dates come from the catalogue
    "incb_narcotic_drugs": ("Narcotic drugs: technical reports, 2003 to 2015 editions, and the 2025 edition's consumption tables", "International Narcotics Control Board"),
    "who_mortality_db": ("WHO Mortality Database: deaths by cause, sex and age, with population", "World Health Organization"),
    "worldbank_population": ("World Development Indicators: total population", "World Bank"),
    "worldbank_gdp_per_capita": ("World Development Indicators: GDP per capita at purchasing-power parity", "World Bank"),
    "euda_drd": ("Statistical Bulletin 2023: drug-induced deaths under national definitions, table DRD-33", "European Union Drugs Agency"),
    "unodc_wdr_seizures": ("World Drug Report 2026, statistical annex: drug seizures by country, drug and year, 2015–24", "United Nations Office on Drugs and Crime"),
    "euda_edr_pages": ("European Drug Report 2024, 2025 and 2026: the chapters on drug-induced deaths and on heroin and other opioids", "European Union Drugs Agency"),
    "reitox_national_reports": ("Reitox national reports to the EMCDDA: Latvia, 2007 edition", "Reitox national focal points"),
    "ihme_gbd": ("Global Burden of Disease 2023: death rates from opioid use disorders", "Institute for Health Metrics and Evaluation"),
    "health_canada_das": ("Drug Analysis Service: analysed samples", "Health Canada"),
    "wonder_pooled": ("CDC WONDER: county deaths by underlying and multiple cause", "Centers for Disease Control and Prevention"),
    "wonder_cvd_drug": ("CDC WONDER: deaths by state with a cardiovascular underlying cause and a drug poisoning contributing, with "
                        "drug-poisoning deaths, 1999–2020", "Centers for Disease Control and Prevention"),
    "wonder_state_year_t404": ("CDC WONDER: synthetic-opioid overdose deaths by state and year, 1999–2020", "Centers for Disease Control and Prevention"),
    "wonder_state_month": ("CDC WONDER: synthetic-opioid overdose deaths by state and month, 2019–24", "Centers for Disease Control and Prevention"),
    "dea_arcos_retail": ("ARCOS Retail Drug Summary, report 4: distribution by state in grams per 100 000 population", "Drug Enforcement Administration"),
    "acs_county_2012": ("American Community Survey 2008–2012 five-year summary file, county tables, with the Small Area Income and Poverty "
                        "Estimates for 2010", "US Census Bureau"),
    "other_settings": ("National reports on opioid markets and deaths in Australia, Austria, Belgium, France, Greece, Italy and Spain", "National agencies, listed in A15"),
    "eu_network_sources": ("EU Drug Markets Report 2019", "European Monitoring Centre for Drugs and Drug Addiction and Europol"),
    "arcos_county": ("ARCOS: oxycodone and hydrocodone dosage units shipped by county and year, 2006–14, as released by the Washington Post",
                     "Drug Enforcement Administration"),
    "wonder_county_year": ("CDC WONDER: county deaths by year that are both drug poisoning and synthetic-opioid poisoning, 2018–25",
                           "Centers for Disease Control and Prevention"),
    "teds_a": ("Treatment Episode Data Set: Admissions, public use file, 2006–23", "Substance Abuse and Mental Health Services Administration"),
    "euda_statsbulletin_2023": ("Statistical Bulletin 2023: high-risk opioid use, opioid agonist treatment and treatment entrants",
                                "European Union Drugs Agency"),
    "statcan_deaths_cause": ("Vital statistics: deaths by selected grouped causes and province, table 13-10-0932-01", "Statistics Canada"),
    "statcan_population": ("Population estimates on 1 July by province, table 17-10-0005-01", "Statistics Canada"),
    "canada_province_prescribing": ("Strong-opioid dispensing by province, fiscal 2007–08 to 2016–17, table 1 of Jones, Kurdyak and Fischer 2020",
                                    "BMC Health Services Research"),
}


def retrieved(text):
    """'2026-11-06 (2006-2015); 2026-11-10 (2016-2023)' -> '2006–15 on 2026-11-06; 2016–23 on 2026-11-10'."""
    return re.sub(r"(\d{4}-\d{2}-\d{2}) \((\d{4})-(\d{4})\)", lambda m: f"{m.group(2)}–{m.group(3)[2:]} on {m.group(1)}", text)


def pages(a, b):
    """Lancet page ranges: 626-627 -> 626–27, 96-97 -> 96–97."""
    k = 0
    while k < len(a) - 2 and len(a) == len(b) and a[k] == b[k]:
        k += 1
    return f"{a}–{b[k:]}"


def est(b, se):
    return f"{b:.3f} ({b - 1.96 * se:.3f} to {b + 1.96 * se:.3f})" if pd.notna(se) else f"{b:.3f}"


def pval(p):
    return "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def eci(b, lo, hi):
    return f"{b:.3f} ({lo:.3f} to {hi:.3f})"


CODINGS = {"anyill": "Any market by 2019, as in the main tests", "noswuk": "Sweden and the UK coded none",
           "current": "Markets present in 2019–21", "current_noee": "Markets present in 2019–21, Estonia coded none"}


def outcome_window_table(r):
    """code/25: the interaction and the heavy-quarter contrasts under each coding of markets."""
    rows = []
    for _, x in r.iterrows():
        death = {"narc": "narcotic poisoning", "drug": "all drug poisoning", "change": "opioid-use-disorder deaths",
                 "oud": "opioid-use-disorder deaths"}[x.outcome.split("_")[0]]
        if x.model.startswith("interaction"):
            what, value = f"Log prescribing × illicit supply, change in {death}", eci(x.estimate, x.lo, x.hi)
        else:
            oc = "level in 2019–21" if x.outcome.endswith("1921") else "change"
            what = f"With and without a market, {death}, {oc}, {x.model.split(', ')[1]}"
            value = f"medians {x.median_with:.2f} and {x.median_without:.2f}; {int(x.n_with)} with a market of {int(x.n)}"
        rows.append({"Source": x.source, "Coding": CODINGS[x.coding], "Comparison": what, "Estimate (95% CI) or medians": value,
                     "p": pval(x.p)})
    return pd.DataFrame(rows)


def fcodes_table(r):
    """code/29: the main country tests with drug-use F codes added to the outcome, in the main analysis's 60 countries."""
    r = r[~r.outcome.str.endswith("all countries passing")]
    out = []
    for _, x in r.iterrows():
        if x.model.startswith("interaction"):
            what, value = "Log prescribing × illicit supply, change", eci(x.estimate, x.lo, x.hi)
        else:
            oc = "level in 2019–21" if x.model.endswith("1921") else "change"
            what = f"Heaviest quarter with and without a market by 2019, {oc}"
            value = f"medians {x.median_with:.2f} and {x.median_without:.2f}; highest without, {x.max_without:.2f}"
        out.append({"Outcome": x.outcome[0].upper() + x.outcome[1:], "Comparison": what, "Estimate (95% CI) or medians": value, "p": pval(x.p)})
    return pd.DataFrame(out)


# The supply coding's notes, in plain sentences. Each key's first item is code/05's note as written; the second is what the appendix prints.
NOTES = {
    "Australia": ("Fentanyl use noted, illicit or diverted unclear. Non-prescribed fentanyl among people who inject drugs, 5-9% a year nationally in 2018-23; fentanyl among clandestine-laboratory substances in 2017-18, with no count; no illicit market reported",
                  "Fentanyl use is reported, but whether it is illicit or diverted is unclear. Between 5% and 9% of people who inject drugs in the national sample used non-prescribed fentanyl each year in 2018–23, and fentanyl was among the substances of clandestine laboratories in 2017–18, without a count. No illicit market is reported."),
    "Austria": ("Treatment data count non-prescribed fentanyl (12 mentions in 2021); fentanyl in 5% of drug-death toxicology in 2021; high-potency painkillers like fentanyl called isolated cases",
                "Treatment data counted non-prescribed fentanyl 12 times in 2021, and fentanyl appeared in 5% of drug-death toxicology that year. The reports call high-potency painkillers such as fentanyl isolated cases."),
    "Belgium": ("The treatment register merges illicit and diverted fentanyl; under 0.1% of treatment episodes in 2022",
                "The treatment register does not separate illicit from diverted fentanyl, which accounted for under 0.1% of treatment episodes in 2022."),
    "Bulgaria": ("A brief outbreak in the early 2010s, during a heroin shortage. Production facilities have also been seized in Bulgaria, at dates not given",
                 "A brief outbreak came in the early 2010s, during a heroin shortage. Production facilities have also been seized in Bulgaria, at dates the source does not give."),
    "Canada": ("British Columbia first; Alberta and Ontario follow. Checked against the Drug Analysis Service below",
               "Illicit fentanyl reached British Columbia first, then Alberta and Ontario. The dates were checked against Health Canada's Drug Analysis Service."),
    "Estonia": ("First seizures 2001-2002, dominant since 2002 after the Taliban ban cut heroin; supply severely disrupted in 2017-2018",
                "Fentanyl was first seized in 2001–02 and has dominated the market since 2002, after the Taliban's ban cut the heroin supply. Supply was severely disrupted in 2017–18."),
    "Finland": ("RAND's case of a conspicuous absence: buprenorphine replaced heroin from 2002. Mounteney reports an outbreak of fentanyl deaths. 3-methylfentanyl first seized in 2001, with later detections isolated",
                "RAND's review treats Finland as a conspicuous absence, since buprenorphine replaced heroin from 2002. Mounteney and colleagues report an outbreak of fentanyl deaths. 3-methylfentanyl was first seized in 2001, and later detections were isolated."),
    "France": ("One small fentanyl laboratory dismantled in 2018 (episodic in the alternative coding); the saved French reports don't mention fentanyl",
               "One small fentanyl laboratory was dismantled in 2018, which makes France episodic in the alternative coding. The French reports we read do not mention fentanyl."),
    "Germany": ("Misuse linked with diverted pharmaceuticals, Bavaria especially; an outbreak of deaths in the same period as the Bulgarian and Slovak ones. Small fentanyl laboratories dismantled in 2007 and 2015 (episodic in the alternative coding)",
                "Misuse is linked with diverted pharmaceuticals, especially in Bavaria, and an outbreak of deaths came in the same period as those in Bulgaria and Slovakia. Small fentanyl laboratories were dismantled in 2007 and 2015, which makes Germany episodic in the alternative coding."),
    "Greece": ("An illicit production site seized; no market reported; the 2020 national report doesn't mention fentanyl",
               "An illicit production site was seized, but no market is reported, and the 2020 national report does not mention fentanyl."),
    "Italy": ("Sporadic analogue detections only (first analogue deaths notified in 2018); non-prescribed fentanyl the main drug for 6 people in treatment in 2021; the 2021 wastewater survey tested for analogues",
              "Only sporadic detections of analogues are reported, and the first deaths from analogues were notified in 2018. Non-prescribed fentanyl was the main drug for 6 people in treatment in 2021, and the 2021 wastewater survey tested for analogues."),
    "Latvia": ("Carfentanil first found in 2012 but scarce until 2015; analogues still growing at the expense of heroin in 2019. The first year dates the market, not earlier isolated seizures: 43 g of 3-methylfentanyl was seized in 2006",
               "Carfentanil was first found on the market in 2012 but stayed scarce until 2015, and analogues were still displacing heroin in 2019. The first year dates the market and leaves out an earlier isolated seizure, 43 g of 3-methylfentanyl in 2006."),
    "Lithuania": ("A recent increase in carfentanil trafficking cases, seizures and toxicology findings, reported in 2018; scale and dates not given",
                  "A recent increase in carfentanil trafficking cases, seizures and toxicology findings was reported in 2018, without its scale or dates."),
    "Portugal": ("An illicit production site seized; no market reported",
                 "An illicit production site was seized, but no market is reported."),
    "Slovakia": ("A brief outbreak in the early 2010s, during a heroin shortage; an illicit laboratory dismantled in 2011",
                 "A brief outbreak came in the early 2010s, during a heroin shortage, and an illicit laboratory was dismantled in 2011."),
    "Spain": ("OEDA counts fentanyl among therapeutic opioid painkillers; 12 emergency episodes in 2022; no illicit manufacture reported",
              "Spain's drug observatory counts fentanyl among therapeutic opioid painkillers; it recorded 12 emergency episodes in 2022 and reports no illicit manufacture."),
    "Sweden": ("A separate online market, dismantled by prosecutions in 2017-2018; no reports after early 2018. Brief episodes in the mid-1990s and early 2000s. Two small fentanyl laboratories dismantled in 2016",
               "A separate online market was dismantled by prosecutions in 2017–18, and nothing is reported after early 2018. Brief episodes came in the mid-1990s and early 2000s, and two small fentanyl laboratories were dismantled in 2016."),
    "United Kingdom": ("An outbreak of fentanyl-related deaths, in the same period as the Bulgarian and Slovak outbreaks (early 2010s), with no earlier history of misuse. A fentanyl processing centre dismantled and a wholesale online vendor arrested in April 2017",
                       "An outbreak of fentanyl-related deaths came in the early 2010s, in the same period as those in Bulgaria and Slovakia, with no earlier history of misuse. A fentanyl processing centre was dismantled and a wholesale online vendor arrested in April 2017."),
    "United States of America": ("Eastern states first. National law-enforcement fentanyl submissions rose from 1,015 in 2013 to 5,343 in 2014. The first year dates the lasting market, not an earlier episode: 1,013 deaths in six jurisdictions in April 2005 to March 2007 were attributed to non-pharmaceutical fentanyl, some of it believed to come from clandestine production in Mexico, where a facility in Toluca was closed in May 2006",
                                 "Illicit fentanyl reached the eastern states first. National law-enforcement fentanyl submissions rose from 1015 in 2013 to 5343 in 2014. The first year dates the lasting market and leaves out an earlier episode: CDC attributed 1013 deaths in six jurisdictions from April 2005 to March 2007 to non-pharmaceutical fentanyl, some of it believed to come from clandestine production in Mexico, where a production facility in Toluca was closed in May 2006."),
}

# Source keys in the supply coding, as cited in the appendix, and the full references they stand for (A11).
SOURCES = {
    "Pardo2019": ("Pardo et al. 2019", "Pardo B, Taylor J, Caulkins JP, Kilmer B, Reuter P, Stein BD. The future of fentanyl and other synthetic opioids. Santa Monica, CA: RAND Corporation, 2019."),
    "Mounteney2015": ("Mounteney et al. 2015", "Mounteney J, Giraudon I, Denissov G, Griffiths P. Fentanyls: are we missing the signs? Highly potent and on the rise in Europe. *Int J Drug Policy* 2015; **26:** 626–31."),
    "EDMR2019": ("EU Drug Markets Report 2019", "European Monitoring Centre for Drugs and Drug Addiction, Europol. EU Drug Markets Report 2019. Luxembourg: Publications Office of the European Union, 2019."),
    "CDC2008": ("CDC 2008", "Centers for Disease Control and Prevention. Nonpharmaceutical fentanyl-related deaths—multiple states, April 2005–March 2007. *MMWR Morb Mortal Wkly Rep* 2008; **57:** 793–96."),
    "Gladden2016": ("Gladden et al. 2016", "Gladden RM, Martinez P, Seth P. Fentanyl law enforcement submissions and increases in synthetic opioid-involved overdose deaths—27 states, 2013–2014. *MMWR Morb Mortal Wkly Rep* 2016; **65:** 837–43."),
    "Ojanpera2008": ("Ojanperä et al. 2008", "Ojanperä I, Gergov M, Liiv M, Riikoja A, Vuori E. An epidemic of fatal 3-methylfentanyl poisoning in Estonia. *Int J Legal Med* 2008; **122:** 395–400."),
    "acic_iddr_2017-18": ("Illicit Drug Data Report 2017–18", "Australian Criminal Intelligence Commission. Illicit Drug Data Report 2017–18. Canberra: ACIC. https://www.acic.gov.au (accessed Nov 1, 2026)."),
    "idrs_national_2023": ("Australian Drug Trends 2023", "Sutherland R, Uporova J, et al. Australian Drug Trends 2023: key findings from the National Illicit Drug Reporting System interviews. Sydney: National Drug and Alcohol Research Centre, UNSW Sydney, 2023."),
    "at_epi_2022": ("Epidemiologiebericht Sucht 2022", "Gesundheit Österreich. Epidemiologiebericht Sucht 2022: illegale Drogen, Alkohol und Tabak. Vienna: Federal Ministry of Social Affairs, Health, Care and Consumer Protection, 2022."),
    "at_epi_2025": ("Epidemiologiebericht Sucht 2025", "Gesundheit Österreich. Epidemiologiebericht Sucht 2025: illegale Drogen, Alkohol, Nikotin und Glücksspiel. Vienna: Federal Ministry of Labour, Social Affairs, Health, Care and Consumer Protection, 2025."),
    "be_tdi_2022": ("TDI annual report 2022", "Sciensano. L'enregistrement TDI en Belgique: rapport annuel, année d'enregistrement 2022. Brussels: Sciensano."),
    "fr_trend_bordeaux_2023": ("TREND Bordeaux 2023", "Perrin S. Substances psychoactives, usagers et marchés: tendances récentes à Bordeaux et en Nouvelle-Aquitaine en 2023. Paris: Observatoire français des drogues et des tendances addictives, 2024."),
    "gr_ektepn_2020": ("EKTEPN annual report 2020", "Greek Documentation and Monitoring Centre for Drugs and Addictions, EKTEPN. Annual report 2020, data for 2019. Athens: University Mental Health, Neurosciences and Precision Medicine Research Institute Costas Stefanis."),
    "it_relazione_2022": ("Relazione al Parlamento 2022", "Dipartimento per le Politiche Antidroga. Relazione annuale al Parlamento sul fenomeno delle tossicodipendenze in Italia, anno 2022. Rome: Presidenza del Consiglio dei Ministri, 2022."),
    "it_dcsa_2019": ("DCSA annual report 2019", "Direzione Centrale per i Servizi Antidroga. Relazione annuale 2019. Rome: Ministero dell'Interno, 2019."),
    "oeda_informe_2024": ("OEDA Informe 2024", "Observatorio Español de las Drogas y las Adicciones. Informe 2024: alcohol, tabaco y drogas ilegales en España. Madrid: Delegación del Gobierno para el Plan Nacional sobre Drogas, 2024."),
    "oeda_mortalidad_2025": ("OEDA mortality indicator 2025", "Observatorio Español de las Drogas y las Adicciones. Informe 2025: indicador mortalidad por reacción aguda a sustancias psicoactivas, 1983–2023. Madrid: Delegación del Gobierno para el Plan Nacional sobre Drogas, 2025."),
    "LatviaNR2007": ("Latvia national report 2007", "Latvian National Focal Point. 2007 national report (2006 data) to the EMCDDA by the Reitox National Focal Point: Latvia. Riga, 2007."),
    "Mounteney2012": ("Mounteney et al. 2012", "Mounteney J, Evans-Brown M, Giraudon I. Fentanyl in Europe: EMCDDA Trendspotter study. Lisbon: European Monitoring Centre for Drugs and Drug Addiction, 2012."),
    "AbelOllo2025": ("Abel-Ollo et al. 2025", "Abel-Ollo K, Tõnisson M, Rausberg P, et al. The nitazene epidemic in Estonia: a first report. *Eur J Public Health* 2025; **35:** 1233–40."),
    "RACyclopropyl2018": ("EMCDDA 2018, cyclopropylfentanyl", "European Monitoring Centre for Drugs and Drug Addiction. Report on the risk assessment of cyclopropylfentanyl. Luxembourg: Publications Office of the European Union, 2018."),
    "RAMethoxyacetyl2018": ("EMCDDA 2018, methoxyacetylfentanyl", "European Monitoring Centre for Drugs and Drug Addiction. Report on the risk assessment of methoxyacetylfentanyl. Luxembourg: Publications Office of the European Union, 2018."),
    # Cited in A4's European check; read among the downloaded sources's euda_edr_pages snapshots of 2026-11-09
    "EDR2025": ("European Drug Report 2025", "European Union Drugs Agency. European Drug Report 2025: trends and developments. Heroin and other opioids. Lisbon: EUDA, 2025. https://www.euda.europa.eu/publications/european-drug-report/2025/heroin-and-other-opioids_en (accessed Nov 9, 2026)."),
    "EDR2026": ("European Drug Report 2026", "European Union Drugs Agency. European Drug Report 2026: trends and developments. Drug-induced deaths. Lisbon: EUDA, 2026. https://www.euda.europa.eu/publications/european-drug-report/2026/drug-induced-deaths_en (accessed Nov 9, 2026)."),
}
A4_SOURCES = ["EDR2025", "EDR2026", "Pardo2019", "AbelOllo2025", "RACyclopropyl2018", "RAMethoxyacetyl2018"]  # the last two in A3's opening
KEY = re.compile(r"(" + "|".join(re.escape(k) for k in sorted(SOURCES, key=len, reverse=True)) + r")")


def cite(sources):
    """'Pardo2019 p. 74 (citing AIHW 2018); other_settings acic_iddr_2017-18 p. 133' -> 'Pardo et al. 2019, p 74, citing AIHW 2018; ...'."""
    if not isinstance(sources, str):
        return ""
    s = sources.replace("other_settings ", "")
    s = re.sub(r"\s*\(citing ([^)]+)\)", r", citing \1", s)
    s = s.replace("EMCDDA 2018e", "EMCDDA 2018")
    s = re.sub(r"\bpp\. (\d+)-(\d+)", lambda m: f"pp {pages(m.group(1), m.group(2))}", s)
    s = re.sub(r"\bpp?\. ", lambda m: m.group(0).replace(".", ""), s)
    s = KEY.sub(lambda m: SOURCES[m.group(1)][0] + ",", s)
    s = re.sub(r",\s*(;|$)", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    return re.sub(r"\s+", " ", s).strip().rstrip(",")


# Models and terms in plain words. Countries (code/06).
C_SAMPLE = {"all": "All countries", "pop 1m+": "Population above 1 million", "review scope": "Within the reviews' scope", "no Estonia": "All but Estonia"}
C_OUTCOME = {"narc_change": "Narcotic poisoning", "drug_change": "All drug poisoning"}
C_MODEL = {"P1": "Prescribing", "P2 established": "Prescribing by established market", "P2 any illicit": "Prescribing by illicit supply",
           "P2 no illicit supply": "Prescribing, countries without illicit supply",
           "P2 any illicit, UK coded none": "Prescribing by illicit supply, the UK coded as none",
           "P2 established, alternative coding": "Prescribing by established market, alternative coding",
           "P2 any illicit, alternative coding": "Prescribing by illicit supply, alternative coding",
           "P2 any illicit or transit": "Prescribing by illicit supply or transit",
           "P4": "Oral opioids and pharmaceutical fentanyl", "P4 within supply": "Oral opioids and pharmaceutical fentanyl by illicit supply",
           "P2 pre-arrival exposure": "Prescribing before supply arrived, by established market"}
C_TERM = {"lrx_pre": "Log prescribing", "lgdp": "Log GDP per capita", "rho": "Spearman correlation with prescribing",
          "est": "Established market", "anyill": "Illicit supply", "anyill_nouk": "Illicit supply, the UK coded as none",
          "est_alt": "Established market, alternative coding", "anyill_alt": "Illicit supply, alternative coding",
          "anyill_transit": "Illicit supply or transit", "loral_pre": "Log oral oxycodone and hydrocodone",
          "lpatch_pre": "Log pharmaceutical fentanyl", "lrx_pre_arrival": "Log prescribing before supply arrived"}
# US states (code/07).
S_OUTCOME = {"overdose rise 2011-13 to 2017-19": "Change in overdose mortality, 2011–13 to 2017–19",
             "log synthetic-opioid rate 2014-20": "Synthetic-opioid mortality, 2014–20",
             "overdose rise 2008-10 to 2011-13 (before fentanyl)": "Change in overdose mortality, 2008–10 to 2011–13, before fentanyl",
             "overdose rate 2008-10": "Overdose mortality, 2008–10", "overdose rate 2011-13": "Overdose mortality, 2011–13",
             "overdose rate 2014-16": "Overdose mortality, 2014–16", "overdose rate 2017-19": "Overdose mortality, 2017–19",
             "change in the gradient from 2008-10": "Overdose mortality, 2008–10 to 2017–19",
             "gradient after the state's arrival": "Overdose mortality, 2008–10 to 2017–19",
             "after arrival, beyond the calendar change in the gradient": "Overdose mortality, 2008–10 to 2017–19"}
ADJ = "adjusted for arrival date, distress and market type"
S_MODEL = {"P2": "Dispensing", "P2 + arrival": "Dispensing, adjusted for arrival date", "P2 + arrival, distress, market": f"Dispensing, {ADJ}",
           "P2 + arrival, distress": "Dispensing, adjusted for arrival date and distress",
           "P4 pills beside all prescribing": f"Pills beside all dispensing, {ADJ}",
           "P4 oral oxycodone and hydrocodone against fentanyl (ARCOS grams)": f"Oral opioids against fentanyl base, {ADJ}",
           "P4 oral alone": f"Oral opioids alone, {ADJ}", "P4 fentanyl alone": f"Fentanyl base alone, {ADJ}",
           "P2 gradient across states": "Slope across states",
           "change in the gradient from 2008-10": "Panel: the slope's change by window from 2008–10",
           "gradient after the state's arrival": "Panel: the slope after arrival",
           "after arrival, beyond the calendar change in the gradient": "Panel: the slope after arrival, beyond its change by window"}
S_TERM = {"lrx": "Log dispensing", "rho": "Spearman correlation with dispensing", "za": "Earlier arrival, per standard deviation",
          "zx": "Economic distress, per standard deviation", "white": "White-powder heroin market",
          "lpills": "Log oxycodone and hydrocodone pills per person, 2006–12", "loral_g": "Log grams of oral oxycodone and hydrocodone",
          "lfent_g": "Log grams of fentanyl base", "lrx_2011": "Log dispensing × 2011–13", "lrx_2014": "Log dispensing × 2014–16",
          "lrx_2017": "Log dispensing × 2017–19", "post": "Windows after arrival", "lrx:post": "Log dispensing × windows after arrival"}
# The triplicate comparison (code/09).
T_WINDOW = {"1985": "1985–89", "1990": "1990–94", "1999": "1999–2001", "2002": "2002–04", "2005": "2005–07", "2008": "2008–10",
            "2011": "2011–13", "2014": "2014–16", "2017": "2017–19"}
T_OUTCOME = {"growth 1995-98 to 2011-13 (before fentanyl)": "Change in overdose mortality, 1995–98 to 2011–13, before fentanyl",
             "growth 2011-13 to 2017-19": "Change in overdose mortality, 2011–13 to 2017–19",
             "growth 1995-98 to 2017-19": "Change in overdose mortality, 1995–98 to 2017–19",
             "log synthetic-opioid rate 2014-20": "Synthetic-opioid mortality, 2014–20"}
T_MODEL = {"triplicate": "Triplicate state", "triplicate, arrival held": "Triplicate state, adjusted for arrival date",
           "triplicate, arrival, distress, market": f"Triplicate state, {ADJ}",
           "gap by window, relative to 1995-98": "Panel: the triplicate states' gap by window, relative to 1995–98",
           "gap by window, arrival held": "Panel: the triplicate states' gap by window, relative to 1995–98, adjusted for arrival"}
T_TERM = {"triplicate": "Triplicate state", "za": "Earlier arrival, per standard deviation", "post": "Windows after arrival",
          **{f"trip_{y}": f"Triplicate state × {w}" for y, w in T_WINDOW.items()}}


def units_states(n, east=False):
    if east:
        return f"{n} eastern states"
    return f"{n} states" if n < 100 else f"49 states, {n // 49} windows each"


def country_tables(r):
    r = r[~r.term.astype(str).str.startswith("C(")].copy()
    parts = r.model.str.split("|").apply(lambda p: [x.strip() for x in p])
    r["kind"], r["outcome"], r["sample"] = parts.str[0], parts.str[1], parts.str[2]
    out = []
    for oc, title in (("narc_change", "narcotic poisoning"), ("drug_change", "all drug poisoning")):
        k = r[r.outcome == oc]
        t = pd.DataFrame({"Sample": k["sample"].map(C_SAMPLE), "Model": k.kind.map(C_MODEL),
                          "Term": k.term.map(lambda x: joinx([C_TERM[p] for p in x.split(":")])),
                          "Estimate (95% CI)": [est(b, se) for b, se in zip(k.b, k.se)], "p": k.p.map(pval), "n": k.n})
        assert not t.isna().any().any(), t[t.isna().any(axis=1)]
        out.append((title, t))
    return out


def state_tables(r):
    r = r[~r.term.astype(str).str.startswith("C(")].copy()
    rows = []
    for _, x in r.iterrows():
        p = [y.strip() for y in x.model.split("|")]
        east = len(p) == 3 and p[1] == "east only"
        outcome = p[-1]
        model = p[1] if p[0] == "P2 steepening" else p[0]
        rows.append(dict(bound=x.bound, Units=units_states(int(x.n), east), Outcome=S_OUTCOME[outcome], Model=S_MODEL[model],
                         Term=S_TERM[x.term] if x.term in S_TERM else joinx([S_TERM[q] for q in x.term.split(":")]),
                         **{"Estimate (95% CI)": est(x.b, x.se), "p": pval(x.p)}))
    t = pd.DataFrame(rows)
    return [(int(b), k.drop(columns="bound")) for b, k in t.groupby("bound")]


H_MODEL = {  # code/19's models, as code/07 labels them, and whether the sample is the eastern states
    "arrival, distress, market": ("P2 + arrival, distress, market", False), "arrival, distress | east only": ("P2 + arrival, distress", True),
    "pills beside all prescribing": ("P4 pills beside all prescribing", False),
    "oral oxycodone and hydrocodone against fentanyl": ("P4 oral oxycodone and hydrocodone against fentanyl (ARCOS grams)", False)}
H_DISTRESS = {"2000 index": "Distress from 2000", "2008-2012 index": "Distress from 2008–12", "both": "Both"}


def hardship_tables(h):
    """code/19: the exposure's estimate in each model that adjusts for distress, under each measure of distress, by suppression bound."""
    h = h[h.term.isin(["lrx", "lpills", "loral_g", "lfent_g"])].copy()
    h["cell"] = [f"{b:.3f} ({lo:.3f} to {hi:.3f})" for b, lo, hi in zip(h.b, h.lo, h.hi)]
    out = []
    for bound, k in h.groupby("bound"):
        w = k.pivot_table(index=["outcome", "model", "term", "n"], columns="distress", values="cell", aggfunc="first").reset_index()
        t = pd.DataFrame({"Units": [units_states(int(n), H_MODEL[m][1]) for n, m in zip(w.n, w.model)], "Outcome": w.outcome.map(S_OUTCOME),
                          "Model": w.model.map(lambda m: S_MODEL[H_MODEL[m][0]]), "Term": w.term.map(S_TERM),
                          **{H_DISTRESS[d]: w[d] for d in H_DISTRESS}})
        assert not t.isna().any().any(), t[t.isna().any(axis=1)]
        out.append((int(bound), t))
    return out


def cardiac_tables(cs, ca):
    """code/20: the share of drug deaths certified under a cardiovascular cause, by state, and the synthetic-opioid models with them added back."""
    pc = lambda v: f"{100 * v:.1f}%"
    pv = lambda p: f"p{pval(p)}" if pval(p).startswith("<") else f"p={pval(p)}"
    sh = []
    for deaths, k in cs.groupby("deaths", sort=False):
        x = k.iloc[0]
        rho = {r.against: f"{r.spearman:.2f} ({pv(r.p)})" for r in k.itertuples()}
        deaths = re.sub(r"(\d{4})-(\d{2,4})", r"\1–\2", deaths)
        sh.append({"Deaths": deaths[0].upper() + deaths[1:], "States listed": int(x.states_listed), "Share in the listed states": pc(x.share_listed_states),
                   "Lowest": pc(x.share_min), "Median": pc(x.share_median), "Highest": pc(x.share_max),
                   "Spearman with dispensing": rho["prescribing 2009-11"], "With arrival date": rho["arrival date"], "With distress": rho["distress"]})
    ca = ca[ca.outcome != "share added back"].copy()
    ca["cell"] = [f"{b:.3f} ({lo:.3f} to {hi:.3f})" for b, lo, hi in zip(ca.b, ca.lo, ca.hi)]
    ca["column"] = [("Published" if o == "published" else f"With them, left-out states at {f}") for o, f in zip(ca.outcome, ca.left_out_states_at)]
    w = ca.drop_duplicates(["formula", "column"]).pivot(index="formula", columns="column", values="cell")
    model = {"lt404 ~ lrx + za": "Dispensing, adjusted for arrival date", "lt404 ~ lrx + za + zx + white": f"Dispensing, {ADJ}"}
    w.index = [model[f.replace("lt404_cv", "lt404")] for f in w.index]
    w = w.groupby(level=0).first()
    ab = pd.DataFrame({"Units": "49 states", "Outcome": "Synthetic-opioid mortality, 2014–20", "Model": w.index, "Term": "Log dispensing",
                       **{c: w[c].values for c in ["Published", "With them, left-out states at 0", "With them, left-out states at 9"]}})
    return pd.DataFrame(sh), ab


def triplicate_tables(r):
    rows = []
    for _, x in r.iterrows():
        p = [y.strip() for y in x.model.split("|")]
        outcome = "Overdose mortality, 1985–89 to 2017–19" if len(p) == 1 else T_OUTCOME[p[0]]
        model = T_MODEL[p[0] if len(p) == 1 else p[1]]
        rows.append(dict(bound=x.bound, Units=units_states(int(x.n)), Outcome=outcome, Model=model, Term=T_TERM[x.term],
                         **{"Estimate (95% CI)": est(x.b, x.se), "p": pval(x.p)}))
    t = pd.DataFrame(rows)
    return [(int(b), k.drop(columns="bound")) for b, k in t.groupby("bound")]


def gbd_table(g):
    model = {"lrx": "Prescribing, adjusted for log GDP per capita", "lrx:anyill": "Prescribing by illicit supply, adjusted for log GDP per capita"}
    rows = []
    for _, x in g.iterrows():
        heavy = "Mann" in x.term
        if x.term == "spearman":
            unit, outcome, m, term = f"{x.n} countries", "Change in opioid-use-disorder deaths, 2010–12 to 2019–21", "Spearman correlation", "Prescribing"
        elif heavy:
            unit = f"{x.n} heaviest prescribers"
            outcome = ("Change in opioid-use-disorder deaths, 2010–12 to 2019–21" if x.outcome == "change"
                       else "Opioid-use-disorder deaths per 100 000, 2019–21")
            m, term = "Mann–Whitney test", "Difference in medians, with against without illicit supply"
        else:
            unit = f"{x.n} countries"
            outcome = ("Change in opioid-use-disorder deaths, 2010–12 to 2019–21" if x.outcome == "change"
                       else "Opioid-use-disorder deaths, 2019–21, log")
            m, term = model[x.term], {"lrx": "Log prescribing", "lrx:anyill": "Log prescribing × illicit supply"}[x.term]
        rows.append({"Units": unit, "Outcome": outcome, "Model": m, "Term": term, "Estimate (95% CI)": est(x.b, x.se), "p": pval(x.p)})
    return pd.DataFrame(rows)


def kg_cell(v):
    """Kilograms in the European seizure table: blank where the country reported no seizure that year."""
    if pd.isna(v):
        return ""
    if v == 0:
        return "0"
    if v < 0.005:
        return "<0.01"
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


EUROPE = ["Estonia", "Latvia", "Bulgaria", "Lithuania", "Slovakia", "Sweden", "United Kingdom"]  # established, then episodic


def europe_part(d):
    """A4's second part, from code/21: the seven European markets after China's class-wide control of 1 May 2019. Returns the
    paragraph and its two tables. Each claim in the paragraph is checked against the data first, so a change upstream stops the
    build instead of leaving the words wrong."""
    r = d.pivot(index="country", columns="year", values="narcotics_per_100k")
    kg = d.pivot(index="country", columns="year", values="kg_reported")
    carf = d.pivot(index="country", columns="year", values="carfentanil_kg")
    assert sorted(EUROPE) == sorted(r.index), f"code/21's European markets changed: {sorted(r.index)}"
    one = lambda v: f"{v:.1f}"
    mean = lambda c, years: r.loc[c, list(years)].mean()
    est, swe = ((mean(c, (2016, 2017)), r.loc[c, 2018], mean(c, (2020, 2021))) for c in ("Estonia", "Sweden"))
    for name, (before, then, after) in (("Estonia", est), ("Sweden", swe)):
        assert then < 0.8 * before and abs(after / then - 1) < 0.15, f"{name}: the fall by 2018, or the level after it, changed"
    ee, se = kg.loc["Estonia"], kg.loc["Sweden"]
    ee_rate = r.loc["Estonia", 2019:2021]
    # Estonia: no annex row for 2018; RAND's forensic figures fill it (10 220 g in 2017, 572 g in January-October 2018, p 81)
    assert pd.isna(ee[2018]) and ee[2019] > 5 and ee[2020] < 1 and ee[2021] < 1 and ee.loc[2022:2024].fillna(0).max() < 0.1, \
        "Estonia's seizures changed"
    assert pd.isna(se[2019]) and se[2018] > 1 and 0.1 < se[2020] < se[2018] and se.loc[2021:2024].max() < 0.1, "Sweden's seizures changed"
    lt, lv = carf.loc["Lithuania"], kg.loc["Latvia", 2019:2021]
    lrate = r.loc["Lithuania", 2015:2021]
    assert lt[2019] > 1 and lt[2020] > 1 and lv.min() > 0.1, "Lithuania's or Latvia's seizures changed"
    assert r.loc["Latvia", 2015:2021].max() < 1.15, "Latvia's rate before 2022 changed"
    low = max(r.loc[c].max() for c in ("Bulgaria", "Slovakia"))
    bg = kg.loc["Bulgaria", 2022]
    assert low < 0.4 and bg > 100 and kg.loc["United Kingdom"].isna().all(), "Bulgaria's, Slovakia's or the UK's rows changed"
    rise = {c: (r.loc[c, 2022:2024].min(), r.loc[c, 2022:2024].max()) for c in ("Estonia", "Latvia")}
    assert all(lo > r.loc[c, 2015:2021].min() for c, (lo, hi) in rise.items()) and kg.loc[["Estonia", "Latvia"], 2022:2024].max().max() < 0.1
    text = ("The second part of this section follows Europe's markets after 1 May 2019, when China controlled all fentanyl-related "
            "substances as a class. It covers the seven European countries that our coding marks as established or episodic markets by "
            "2019, with their fentanyl-type seizures from the same table of the annex and their narcotic-poisoning deaths from the WHO "
            "Mortality Database, both to 2024 where countries reported. In Estonia and Sweden, narcotic-poisoning mortality fell in 2018, "
            f"after arrests of their main distributors and before the class-wide control: from {one(est[0])} per 100 000 in 2016–17 to "
            f"{one(est[1])} in 2018 in Estonia, and from {one(swe[0])} to {one(swe[1])} in Sweden. In Sweden, prosecutions of the three "
            "biggest online sellers of fentanyl analogues in late 2017 and early 2018 ended the analogue market, which had caused 101 "
            "deaths in 2017, and by RAND's writing in 2019 the last fentanyl-related death had come in August 2018 (Pardo et al. 2019, pp "
            "89, 92). In Estonia, police arrested most of the high-level traffickers in 2017, fentanyl was scarce for several months, and "
            "the seizures that Estonia's forensic institute recorded fell from 10.2 kg in 2017 to 0.6 kg in January to October 2018; the "
            "annex has no row for Estonia in 2018. Supply was then partly restored (Pardo et al. 2019, pp 81–82). Estonia seized "
            f"{one(ee[2019])} kg in 2019, but a study of its market dates the end of the fentanyl era to the end of 2017 and reports "
            "that fentanyl was uncommon in 2019 despite one large seizure (Abel-Ollo et al. 2025, pp 1233–34). Its seizures were "
            f"{one(ee[2020])} kg in 2020 and {one(ee[2021])} kg "
            f"in 2021, and its mortality stayed between {one(ee_rate.min())} and {one(ee_rate.max())} per 100 000 in 2019–21, near its "
            f"2018 level. Sweden's seizures, {one(se[2018])} kg in 2018, fell to {one(se[2020])} kg in 2020 and stayed below 0.1 kg a year "
            "from 2021, and its mortality in 2020–21 was also near its 2018 level, at "
            f"{one(swe[2])}. These later falls in seizures fit the class-wide control and continued policing alike.\n\n"
            "In Latvia and Lithuania, whose markets ran on carfentanil, supply continued after the "
            f"control. Lithuania reported {one(lt[2019])} kg of carfentanil in 2019 and {one(lt[2020])} kg in 2020, and the European Drug "
            "Report 2025 records that EUDA's assessment in late 2024 found carfentanil still dominant there; Latvia reported "
            f"{one(lv.min())} to {one(lv.max())} kg of fentanyl-type drugs a year in 2019–21. Lithuania's narcotic-poisoning rate "
            f"had fallen from {one(lrate[2015])} per 100 000 in 2015 to {one(lrate[2018])} in 2018 and stayed between "
            f"{one(lrate.loc[2019:2021].min())} and {one(lrate.loc[2019:2021].max())} through 2021, and Latvia's stayed at or below "
            f"{one(r.loc['Latvia', 2015:2021].max())}. In Bulgaria and Slovakia, whose episodes came in the early 2010s, rates stayed at "
            f"or below {one(low)} per 100 000 throughout, and Bulgaria's {one(bg)} kg in 2022 came while its rate stayed between "
            f"{one(r.loc['Bulgaria', 2021:2023].min())} and {one(r.loc['Bulgaria', 2021:2023].max())} in 2021–23. The UK "
            "reported no fentanyl-type seizure to UNODC in any year, so this check cannot follow its market.\n\n"
            "The falls in mortality in Estonia and Sweden, then, came in 2018, after local arrests and before the class-wide control. "
            "After the control, Estonia's seizures were low and its mortality stayed near its 2018 level, and the carfentanil "
            "markets of Latvia and Lithuania continued. From 2022, after the period this study covers, deaths rose again in Estonia and Latvia, to "
            f"{one(rise['Estonia'][0])}–{one(rise['Estonia'][1])} and {one(rise['Latvia'][0])}–{one(rise['Latvia'][1])} per 100 000, "
            "while their fentanyl-type seizures stayed near zero. The European Drug Report 2026 links nitazenes, synthetic opioids of "
            "another class, to 43 of Estonia's 97 drug-induced deaths in 2024 and to 36 of Latvia's 77; isotonitazene was first detected in "
            "Estonia in October 2019 (Abel-Ollo et al. 2025, p 1234), so a synthetic supply was present before deaths rose. The first table gives seizures "
            "by year, with a blank where a country reported none, and the second gives deaths per 100 000.")
    status = d.drop_duplicates("country").set_index("country").status
    cols = lambda f, m: {"Country": [SHORT.get(c, c) for c in EUROPE], "Status by 2019": [status[c] for c in EUROPE],
                         **{str(y): [f(m.loc[c, y]) for c in EUROPE] for y in sorted(m.columns)}}
    seizures = pd.DataFrame(cols(kg_cell, kg))
    deaths = pd.DataFrame(cols(lambda v: "" if pd.isna(v) else one(v), r))
    return text, seizures, deaths


def main(out, paper, catalog):
    out = pathlib.Path(out)
    parts = ["# Appendix", "",
             "This appendix gives the sources of the data (A1), the check on the extraction of INCB's tables (A2), illicit fentanyl supply by "
             "country with the evidence for each code (A3), the seizure check on that coding, with Europe's markets after China's class-wide "
             "control of 2019 (A4), the national-count check on WHO's death counts (A5), the check on GBD's modelled estimates (A6), and every model we fitted, with its estimate: for countries (A7), for US "
             "states, with three further checks, on distress, on the coding of drug deaths and on the years since arrival (A8), and for the triplicate comparison (A9), and four further tests: within states across counties (A10), "
             "negative-control outcomes (A11), the population at risk (A12) and Canada's provinces (A13). A14 "
             "is the STROBE checklist, and A15 lists the sources cited here. The code "
             "that produces every table is public at https://github.com/jfrancisohalloran/prescribing-fentanyl-mortality.", ""]

    c = pd.read_csv(catalog)
    c = c[c.dataset.isin(DATASETS)].set_index("dataset").reindex(DATASETS)
    c = pd.DataFrame({"Source": [PLAIN_SOURCES[d][0] for d in DATASETS], "Publisher": [PLAIN_SOURCES[d][1] for d in DATASETS],
                      "Retrieved": [retrieved(v) for v in c.retrieved.astype(str)]})
    parts += ["## A1. Data sources", "", "Every dataset came from a public source; the table gives each one, its publisher and the date we retrieved it.", "", md(c), ""]

    x = pd.read_csv(out / "incb_sddd_2000_2014.csv")
    ok = x.adds_up.astype(bool)
    miss = "; ".join(f"{SHORT.get(r.country, r.country)} in {period(r.window)}" for r in x[~ok].itertuples())
    e = x.groupby(["edition", "window"]).agg(n=("country", "count"), match=("adds_up", "sum")).reset_index()
    e = pd.DataFrame({"INCB edition": e.edition, "Years": e.window.str.replace("-", "–"), "Countries": e.n, "Matching the printed total": e.match})
    parts += ["## A2. The check on the INCB extraction", "",
              f"We read each country's consumption from table XIV.1 of INCB's technical report, which is table XI.1 in the 2003 edition and "
              f"XIV.1.a from 2012. Each country's figures for the separate substances should sum to the total INCB prints; buprenorphine, "
              f"where listed, is not part of that total. Summed across substances, {int(ok.sum())} of the {len(x)} country figures matched; "
              f"the other, {miss}, was off by rounding. The table gives the count for each edition and period.", "", md(e), ""]

    a = pd.read_csv(out / "synthetic_arrival.csv")
    k = a[(a.status != "none known") | a.transit | (a.status_alt != a.status) | a.sources.notna()].copy()
    for name, note in k[["country", "note"]].dropna().itertuples(index=False):
        assert name in NOTES and NOTES[name][0] == note, f"code/05's note for {name} changed; update its plain version in NOTES"
    yes = lambda v: "yes" if v else ""
    year = lambda v: "" if pd.isna(v) else str(int(v))
    country = k.country.map(lambda n: SHORT.get(n, n))
    coding = pd.DataFrame({"Country": country, "Status by 2019": k.status, "Alternative coding": k.status_alt,
                           "Transit only": k.transit.map(yes), "Lasting market first seen": k.first_seen.map(year), "At scale": k.at_scale.map(year),
                           "Ended": k.ended.map(year), "Product": k["product"].fillna("")})
    evidence = pd.DataFrame({"Country": country, "Sources": k.sources.map(cite),
                             "Note": k.apply(lambda r: NOTES[r.country][1] if r.country in NOTES else "", axis=1)})
    evidence = evidence[(evidence.Sources != "") | (evidence.Note != "")]
    parts += ["## A3. Illicit fentanyl supply by country, with its evidence", "",
              "We coded each country's illicit fentanyl supply from seizure, forensic and market evidence, never from deaths. The status by "
              "2019 is established, episodic, diverted only, unclear or none known, as the Methods define them. The alternative coding counts "
              "dismantled laboratories as episodes and Lithuania as an established market. Transit-only countries seize fentanyl in transit "
              "without a known consumer market, and they count as supply only in a sensitivity analysis. The first table gives each code; the "
              "second gives its sources, by author and page, with the full references in A15, and a note on the evidence. Pages are printed "
              "page numbers for RAND's review, the journal articles and the EU Drug Markets Report, and pages of the file for the "
              "national reports. Countries not listed "
              "are coded as none known, meaning that no market was reported by 2019. The coding covers fentanyl and its analogues, not "
              "nitazenes, and it codes markets, not single detections: EMCDDA's risk assessments of 2018 record cyclopropylfentanyl "
              "or methoxyacetylfentanyl detected in Austria, Belgium, Denmark, Finland and France, which are coded here as having no "
              "market.", "", md(coding), "", md(evidence), ""]
    # The author's check of the passages behind the codes: the rows of the check sheet, when the sheet is at hand
    sheet = pathlib.Path(paper).parent / "docs" / "label_check_for_John_2026-11-19.md"
    if sheet.exists():
        rows, last = [], ""
        for line in sheet.read_text().splitlines():
            m = re.match(r"^\| (U0|\d+) \| (.*)\|$", line)
            if m:
                c = [x.strip() for x in m.group(2).split("|")]
                last = c[0] or last
                rows.append({"Country": last, "Code or claim": "; ".join(x for x in (c[1], c[2]) if x), "Source and page": c[3],
                             "What the page says": c[4].replace(" (find the words)", "")})
        assert len(rows) == 35, len(rows)
        parts += ["The author checked the passages behind the codes of 20 countries, the nine with a market and the 11 heaviest "
                  "prescribers without one, against their sources, page by page, on 19 November 2026. The table gives the 35 passages "
                  "checked; all agreed with the codes.", "", md(pd.DataFrame(rows)), ""]

    u = pd.read_csv(out / "unodc_fentanyl_seizures.csv")
    u = pd.DataFrame({"Country": u.Country.map(lambda n: SHORT.get(n, n)), "Status by 2019": u.status, "Years reported": u.years_reported,
                      "First year": u.first_year.map(year), "Kilograms, 2015–21": u.kg_2015_21.round(1),
                      "Grams per million people a year": u.g_per_million_per_year.round(1)})
    e_text, e_seized, e_deaths = europe_part(pd.read_csv(out / "europe_after_2019.csv"))
    parts += ["## A4. Fentanyl seizures reported to UNODC", "",
              "UNODC's seizure data are a check on the supply coding that does not depend on the reviews. We calculated each country's totals "
              "and rates from table 7.1 of UNODC's World Drug Report 2026 statistical annex, with population from the World Bank; both are "
              "listed in A1. The table gives every country that "
              "reported a fentanyl-type seizure in 2015–21, with its status in our coding.", "", md(u), "",
              e_text, "", "Fentanyl-type seizures reported to UNODC, kilograms, by year:", "", md(e_seized), "",
              "Narcotic-poisoning deaths per 100 000, WHO Mortality Database, by year:", "", md(e_deaths), ""]

    ed = pd.read_csv(out / "euda_drd_check.csv")
    lat = ed[(ed.country == "Latvia") & ed.year.between(2010, 2021)]
    lat = pd.DataFrame({"Year": lat.year.astype(int), "National count, EUDA": lat.euda_national.round(0).astype("Int64"),
                        "Narcotic poisoning, WHO": lat.narcotics.round(0).astype("Int64"), "All drug poisoning, WHO": lat.drug_poisoning.round(0).astype("Int64")})
    parts += ["## A5. National counts of drug deaths against WHO's coding", "",
              "Latvia is the established market whose deaths stayed low in WHO's coding. The table compares its national count of "
              "drug-induced deaths, from EUDA's table DRD-33, with WHO's counts of narcotic and all drug poisoning.", "", md(lat), ""]
    nw = pd.read_csv(out / "norway_national.csv")
    nt = pd.DataFrame({"Country": nw.country, "Illicit fentanyl market by 2019": nw.status.map(lambda v: v[0].upper() + v[1:]),
                       "2010–12": nw.rate_2010_12.map(lambda v: f"{v:.2f}"),
                       "2019–21": nw.rate_2019_21.map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
                       "Change, log ratio": nw.change.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")})
    parts += ["Norway is among the world's 15 heaviest consumers of opioid analgesics but is not among the 60 countries: WHO's file holds "
              "only its condensed cause list, and only to 2016. The table gives national counts of drug-induced deaths per 100 000, all "
              "ages, from EUDA's table DRD-33 over the World Bank's population, for the European heavy prescribers with Norway added. "
              "EUDA has no UK count after 2018, and Belgium's rate for 2019–21 rests on a single year.", "", md(nt), ""]

    g = pd.read_csv(out / "gbd_check_results.csv")
    parts += ["## A6. The check on GBD's modelled estimates", "",
              "GBD 2023 models death rates from opioid use disorders for 178 of the countries with prescribing data, 170 of them with GDP. "
              "The models mirror those for WHO's counts: linear regressions with robust standard errors, a Spearman correlation, and the "
              "comparison of the heaviest-prescribing quarter with and without illicit supply.", "", md(gbd_table(g)), ""]

    r = pd.read_csv(out / "tests_results.csv")
    tabs = country_tables(r)
    parts += ["## A7. Every model for countries", "",
              "Each country model is a linear regression, with robust standard errors, of the change in mortality from 2010–12 to 2019–21, as "
              "a log ratio, on prescribing in 2000–11, entered as the logarithm of one plus S-DDD and adjusted for log GDP per capita. Spearman "
              "correlations stand beside them. Each model was fitted in four samples and under the supply codings that A3 describes; oral "
              "opioids are oxycodone and hydrocodone, and prescribing before supply arrived keeps only the periods that ended before illicit "
              f"fentanyl was first seen. The first table gives narcotic poisoning, the second all drug poisoning, {sum(len(t) for _, t in tabs)} "
              "rows in all.", ""]
    for title, t in tabs:
        parts += [f"Change in {title}, 2010–12 to 2019–21:", "", md(t), ""]
    ac = pd.read_csv(out / "audit_checks.csv")  # read here, used in A7 and A9
    ci = ac[ac.check == "country interaction"]
    lo = pd.DataFrame({"Countries": ci.model.map(lambda v: v[0].upper() + v[1:]), "Number": ci.n, "Outcome": "Change in narcotic-poisoning mortality",
                       "Term": "Log prescribing × illicit supply", "Estimate (95% CI)": [est(b, se) for b, se in zip(ci.b, ci.se)],
                       "p": ci.p.map(pval)})
    parts += ["Three further checks close this section. The first: few countries had illicit supply, so the model of prescribing by illicit supply "
              "was fitted again with the USA, Canada or both left out. The estimate keeps its size and loses its precision.", "", md(lo), ""]
    ow = pd.read_csv(out / "outcome_window_results.csv")
    ow = ow[~ow.model.str.contains("13 heaviest") | (ow.coding == "anyill")]
    parts += ["The second concerns the coding of markets against the outcome window. The main tests count a market present at any time by "
              "2019, so Sweden's, dismantled in 2017–18, and the UK's, whose deaths in 2019–21 are not known to be fentanyl's, count as "
              "supply. The table refits the interaction and the heavy-quarter contrasts with Sweden and the UK coded none; with markets "
              "present in 2019–21, the USA, Canada, Estonia, Latvia and Lithuania; and with Estonia also coded none, since its fentanyl era "
              "is dated to the end of 2017. The contrast among the 13 heaviest prescribers of the 60 is given once, since each coding "
              "leaves the same two markets among them. GBD's rows use its 45 heaviest-prescribing countries.", "", md(outcome_window_table(ow)), ""]
    fcd = pd.read_csv(out / "fcodes_results.csv")
    parts += ["The third concerns deaths that some countries code to drug use rather than to poisoning. The table refits the main tests "
              "with the F codes for drug use added to the outcome: to narcotic poisoning, F11–F12, F14–F16 and F19, the underlying causes "
              "of EUDA's Selection B, whose further conditions on multiple causes WHO's file cannot apply; and to all drug poisoning, "
              "F11–F16 and F18–F19, the UK's definition. The 60 countries are those of the main tests; the added codes would admit a few "
              "more, which change nothing material.", "", md(fcodes_table(fcd)), ""]

    s = pd.read_csv(out / "us_states_results.csv")
    stabs = state_tables(s)
    parts += ["## A8. Every model for US states", "",
              "Dispensing and distress are averaged to states over counties with at least 1,000 non-Hispanic white residents in the 2000 census; Adams, Jefferson and Weld counties, Colorado, are absent because the NCHS model-based estimates report them combined with Boulder and Broomfield, and Bedford and Clifton Forge cities, Virginia, because they later merged into their counties. "
              "Each US state model is a linear regression, with robust standard errors, of a state's mortality on its opioid dispensing in "
              "2009–11, entered as a logarithm. Arrival date enters as how early illicit fentanyl arrived, and economic distress as the state's "
              "average; both are in standard deviations. Market type marks the white-powder heroin markets east of the Mississippi, and the "
              "eastern states are those markets. The panels of states by window have state and window fixed effects and errors clustered by "
              "state. WONDER suppresses county counts of one to nine deaths, and every model was fitted with those counts entered at one, five "
              f"and nine; the three tables give each, {sum(len(t) for _, t in stabs)} rows in all.", ""]
    for bound, t in stabs:
        parts += [f"Suppressed county counts entered at {bound}:", "", md(t), ""]

    hs = pd.read_csv(out / "hardship_2010_summary.csv").iloc[0]
    htabs = hardship_tables(pd.read_csv(out / "hardship_2010_check.csv"))
    parts += ["Three further checks close this section. The first concerns economic distress, which the models above measure from the 2000 "
              "census. We rebuilt the same index, the first principal component of the same six measures, from the American Community "
              "Survey's 2008–2012 summary file and the Census Bureau's 2010 poverty estimates for the same counties; both sources are listed "
              f"in A1. Across the {int(hs.counties)} counties with dispensing data, it correlates {hs.r_county:.2f} with the 2000 index, and "
              f"across the {int(hs.states)} states, averaged in the same way, {hs.r_state:.2f}. Each model that adjusts for distress was "
              "fitted again with the 2008–12 index and with both indexes, which also adjusts for the change between them. The tables give "
              f"the exposure's estimate with 95% CI under each, {sum(len(t) for _, t in htabs)} rows in all.", ""]
    for bound, t in htabs:
        parts += [f"Distress measured in 2000 and in 2008–12, suppressed county counts entered at {bound}:", "", md(t), ""]
    share_t, add_t = cardiac_tables(pd.read_csv(out / "cardiac_coding_shares.csv"), pd.read_csv(out / "cardiac_coding_addback.csv"))
    parts += ["The second check concerns drug deaths that counts by underlying cause leave out: drug poisonings recorded as contributing "
              "to a death whose underlying cause is cardiovascular. CDC WONDER gives these deaths by state, listing a state only when it "
              "has ten or more. The first table gives, for all drug poisonings and for synthetic opioids, the share of drug deaths that a "
              "count by underlying cause misses, by state, and that share's Spearman correlation with dispensing, arrival date and "
              "distress. The second "
              "adds the synthetic-opioid deaths among them back to each state's synthetic-opioid mortality in 2014–20, with the states that "
              "WONDER leaves out entered at zero and at nine, and refits the two dispensing models of table 2.", "", md(share_t), "", md(add_t), ""]

    fx = pd.read_csv(out / "fixed_span.csv")
    fx = fx[fx.span != "2014-20 pooled"]
    what = {"arrival date": ("None", "Earlier arrival, per standard deviation"), "market type": ("None", "White-powder heroin market"),
            "dispensing": ("None", "Log dispensing, 2009–11"), "dispensing, arrival date": ("Arrival date", "Log dispensing, 2009–11"),
            "dispensing, arrival date, distress, market type": ("Arrival date, distress, market type", "Log dispensing, 2009–11"),
            "triplicate": ("None", "Triplicate state"), "triplicate, arrival date": ("Arrival date", "Triplicate state")}
    ft = pd.DataFrame({"Synthetic-opioid mortality, log": fx.span.str.replace("-", "–").map(lambda v: v[0].upper() + v[1:]),
                       "Hidden counts entered at": fx.hidden, "Term": fx.model.map(lambda v: what[v][1]),
                       "Adjusted for": fx.model.map(lambda v: what[v][0]), "Estimate (95% CI)": [est(b, se) for b, se in zip(fx.b, fx.se)],
                       "p": fx.p.map(pval), "R²": fx.r2.map(lambda v: f"{v:.2f}")})
    pooled = pd.read_csv(out / "fixed_span.csv").query("span == '2014-20 pooled' and model == 'arrival date'").r2
    k13 = fx[(fx.span == "years 1-3 after the arrival year") & (fx.model == "arrival date")].r2
    parts += ["The third check concerns the years since arrival. A rate pooled over 2014–20 gives a state reached in 2019 two years with "
              "fentanyl and a state reached in 2014 seven, so the arrival date's hold on pooled rates is mostly years of exposure: it "
              f"explains {100 * pooled.min():.0f}% of the variation between states in the pooled rate and {100 * k13.min():.0f}% to "
              f"{100 * k13.max():.0f}% of the rate over the three calendar years after each state's own arrival year. We therefore fitted "
              "the models of dispensing, of triplicate status and of market type on that rate, and on the rate over years two to four. "
              "Deaths by state and year are from CDC WONDER, to 2020 from the final file and for 2021–24 from monthly counts, both listed "
              "in A1, with the counts of one to nine that WONDER hides entered at one and at nine. In all 49 states, dispensing and "
              "triplicate status are not clearly associated with mortality at the same years after arrival, though without the District of "
              "Columbia the dispensing estimate is larger, as the end of this section shows; states in the white-powder heroin markets "
              f"east of the Mississippi have higher rates. The table gives each estimate, {len(ft)} rows in all.", "", md(ft), ""]
    fc = pd.read_csv(out / "fixed_span_certain.csv")
    names = {"NV": "Nevada", "ND": "North Dakota", "SD": "South Dakota", "UT": "Utah"}
    left = [names[k] for k in fc.left_out.iloc[0].split()]
    assert len(left) == 4, left
    ct = pd.DataFrame({"Synthetic-opioid mortality, log": fc.span.str.replace("-", "–").map(lambda v: v[0].upper() + v[1:]),
                       "Hidden counts entered at": fc.hidden, "Term": fc.model.map(lambda v: what[v][1]),
                       "Adjusted for": fc.model.map(lambda v: what[v][0]), "Estimate (95% CI)": [est(b, se) for b, se in zip(fc.b, fc.se)],
                       "p": fc.p.map(pval), "R²": fc.r2.map(lambda v: f"{v:.2f}"), "States": fc.n})
    parts += ["The arrival dates carry error. Resampling each state's laboratory reports gives every date a standard deviation: across "
              f"the 49 states its median is {fc.median_sd.iloc[0]:.1f} years, and it exceeds a year in four, "
              f"{', '.join(left[:-1])} and {left[-1]}, all states with late dates and few reports. The table repeats the arrival and "
              "dispensing models without those states. The dates and their standard deviations are in the code archive.", "", md(ct), ""]
    sc = pd.read_csv(out / "fixed_span_scaled.csv")
    sc = sc[sc.term.isin(["lrx", "triplicate"]) & sc.model.isin(["dispensing, arrival date", "dispensing, arrival date, distress, market type",
                                                                  "triplicate, arrival date"])]
    st = pd.DataFrame({"Synthetic-opioid mortality, log": sc.span.str.replace("-", "–").map(lambda v: v[0].upper() + v[1:]),
                       "Divided by the national rate": sc.scaled_by_national_rate.map({True: "Yes", False: "No"}),
                       "Hidden counts entered at": sc.hidden, "Term": sc.model.map(lambda v: what[v][1]),
                       "Adjusted for": sc.model.map(lambda v: what[v][0]), "Estimate (95% CI)": [eci(b, lo, hi) for b, lo, hi in zip(sc.b, sc.lo, sc.hi)],
                       "Ratio, 75th to 25th percentile of dispensing (95% CI)": [
                           "" if pd.isna(w) else f"{w:.2f} ({a:.2f} to {z:.2f})" for w, a, z in zip(sc.wave_ratio, sc.wave_ratio_lo, sc.wave_ratio_hi)]})
    rng = sc[(sc.term == "lrx") & sc.scaled_by_national_rate]
    parts += ["A state reached in 2014 has its window in 2015–17 and one reached in 2020 in 2021–23, when national synthetic-opioid "
              "mortality was several times higher. Holding the arrival date absorbs a linear calendar trend but not the curve of the "
              "national rate, so the table repeats the main models with each state's window rate divided by the national rate over the "
              "same three calendar years, the deaths of the 49 states over their population. The last column turns each dispensing "
              "estimate into the ratio of mortality it implies between states at the 75th and 25th percentiles of dispensing, a ratio of "
              f"{rng.iqr_ratio.iloc[0]:.2f}. Scaled, the upper limits allow ratios of {rng.wave_ratio_hi.min():.2f} to "
              f"{rng.wave_ratio_hi.max():.2f}; the estimates move little.", "", md(st), ""]
    lo1 = pd.read_csv(out / "state_leave_one_out.csv")
    rows = []
    for b in (1, 5, 9):
        k = lo1[lo1.bound == b]
        full49, dc, others = k[k.left_out == "none"].iloc[0], k[k.left_out == "DC"].iloc[0], k[~k.left_out.isin(["none", "DC"])]
        nxt = others.sort_values("cooks_distance", ascending=False).iloc[0]
        rows.append({"Hidden counts entered at": b, "All 49": eci(full49.b, full49.lo, full49.hi), "Without DC": eci(dc.b, dc.lo, dc.hi),
                     "The other 48 left out in turn": f"{others.b.min():.2f} to {others.b.max():.2f}",
                     "Cook's distance, DC; next largest": f"{dc.cooks_distance:.2f}; {nxt.left_out} {nxt.cooks_distance:.2f}"})
    dcx = lo1[(lo1.bound == 5) & (lo1.left_out == "DC")].iloc[0]
    parts += ["The main state model, log dispensing adjusted for arrival over the three years after the arrival year, was refitted "
              "without each state in turn. The District of Columbia, a single city with the lowest dispensing of the 49, "
              f"{dcx.dispensing:.1f} per 100, and high mortality after arrival, weighs far more than any state, and without it the estimate "
              "is larger at every bound. The county test of A10 asks the sharper question, the rise within each state, where DC adds "
              "nothing.", "", md(pd.DataFrame(rows)), ""]

    tr = pd.read_csv(out / "alpert_rerun_results.csv")
    ttabs = triplicate_tables(tr)
    parts += ["## A9. Every model for the triplicate comparison", "",
              "The five triplicate states are California, Idaho, Illinois, New York and Texas. The panels compare their log overdose mortality "
              "with the other states', window by window, relative to 1995–98, with state and window fixed effects and errors clustered by "
              "state, as in Alpert and colleagues; deaths in 1985–98 use the corresponding ICD-9 codes, as in the original study. The "
              "cross-state models are linear regressions with robust standard errors. As in A8, each "
              f"model was fitted with suppressed county counts entered at one, five and nine, {sum(len(t) for _, t in ttabs)} rows in all.", ""]
    for bound, t in ttabs:
        parts += [f"Suppressed county counts entered at {bound}:", "", md(t), ""]
    tc = ac[ac.check.str.startswith("triplicate")]
    outcome = {"triplicate growth_since_launch": "Growth in overdose mortality, 1995–98 to 2017–19",
               "triplicate lt404": "Synthetic-opioid mortality, 2014–20, log"}
    reg = pd.DataFrame({"Outcome": tc.check.map(outcome), "Adjusted for": tc.model.map(lambda v: v[0].upper() + v[1:]), "Term": "Triplicate state",
                        "Estimate (95% CI)": [est(b, se) for b, se in zip(tc.b, tc.se)], "p": tc.p.map(pval)})
    lv = pd.read_csv(out / "triplicate_levels.csv").set_index("window").gap_rel_1990_94
    ours = (lv["1999-01"], lv[["2002-04", "2005-07", "2008-10"]].mean(), lv[["2011-13", "2014-16", "2017-19"]].mean())
    parts += ["Two further checks close this section, both with suppressed counts entered at five. The first concerns region. Three of "
              "the five triplicate states received fentanyl late: California and Idaho are in the West, and Texas, like the western "
              "states, was a market for black tar heroin. Every western state received fentanyl late, so late arrival and the western "
              "heroin market coincide. The table gives the triplicate difference adjusted for arrival date, for an indicator of the "
              "Census Bureau's western region, and for market type. The western region accounts for as much of the growth gap as the "
              "arrival date does, and the two cannot be told apart; the difference in synthetic-opioid mortality pooled over 2014–20 remains "
              "under each adjustment that includes arrival or market type, but that pooled rate mostly reflects years since arrival, "
              "and at the same years after arrival the difference is absent, as the third check of A8 shows.", "", md(reg), "",
              "The second sets our rebuild beside the published estimates, in their units. Alpert and colleagues estimated that, "
              "relative to 1991–95, states without triplicate programmes gained 1.125 overdose deaths per 100 000 in 1996–2000, 4.221 in "
              "2001–10 and 6.944 in 2011–17. In our rebuild from county counts, relative to 1990–94, the unweighted difference between "
              f"the two groups of states grew by {ours[0]:.2f} in 1999–2001, {ours[1]:.2f} on average in 2002–10 and {ours[2]:.2f} in "
              "2011–19.", ""]
    tt = pd.read_csv(out / "triplicate_trend.csv")
    label = {"trip_trend": "Triplicate trend before the launch, per year"}
    def term(t):
        if t in label:
            return label[t]
        y = int(t.split("_")[1])
        end = {1985: 1989, 1990: 1994, 1995: 1998}.get(y, y + 2)
        return f"Gap in {y}–{str(end)[2:]}"
    tw = pd.DataFrame({"Model": tt.model.map(lambda v: v[0].upper() + v[1:]), "Hidden counts entered at": tt.bound, "Term": tt.term.map(term),
                       "Estimate (95% CI)": [eci(b, lo, hi) for b, lo, hi in zip(tt.b, tt.lo, tt.hi)], "p": tt.p.map(pval)})
    tr1 = tt[(tt.term == "trip_trend")]
    parts += ["A third check concerns the years before the launch. The reference window, 1995–98, contains OxyContin's launch in 1996, "
              "and the gap in 1990–94 relative to it is positive. The table fits two further panels at each bound: one with a linear "
              "trend for the triplicate states identified from the three windows to 1995–98, so that each later gap is measured against "
              "that trend extended; and the event panel with 1990–94 as the reference. The triplicate states' trend before the launch "
              f"is flat, {tr1.b.min():.3f} to {tr1.b.max():.3f} a year, and the later gaps are as large or larger under either panel, "
              "though less precise against the extended trend.", "", md(tw), ""]

    # ---- A10-A13: the four tests added on 2026-11-19
    ca = pd.read_csv(out / "county_arrival_results.csv")
    keep = ~ca.term.str.match(r"^(lrx|lpills|zx)_x_20\d\d")
    ca = ca[keep]

    def cterm(t):
        names = {"lrx_x_after": "Log dispensing × share of window after arrival", "zx_x_after": "Distress × share after arrival",
                 "lpills_x_after": "Log pills × share after arrival", "lrx_x_years_since": "Log dispensing × years since arrival",
                 "lrx": "Log dispensing"}
        if t in names:
            return names[t]
        if t.startswith("lrx_x_e"):
            return {"lrx_x_e<=-3": "Log dispensing × three or more windows before arrival's",
                    "lrx_x_e=-2": "Log dispensing × two windows before arrival's",
                    "lrx_x_e=0": "Log dispensing × the window holding the arrival year",
                    "lrx_x_e=+1": "Log dispensing × the window after it"}[t]
        if t.startswith("lrx_x_since_"):
            return f"Log dispensing × {t[12:]} years since arrival"
        return t
    model = lambda v: (v[2:] if v[:2] in ("1 ", "2 ", "3 ", "4 ", "6 ") else v)
    ct = pd.DataFrame({"Model": ca.model.map(lambda v: model(v)[0].upper() + model(v)[1:]), "Hidden counts entered at": ca.bound.astype(str),
                       "Term": ca.term.map(cterm),
                       "Estimate (95% CI)": ["" if pd.isna(lo) else eci(b, lo, hi) for b, lo, hi in zip(ca.b, ca.lo, ca.hi)],
                       "Ratio, 75th to 25th percentile (95% CI)": ["" if pd.isna(r) else f"{r:.2f} ({a:.2f} to {z:.2f})" for r, a, z in zip(ca.ratio, ca.ratio_lo, ca.ratio_hi)],
                       "p": ca.p.map(pval)})
    ct.loc[ca.term.str.startswith("Wald").values, "Estimate (95% CI)"] = [f"χ² {b:.2f}, 2 df" for b in ca.b[ca.term.str.startswith("Wald")]]
    m0 = pd.read_csv(out / "county_arrival_results.csv").query("model == 'main' and term == 'lrx_x_after'").iloc[0]
    parts += ["## A10. The county test", "",
              "Within each state, did counties that prescribed more before fentanyl rise more once the state's fentanyl arrived? The panel "
              f"holds {int(m0.counties)} counties of the 48 contiguous states and DC, those with at least 1,000 non-Hispanic white residents in "
              "the 2000 census and a CDC dispensing rate for 2009–11, over five windows "
              "of overdose deaths, 2005–07 to 2017–19, from CDC WONDER. Counties without a dispensing rate, and counties with no death in any "
              "window, are left out. The model is Poisson pseudo-maximum likelihood with person-years as the offset and fixed effects for "
              "county, state by window and log dispensing by window, so each state's arrival, policies and everything else common to a "
              "state in a window are held, and the national prescribing gradient may differ in every window. The term of interest is log "
              "dispensing times the share of the window's three years in or after the state's arrival year, with errors clustered by state. "
              "The ratio compares a county at the 75th percentile of dispensing with one at the 25th, for a window wholly after arrival. "
              "Hidden counts of one to nine enter at one, five and nine, and in a censored model as known only to lie between them. The "
              "event-time model replaces the share with windows relative to the one holding the arrival year, the window before it being "
              "the reference, and the Wald test asks whether the windows before arrival are jointly zero. The synthetic-opioid models use "
              "the county-year file for 2018–25, whose deaths are both drug poisoning and T40.4, and the pooled 2014–20 file, within states. "
              "The national gradients by window are in the code archive.", "", md(ct), ""]

    nc = pd.read_csv(out / "negative_controls_results.csv")
    nc = nc[(nc["sample"] == "main 60") & (nc.variant == "code/02's missing years")]
    coding = {"anyill": "Any market by 2019", "current": "Markets present in 2019–21"}
    nt = pd.DataFrame({"Outcome": nc.outcome.map(lambda v: v[0].upper() + v[1:]), "Test": nc.model.map(lambda v: v[0].upper() + v[1:]),
                       "Supply coding": nc.coding.map(lambda v: coding.get(v, "")), "Countries": nc.n,
                       "Estimate (95% CI)": ["" if pd.isna(e) else (f"{e:.2f}" if pd.isna(lo) else eci(e, lo, hi)) for e, lo, hi in zip(nc.estimate, nc.lo, nc.hi)],
                       "p": nc.p.map(pval)})
    ei = pd.read_csv(out / "external_by_intent.csv")
    it = pd.DataFrame({"Outcome": ei.outcome, "Supply coding": ei.coding.map(lambda v: v[0].upper() + v[1:]), "Countries": ei.n,
                       "Interaction (95% CI)": [eci(b, lo, hi) for b, lo, hi in zip(ei.b, ei.lo, ei.hi)], "p": ei.p.map(pval)})
    en = pd.read_csv(out / "external_net.csv")
    nt2 = pd.DataFrame({"Control": en.control, "Narcotic interaction": en.how.map(lambda v: v[0].upper() + v[1:]), "Countries": en.n,
                        "Estimate (95% CI)": [eci(b, lo, hi) for b, lo, hi in zip(en.b, en.lo, en.hi)], "p": en.p.map(pval)})
    mk = pd.read_csv(out / "external_market_countries.csv")
    mt = pd.DataFrame({"Country": mk.country.map(lambda n: SHORT.get(n, n)), "Narcotic poisoning": mk.narcotics_chg.map(lambda v: f"{v:.2f}"),
                       "Self-harm": mk.selfharm_chg.map(lambda v: f"{v:.2f}"), "Assault": mk.assault_chg.map(lambda v: f"{v:.2f}"),
                       "Alcohol poisoning": mk.alcohol_chg.map(lambda v: "" if pd.isna(v) else f"{v:.2f}"), "Road traffic": mk.road_chg.map(lambda v: f"{v:.2f}")})
    us = pd.read_csv(out / "negative_controls_us_states.csv")
    us = us[us.outcome.isin(["alcohol-induced", "suicide"]) & (us.term == "lrx") & (us.bound == 5)] if "alcohol-induced" in set(us.outcome) else us[(us.term == "lrx") & (us.bound == 5) & ~us.outcome.isin(["overdose"])]
    ut = pd.DataFrame({"Outcome": us.outcome.map(lambda v: v[0].upper() + v[1:]), "Measure": us.measure, "Model": us.model.map(lambda v: v[0].upper() + v[1:]),
                       "Log dispensing (95% CI)": [eci(b, lo, hi) for b, lo, hi in zip(us.b, us.lo, us.hi)], "p": us.p.map(pval)})
    parts += ["## A11. Negative-control outcomes", "",
              "Fentanyl cannot cause alcohol poisoning (X45, X65 and Y15), road traffic deaths (V01–V89) or other external causes (V01–Y89 "
              "less drug poisoning). If the main country results reflected how countries record deaths, their wealth or what sets North "
              "America apart, these outcomes would show them too. Each outcome goes through the main country tests in the main analysis's "
              "60 countries, with the same prescribing, GDP and supply codes; alcohol poisoning has data for 56, so its heaviest quarter is 14. "
              "The first table gives every test, beside narcotic poisoning from the same code.", "", md(nt), "",
              "The external causes split by intent, drug poisoning excluded from each part: the interaction lies in self-harm and assault, "
              "and not in undetermined intent, where uncounted drug deaths could hide.", "", md(it), "",
              "The narcotic interaction with each control's change subtracted from narcotic change, and with it held as a covariate:", "",
              md(nt2), "", "Each market country's change from 2010–12 to 2019–21, as a log ratio, beside the median of the countries without "
              "a market by 2019:", "", md(mt), "",
              "Across US states, with hidden county counts at five, alcohol-induced deaths and suicide on log dispensing. The pooled files "
              "allow levels in 2010–20 only, and a change from 1999–2001 that spans the prescribing years. Prescribing predicts suicide "
              "levels across states, so a level gradient across states would be confounded; the paper's state estimates of synthetic-opioid "
              "mortality are adjusted for arrival and measured at the same years after it.", "", md(ut), ""]

    pr = pd.read_csv(out / "population_at_risk_results.csv")
    pr = pr[pr.term.isin(["lrx", "ladm", "ladm:post", "lrx:post", "rho", "lrx_pre"]) & ~pr.formula.str.contains("lrx_2011", na=False)]
    tlab = {"lrx": "Log dispensing", "ladm": "Log treatment admissions", "ladm:post": "Log admissions × after arrival",
            "lrx:post": "Log dispensing × after arrival", "rho": "Spearman's rho", "lrx_pre": "Log prescribing"}
    pt = pd.DataFrame({"Test": pr.test, "Outcome": pr.outcome.map(lambda v: v[0].upper() + v[1:]), "Units": pr["sample"].map(lambda v: v[0].upper() + v[1:]),
                       "Hidden counts at": pr.bound.map(lambda v: "" if pd.isna(v) else str(int(v))), "Model": pr.formula, "Number": pr.n,
                       "Term": pr.term.map(tlab), "Estimate (95% CI)": [f"{b:.2f}" if pd.isna(lo) else eci(b, lo, hi) for b, lo, hi in zip(pr.b, pr.lo, pr.hi)],
                       "p": pr.p.map(pval)})
    pm = pd.read_csv(out / "population_at_risk_market_countries.csv")
    pmt = pd.DataFrame({"Measure": pm.measure.map(lambda v: v[0].upper() + v[1:]), "Country": pm.country.map(lambda n: SHORT.get(n, n)), "Market": pm.status,
                        "Value": pm.value.map(lambda v: "" if pd.isna(v) else f"{v:.2f}"), "Year": pm.year.map(lambda v: "" if pd.isna(v) else str(int(v))),
                        "Rank": pm["rank"].map(lambda v: "" if pd.isna(v) else str(int(v))) + pm["of"].map(lambda v: f" of {int(v)}"),
                        "Note": pm.note.fillna("")})
    parts += ["## A12. The population at risk", "",
              "If prescribing built the population fentanyl reached, states that prescribed more should have had more people in opioid "
              "treatment before fentanyl, and that population should carry the mortality after arrival. US states: admissions to "
              "publicly funded treatment with heroin, non-prescription methadone or other opiates in any of the three substance fields of "
              "the Treatment Episode Data Set, 2009–11, per 100 000 of WONDER's population; four states lack a year in that window "
              "(Alabama, DC, Mississippi and West Virginia), which the complete-years rows leave out. S1 relates admissions to dispensing; "
              "S2 adds them to the paper's state model, the synthetic-opioid rate over the three years after the arrival year; S3 adds "
              "admissions times after arrival to the panel of states by window. Europe: EUDA's estimates of high-risk opioid use, opioid "
              "agonist treatment clients and opioid treatment entrants, the estimate closest to 2010 and before the country's market, on "
              "log prescribing and log GDP (C1). Admissions measure treatment capacity as well as need; across states they track "
              "buprenorphine and methadone distribution.", "", md(pt), "",
              "The European market countries' at-risk measures and ranks:", "", md(pmt), ""]

    cp = pd.read_csv(out / "canada_provinces_table.csv").dropna(subset=["presc_2007_2011"]).sort_values("presc_2007_2011", ascending=False)
    cpt = pd.DataFrame({"Province": cp.name, "Strong-opioid DDD per 1000 a day, fiscal 2007–08 to 2011–12": cp.presc_2007_2011.map(lambda v: f"{v:.2f}"),
                        "Arrival": cp.arrival.map(lambda v: "none by 2021" if pd.isna(v) or v > 2021 else str(int(v))),
                        "Poisoning per 100 000, 2010–12": cp.all_2010_2012.map(lambda v: f"{v:.2f}"), "2019–21": cp.all_2019_2021.map(lambda v: f"{v:.2f}"),
                        "Change, log ratio": cp.chg_all.map(lambda v: f"{v:.2f}"),
                        "PHAC opioid deaths per 100 000, 2016–21": cp.phac_opioid_2016_2021.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")})
    cr = pd.read_csv(out / "canada_provinces_results.csv")
    crt = pd.DataFrame({"Test": cr.test.str[2:].map(lambda v: v[0].upper() + v[1:]), "Variant": cr.variant.map(lambda v: v[0].upper() + v[1:]),
                        "Method": cr.method, "Term": cr.x, "Outcome": cr.y,
                        "Estimate (95% CI)": [eci(b, lo, hi) for b, lo, hi in zip(cr.estimate, cr.lo, cr.hi)], "p": cr.p.map(pval), "Provinces": cr.n})
    parts += ["## A13. Canada's provinces", "",
              "Prescribing is strong-opioid dispensing in defined daily doses per 1000 people a day, retail pharmacies, codeine and "
              "methadone excluded, the mean of fiscal 2007–08 to 2011–12 from table 1 of Jones and colleagues, transcribed twice and checked "
              "cell by cell against the published table. The outcome is poisoning of any intent, X40–X49, X60–X69 and Y10–Y19, from "
              "Statistics Canada table 13-10-0932-01, since no public table gives narcotic poisoning by province; nationally the narcotic "
              "codes are about a third of it. Arrival is the first year in which fentanyl made up at least 1% of the province's samples "
              "analysed by Health Canada's Drug Analysis Service and stayed there the next year; RAND's review places British Columbia's "
              "first detection in 2012, two years before the 1% share. The territories, which have no prescribing figures, are left out. "
              "PHAC's apparent opioid toxicity deaths, from 2016, are a check.", "", md(cpt), "", md(crt), ""]

    parts += ["## A14. STROBE checklist", "",
              "The checklist gives where the paper reports each item of the STROBE statement for cross-sectional studies, the version that "
              "fits a study of places observed over the same periods. Each location is a section and paragraph of the manuscript, or a "
              "section of this appendix, so it holds however the file is paginated. Items that concern individual participants are marked "
              "as not applicable.", "",
              "| Item | What the item asks for | Where the paper reports it |", "|---|---|---|",
              "| 1a | The design, in a common term, in the title or summary | Title; Summary, Methods |",
              "| 1b | An informative and balanced summary | Summary |",
              "| 2 | The scientific background and rationale | Introduction, paragraphs 1–3 |",
              "| 3 | Specific objectives, with any predictions | Introduction, paragraph 4 |",
              "| 4 | The key elements of the design, early in the paper | Methods, paragraph 1 |",
              "| 5 | The setting, locations and dates | Methods, Data, paragraphs 1–5; A1 |",
              "| 6a | Which countries, states, counties and provinces were included, and how they were chosen | Methods, Data, paragraphs 1, 2, 4 and 5; Results, paragraph 1; A10, A13 |",
              "| 7 | Outcomes, exposures, supply and adjustment variables, defined | Methods, Data, paragraphs 1–5; A3 |",
              "| 8 | Sources and measurement for each variable, and their comparability | Methods, Data, paragraphs 1–5; A1–A5, A10–A13 |",
              "| 9 | Efforts to address bias | Methods, Data, paragraph 3: supply coded without deaths, and the checks on that coding; paragraph 5: "
              "negative-control outcomes; A3–A5, A11 |",
              "| 10 | How the number of countries, states, counties and provinces arose | Methods, Data, paragraphs 4–5; Results, paragraphs 1, 6 and 7; A10, A13 |",
              "| 11 | How quantitative variables were handled | Methods, Data, paragraphs 1, 2 and 4; Methods, Statistical analysis, paragraphs 1–3 |",
              "| 12a | Statistical methods, including adjustment | Methods, Statistical analysis, paragraphs 1–3 |",
              "| 12b | Methods for subgroups and interactions | Methods, Statistical analysis, paragraphs 1–3 |",
              "| 12c | How missing data were handled | Methods, Data, paragraphs 1, 2 and 4: at least two periods of data, World Bank population "
              "where WHO's has gaps, and suppressed counts entered at one, five and nine |",
              "| 12d | Sampling strategy | Not applicable: every country, state and province with the data was included; counties with at least "
              "1,000 non-Hispanic white residents, a dispensing rate and at least one death (A10) |",
              "| 12e | Sensitivity analyses | Methods, Statistical analysis, paragraphs 1–3; A4–A13 |",
              "| 13a | The number of units at each stage | Results, paragraphs 1, 6 and 7; A7–A13 give the number in each model |",
              "| 13b | Why units were left out | Results, paragraph 1; A10, A13 |",
              "| 13c | A flow diagram | Not applicable |",
              "| 14a | The characteristics of the units | Results, paragraphs 1 and 5; table 1 |",
              "| 14b | Units with missing data for each variable | A7–A13 give the number in each model |",
              "| 15 | Outcome data | Results, paragraphs 2–8; figures 1–4 |",
              "| 16a | Unadjusted and adjusted estimates with 95% CIs | Results, paragraphs 2–8; table 2; A7–A13 |",
              "| 16b | Category boundaries for continuous variables | Results, paragraph 2: the heaviest-prescribing quarter, above 5214 S-DDD |",
              "| 16c | Relative risk turned into absolute risk | Not applicable |",
              "| 17 | Other analyses, such as interactions and sensitivity analyses | Results, paragraphs 2–8; A4–A13 |",
              "| 18 | Key results | Discussion, paragraph 1 |",
              "| 19 | Limitations | Discussion, paragraph 4 |",
              "| 20 | A cautious overall interpretation | Discussion, paragraphs 2–3 |",
              "| 21 | Generalisability | Discussion, paragraph 3 |",
              "| 22 | Funding and the funder's role | Summary, Funding; Methods, Role of the funding source |", ""]

    cited = sorted({m.group(1) for s in a.sources.dropna() for m in KEY.finditer(s)} | set(A4_SOURCES), key=lambda key: SOURCES[key][1])
    parts += ["## A15. Sources cited in this appendix", "",
              "The supply coding in A3 cites these sources by author or title and page, and the European check in A4 cites RAND's "
              "review by author and page and the two European Drug Reports by title.", ""]
    parts += [f"{i}. {SOURCES[key][1]}" for i, key in enumerate(cited, 1)] + [""]

    (pathlib.Path(paper) / "APPENDIX.md").write_text("\n".join(parts))
    print("appendix sections:", sum(p.startswith("## ") for p in parts),
          "; model rows:", sum(len(t) for _, t in tabs), sum(len(t) for _, t in stabs), sum(len(t) for _, t in ttabs), "; sources cited:", len(cited))


if __name__ == "__main__":
    main(*sys.argv[1:4])
