"""Pull the public data for the paper into the source folders, each source in its own dated folder with downloads.tsv and MANIFEST.sha256.

- who_mortality_db: the WHO Mortality Database raw files (ICD-10 parts 1-6 and the non-standard part, population, country codes, notes,
  availability and documentation), direct downloads with no registration. WHO's conditions: no commercial use, acknowledge WHO, and refer
  anyone who asks for the data to WHO rather than passing it on.
- oecd_pharma_consumption: OECD Health Statistics, pharmaceutical consumption in defined daily doses per 1,000 inhabitants per day, by ATC
  group (N02A is opioids), through OECD's public SDMX service.

- worldbank_population and worldbank_gdp_per_capita: World Development Indicators through the World Bank's API.
- dea_arcos_retail: DEA ARCOS Retail Drug Summary, Report 4 (state grams per 100,000 population by drug), 2006-2015, from DEA's
  public report page.
- incb_narcotic_drugs: INCB's narcotic drugs technical reports (free downloads from INCB's file server): the 2025 edition's Excel tables,
  including consumption in S-DDD for 2015-2024, and the 2003, 2006, 2009, 2012 and 2015 editions' statistical tables.

  python3 code/01_pull_public.py [date folder, default today] [dataset ...]
"""
import datetime as dt
import hashlib
import pathlib
import sys
import urllib.request

LAKE = pathlib.Path.home() / "data/ds/raw"
INCB = "https://www.incb.org/incb/uploads/documents/Narcotic-Drugs/"
ARCOS = "https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/"
WHO ="https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/"
SOURCES = {
    "who_mortality_db": {
        "mort_documentation.zip": WHO + "mort_documentation71f9e29d-7e3f-41e6-aafc-c4c1775c7aa3.zip?sfvrsn=40cce9be_42",
        "mort_availability.zip": WHO + "mort_availability.zip?sfvrsn=23261e11_33",
        "mort_country_codes.zip": WHO + "mort_country_codes.zip?sfvrsn=800faac2_5&ua=1",
        "mort_notes.zip": WHO + "mort_notes.zip?sfvrsn=2020c859_19&ua=1",
        "mort_pop.zip": WHO + "mort_pop.zip?sfvrsn=937039fc_26&ua=1",
        "morticd10_part1.zip": WHO + "morticd10_part1.zip?sfvrsn=e2a4f93a_19&ua=1",
        "morticd10_part2.zip": WHO + "morticd10_part2.zip?sfvrsn=6e55000b_6&ua=1",
        "morticd10_part3.zip": WHO + "morticd10_part3.zip?sfvrsn=9f1111a2_14&ua=1",
        "morticd10_part4.zip": WHO + "morticd10_part4.zip?sfvrsn=259c5c23_30&ua=1",
        "morticd10_part5.zip": WHO + "morticd10_part5.zip?sfvrsn=ad970d0b_34&ua=1",
        "morticd10_part6.zip": WHO + "morticd10_part6.zip?sfvrsn=ec801a61_4",
        "morticd10_add.zip": WHO + "morticd10_add.zip?sfvrsn=47111c9f_4",
    },
    "oecd_pharma_consumption": {
        "phmc_consum.csv": "https://sdmx.oecd.org/public/rest/data/OECD.ELS.HD,HEALTH_PHMC@DF_PHMC_CONSUM,/all?format=csvfilewithlabels",
    },
    # Total population by country and year, to fill the WHO population file's gaps (it has none for the US or Canada in recent years).
    "worldbank_population": {
        "SP.POP.TOTL.json": "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000&date=1990:2024",
    },
    # GDP per capita at purchasing-power parity, constant international dollars, to hold wealth constant (prescribing follows it).
    "worldbank_gdp_per_capita": {
        "NY.GDP.PCAP.PP.KD.json": "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.KD?format=json&per_page=20000&date=1990:2024",
    },
    # DEA ARCOS Retail Drug Summary, Report 4: each state's yearly distribution to retail registrants in grams per 100,000 population,
    # by drug (fentanyl base, oxycodone, hydrocodone and the rest), 2006-2015, to separate pills from patches (P4 inside the US).
    "dea_arcos_retail": {f"{y}_rpt4.pdf": f"{ARCOS}{y}/{y}{'-' if y == 2011 else '_'}rpt4.pdf" for y in range(2006, 2016)},
    # UNODC World Drug Report 2026 statistical annex, table 7.1: drug seizures by country, drug and year, 2015-2024 (a global supply check).
    # UNODC's site terms: personal, non-commercial use; no redistribution or derivative works.
    # The report's own notice (World Drug Report 2026, copyright page, read 2026-11-10) is wider: it may be reproduced in whole or in part
    # for educational or non-profit purposes with acknowledgement of the source; commercial use needs UNODC's written permission. So the
    # paper prints calculated seizure figures, cited, and the repository holds none of UNODC's files.
    "unodc_wdr_seizures": {
        "7.1_Drug_seizures_2015-2024.xlsx": "https://www.unodc.org/documents/data-and-analysis/WDR_2026/Annex/7.1_Drug_seizures_2015-2024.xlsx",
    },
    # INCB's narcotic drugs technical reports. Table XIV.1 in each edition since 2003 gives each country's average consumption in S-DDD
    # per million inhabitants per day over the edition's last three years; the 2025 edition adds an annual S-DDD file for 2015-2024.
    "incb_narcotic_drugs": {
        "2025_1_Data_tables.xlsx": INCB + "Technical-Publications/2025/1_Data_tables_2025.xlsb.xlsx",
        "2025_2_Consumption_in_S-DDD_2015-2024.xlsx": INCB + "Technical-Publications/2025/2_Consumption_in_S-DDD_2015-2024.xlsx",
        "2015_full_report.pdf": INCB + "Technical-Publications/2015/NAR-tech_pub_2015.pdf",
        "2012_part4_tables.pdf": INCB + "Technical-Publications/2012/NDR_2012_Part4_Tables_EFS.pdf",
        "2009_part4_tables.pdf": INCB + "Technical-Publications/2009/Part-4_Tables_of_reported_statistics.pdf",
        "2006_full_report.pdf": INCB + "Technical-Publications/2006/Narcotics_2006_ebook.pdf",
        "2003_full_report.pdf": INCB + "Technical-Publications/2003/NARCOTICS2003.pdf",
    },
}


def fetch(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "research-download/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
        server_date = r.headers.get("Last-Modified") or r.headers.get("Date") or ""
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    return server_date


def main(stamp, only=()):
    for dataset, files in SOURCES.items():
        if only and dataset not in only:
            continue
        folder = LAKE / dataset / stamp
        folder.mkdir(parents=True, exist_ok=True)
        log, sums = [], []
        for name, url in files.items():
            path = folder / name
            server_date = fetch(url, path)
            data = path.read_bytes()
            log.append(f"{name}\t{url}\t{len(data)}\t{server_date}\t{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
            print(f"{dataset}/{name}: {len(data):,} bytes")
        (folder / "downloads.tsv").write_text("file\turl\tbytes\tserver_date\tretrieved_utc\n" + "\n".join(log) + "\n")
        (folder / "MANIFEST.sha256").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else dt.date.today().strftime("%Y%m%d"), tuple(sys.argv[2:]))
