"""Canada's provinces: did prescribing before fentanyl, or fentanyl's arrival, set poisoning mortality inside one country?

Ten provinces are a replication and a description, not a powered test. Every test here was fixed before any province's mortality was
looked at.

Inputs, all public:
  - Deaths: Statistics Canada table 13-10-0932-01, deaths and crude rates by province of residence and grouped cause, 2000-2024. It is
    the only current StatCan table found with provinces and poisoning causes. It does not split poisoning by drug, so the provincial
    outcome is all poisoning by any intent: accidental X40-X49, intentional X60-X69 and undetermined intent Y10-Y19, which also holds
    alcohol, gases and other chemicals. Table 13-10-0156-01 has the narcotic codes X42, X62 and Y12, but for Canada only; it is used to
    say how much of the provincial measure is narcotic poisoning nationally, and to compare Canada's narcotic rate with the paper's.
  - Population: StatCan table 17-10-0005-01, 1 July estimates, used to pool the three territories and to check 13-10-0932's crude rates.
  - Arrival: Health Canada's Drug Analysis Service, YearlyCounts_20260105.csv, samples analysed by province and year from 2012. Arrival
    is the first year fentanyl was at least 1% of the province's analysed samples and stayed at least 1% the next year, the paper's US
    rule applied to years. A province never meeting it by 2025 has no arrival. Variants: samples other than cannabis as the denominator,
    since cannabis submissions fell after legalisation in 2018; and fentanyl plus the named fentanyl analogue columns. The column
    "OTHER ANALOGUES" is left out because the file does not say what it holds.
  - Prescribing: strong-opioid dispensing in defined daily doses per 1000 population per day, from IQVIA's CompuScript, as printed in
    Table 1 of Jones, Kurdyak and Fischer 2020, BMC Health Services Research 20:677, page 3, for 2007 to 2016. The table labels rows by
    fiscal year because it pairs dispensing with fiscal-year hospital data; its text says the first dispensing row is 2007 and the last
    2016, so rows are read as calendar years. Two transcriptions, one from the PDF page and one from the publisher's table image, are
    diffed here; the 2011 row is checked against Table 1 of Jones, Lee, Kaoser and Fischer 2021, IJERPH 18:7507, read from its XML.
  - PHAC's apparent opioid toxicity deaths, crude rate by province, 2016-2021, overall rows only, as a check on the StatCan measure.
    Rows by opioid type are never read past the filter.

Tests, fixed in advance, on the ten provinces (the territories have no prescribing data; they are described, pooled, not tested):
  1. Before fentanyl: Spearman correlation of mean prescribing 2007-2011 with mean poisoning mortality 2007-2011. Variant: 2005-2011
     mortality.
  2. After arrival: change = log of mean mortality 2019-21 over mean 2010-12. Spearman with prescribing and with arrival year (a
     province without arrival by 2021 is coded 2022, after the window). OLS with HC3 errors of change on log prescribing, then with
     arrival year added. Variants: accidental poisoning only; age-standardised rates; a baseline of the three years before each
     province's arrival (2010-12 where there is none by 2021); levels in 2019-21 and PHAC's 2016-21 rate as outcomes; the two DAS
     arrival variants.
  Spearman intervals use Fisher's z with the Bonett-Wright standard error; Spearman p-values are two-sided permutation p-values from
  200 000 random pairings. OLS intervals are normal, as in the paper, with t intervals on n minus parameters beside them.

Run from papers/4b-prior-prescribing:

  /opt/homebrew/bin/python3.13 -I code/32_canada_provinces.py <statcan_deaths_cause date folder> <statcan_population date folder> \
      <health_canada_das date folder> <PHAC ca_harms.csv> <canada_province_prescribing date folder> <out folder>

Writes, in the out folder: canada_provinces_table.csv (one row per province and the pooled territories), canada_provinces_arrival.csv
(DAS shares by province and year), canada_provinces_results.csv (every test), canada_provinces_national.csv (Canada's narcotic and all
poisoning by year) and canada_provinces_checks.csv.
"""
import pathlib
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PROV = {"NL": "Newfoundland and Labrador", "PE": "Prince Edward Island", "NS": "Nova Scotia", "NB": "New Brunswick", "QC": "Quebec",
        "ON": "Ontario", "MB": "Manitoba", "SK": "Saskatchewan", "AB": "Alberta", "BC": "British Columbia"}
TERR = {"YT": "Yukon", "NT": "Northwest Territories", "NU": "Nunavut"}
ALL_POISON = ["[X40-X49]", "[X60-X69]", "[Y10-Y19]"]
SEED = 20261119
NPERM = 200_000


def read_zip(path, usecols=None):
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if "MetaData" not in n][0]
    return pd.read_csv(z.open(name), usecols=usecols, low_memory=False)


def code_of(label):
    return label[label.rfind("["):] if "[" in label else label


# ------------------------------------------------------------------------------------------------------------------ inputs
def load_deaths(folder):
    d = read_zip(folder / "13100932-eng.zip", ["REF_DATE", "GEO", "Sex", "Cause of death (ICD-10)", "Characteristics", "VALUE", "STATUS"])
    d = d[d.Sex == "Both sexes"].copy()
    d["geo"] = d.GEO.str.replace(", place of residence", "", regex=False)
    d["code"] = d["Cause of death (ICD-10)"].map(code_of)
    d = d[d.code.isin(ALL_POISON + ["[A00-Y89]"])]
    d["char"] = d.Characteristics.map({"Number of deaths": "n", "Crude mortality rate per 100,000 population": "crude",
                                       "Age-standardized mortality rate per 100,000 population": "asr"})
    d = d.rename(columns={"REF_DATE": "year"})
    return d[["year", "geo", "code", "char", "VALUE", "STATUS"]]


def load_pop(folder):
    p = read_zip(folder / "17100005-eng.zip", ["REF_DATE", "GEO", "Gender", "Age group", "VALUE"])
    p = p[(p.Gender == "Total - gender") & (p["Age group"] == "All ages")]
    return p.rename(columns={"REF_DATE": "year", "GEO": "geo", "VALUE": "pop"})[["year", "geo", "pop"]]


def load_national(folder, pop):
    d = read_zip(folder / "13100156-eng.zip", ["REF_DATE", "GEO", "Age group", "Sex", "Cause of death (ICD-10)", "VALUE"])
    d = d[(d["Age group"] == "Total, all ages") & (d.Sex == "Both sexes")].copy()
    d["code"] = d["Cause of death (ICD-10)"].map(code_of)
    w = d.pivot_table(index="REF_DATE", columns="code", values="VALUE", aggfunc="sum")
    out = pd.DataFrame(index=w.index)
    out["x42"] = w["[X42]"]
    out["narcotic_x42_x62_y12"] = w["[X42]"] + w["[X62]"] + w["[Y12]"]
    out["drug_x40_x44_x60_x64_y10_y14"] = w[[f"[X4{i}]" for i in range(5)] + [f"[X6{i}]" for i in range(5)] + [f"[Y1{i}]" for i in range(5)]].sum(axis=1)
    three = [f"[{L}{i}]" for L, base in (("X", 4), ("X", 6), ("Y", 1)) for i in range(base * 10, base * 10 + 10)]
    out["all_poisoning"] = w[three].sum(axis=1)
    out["accidental_x40_x49"] = w["[X40-X49]"]
    out["accidental_check_x40_to_x49_summed"] = w[three[:10]].sum(axis=1)
    cp = pop[pop.geo == "Canada"].set_index("year")["pop"]
    out["pop"] = cp.reindex(out.index)
    for c in ["x42", "narcotic_x42_x62_y12", "drug_x40_x44_x60_x64_y10_y14", "all_poisoning"]:
        out[c + "_rate"] = out[c] / out["pop"] * 1e5
    out["narcotic_share_of_all_poisoning"] = out["narcotic_x42_x62_y12"] / out["all_poisoning"]
    out["drug_share_of_all_poisoning"] = out["drug_x40_x44_x60_x64_y10_y14"] / out["all_poisoning"]
    out.index.name = "year"
    return out


def arrival_from_das(das, num_cols, denom):
    rows, arr = [], {}
    for loc, g in das.groupby("LOCATION"):
        g = g.sort_values("YEAR").set_index("YEAR")
        num = g[num_cols].sum(axis=1)
        den = g["TOTAL"] - (g["CANNABIS"] if denom == "non-cannabis" else 0)
        share = num / den
        a = None
        yrs = list(share.index)
        for i, y in enumerate(yrs[:-1]):
            if share[y] >= 0.01 and share[yrs[i + 1]] >= 0.01:
                a = y
                break
        arr[loc] = a
        for y in yrs:
            rows.append({"das_location": loc, "year": y, "share": share[y]})
    return arr, pd.DataFrame(rows)


def load_prescribing(folder):
    a = pd.read_csv(folder / "jones2020_table1_transcription_a.csv").set_index("province")
    a.columns = [int(c[1:]) for c in a.columns]
    b = pd.read_csv(folder / "jones2020_table1_transcription_b.csv")
    b["year"] = b.table_row.str[:4].astype(int)
    b = b.pivot(index="province", columns="year", values="ddd_per_1000_per_day")
    diff = (a.sort_index() - b.sort_index()).abs()
    # Table 1 of Jones et al. 2021, from its XML: "2011-2018" column holds "change (2011 level-2018 level)".
    root = ET.parse(folder / "Jones2021_PMC8307682.1.xml").getroot()
    j21 = {}
    for tw in root.iter("table-wrap"):
        for tr in tw.iter("tr"):
            cells = [" ".join("".join(c.itertext()).split()) for c in tr if c.tag in ("td", "th")]
            if len(cells) > 2 and "(" in cells[1]:
                inner = cells[1][cells[1].index("(") + 1:cells[1].index(")")]
                nums = re.findall(r"\d+\.\d+", inner)
                if len(nums) == 2:
                    j21[{"MN": "MB", "PEI": "PE"}.get(cells[0], cells[0])] = (float(nums[0]), float(nums[1]))
    return a, diff, j21


# ------------------------------------------------------------------------------------------------------------------ statistics
def spearman(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = stats.spearmanr(x, y).statistic
    se = np.sqrt((1 + r ** 2 / 2) / (n - 3))
    z = np.arctanh(r)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    sims = np.empty(NPERM)
    ry_c = ry - ry.mean()
    rx_c = rx - rx.mean()
    denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
    for k in range(0, NPERM, 20_000):
        m = min(20_000, NPERM - k)
        idx = np.argsort(rng.random((m, n)), axis=1)
        sims[k:k + m] = (ry_c[idx] * rx_c).sum(axis=1) / denom
    p = (np.sum(np.abs(sims) >= abs(r) - 1e-12) + 1) / (NPERM + 1)
    return r, lo, hi, p, n


def ols(formula, d, term):
    m = smf.ols(formula, data=d).fit(cov_type="HC3")
    b, se = m.params[term], m.bse[term]
    tcrit = stats.t.ppf(0.975, m.df_resid)
    return {"estimate": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": m.pvalues[term], "lo_t": b - tcrit * se,
            "hi_t": b + tcrit * se, "se_hc3": se, "n": int(m.nobs), "r2": m.rsquared}


# ------------------------------------------------------------------------------------------------------------------ main
def main(deaths_dir, pop_dir, das_dir, harms_csv, presc_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    checks = []

    pop = load_pop(pop_dir)
    deaths = load_deaths(deaths_dir)
    pois = deaths[deaths.code.isin(ALL_POISON)]
    missing = pois[pois.VALUE.isna()]
    checks.append({"check": "13-10-0932 poisoning cells missing, both sexes, 2000-2024", "value": len(missing),
                   "detail": "; ".join(sorted(set(missing.geo + " " + missing.char))) [:400]})
    n = pois[pois.char == "n"].pivot_table(index=["geo", "year"], columns="code", values="VALUE", aggfunc="sum")
    n["all"] = n[ALL_POISON].sum(axis=1, min_count=3)
    n["acc"] = n["[X40-X49]"]
    asr = pois[pois.char == "asr"].pivot_table(index=["geo", "year"], columns="code", values="VALUE", aggfunc="sum")
    crude = pois[pois.char == "crude"].pivot_table(index=["geo", "year"], columns="code", values="VALUE", aggfunc="sum")
    pp = pop.set_index(["geo", "year"])["pop"]
    pp_terr = pop[pop.geo.isin(TERR.values())].groupby("year")["pop"].sum()

    # crude-rate check: StatCan's crude rate against deaths over 1 July population
    rec = []
    for (g, y), row in n.iterrows():
        if g in PROV.values() and (g, y) in pp.index and (g, y) in crude.index:
            rec.append(abs(row["[X40-X49]"] / pp[(g, y)] * 1e5 - crude.loc[(g, y), "[X40-X49]"]))
    checks.append({"check": "max abs difference, StatCan crude X40-X49 rate vs deaths / 17-10-0005 population x 1e5, provinces", "value": round(max(rec), 3), "detail": f"{len(rec)} province-years"})

    def series(geo, kind):
        if geo == "Territories":
            sub = n.loc[[t for t in TERR.values() if t in n.index.get_level_values(0)]]
            tot = sub.groupby(level="year")[["all", "acc"]].sum(min_count=1)
            return (tot["all" if kind == "all" else "acc"] / pp_terr.reindex(tot.index) * 1e5)
        if kind == "asr_all":
            return asr.loc[geo][ALL_POISON].sum(axis=1, min_count=3)
        col = "all" if kind == "all" else "acc"
        return n.loc[geo][col] / pp.loc[geo].reindex(n.loc[geo].index) * 1e5

    # arrival
    das = pd.read_csv(das_dir / "YearlyCounts_20260105.csv")
    analogues = ["CARFENTANIL", "FURANYL FENTANYL", "ACETYL FENTANYL", "CYCLOCPROPYL FENTANYL", "MeOAc FENTANYL", "para-FLUOROFENTANYL", "BROMOFENTANYL"]
    arr_main, sh_main = arrival_from_das(das, ["FENTANYL"], "total")
    arr_nc, sh_nc = arrival_from_das(das, ["FENTANYL"], "non-cannabis")
    arr_an, sh_an = arrival_from_das(das, ["FENTANYL"] + analogues, "total")
    sh = sh_main.rename(columns={"share": "fentanyl_share_of_all_samples"})
    sh["fentanyl_share_of_noncannabis_samples"] = sh_nc.share.values
    sh["fentanyl_and_named_analogues_share_of_all_samples"] = sh_an.share.values
    sh.to_csv(out / "canada_provinces_arrival.csv", index=False, float_format="%.5f")

    # prescribing
    pres, diff, j21 = load_prescribing(presc_dir)
    checks.append({"check": "transcriptions A and B of Jones 2020 Table 1, cells differing", "value": int((diff > 0).sum().sum()),
                   "detail": "100 cells, 10 provinces x 2007-2016"})
    mism = [f"{p}: {pres.loc[p, 2011]} vs {j21[p][0]}" for p in PROV if p in j21 and abs(pres.loc[p, 2011] - j21[p][0]) > 0.051]
    checks.append({"check": "Jones 2020 2011 row vs Jones 2021 Table 1 2011 levels, provinces differing by more than 0.05", "value": len(mism),
                   "detail": "; ".join(mism) if mism else f"{len([p for p in PROV if p in j21])} provinces matched"})

    harms = pd.read_csv(harms_csv, dtype=str)
    h = harms[(harms.Substance == "Opioids") & (harms.Source == "Deaths") & (harms.Specific_Measure == "Overall numbers")
              & (harms.Time_Period == "By year") & (harms.Unit == "Crude rate") & harms.Aggregator.isna() & harms.Disaggregator.isna()]
    h = h.assign(year=pd.to_numeric(h.Year_Quarter, errors="coerce"), rate=pd.to_numeric(h.Value, errors="coerce"))
    h = h[h.year.between(2016, 2021)]

    das_code = {**{k: k for k in PROV}, "Territories": "T"}
    rows = []
    for code, geo in list(PROV.items()) + [("TERR", "Territories")]:
        s_all, s_acc = series(geo, "all"), series(geo, "acc")
        s_asr = series(geo, "asr_all") if geo != "Territories" else pd.Series(dtype=float)
        dc = das_code.get(code, "T")
        a = arr_main.get(dc)
        r = {"province": code, "name": geo,
             "presc_2007_2011": pres.loc[code, 2007:2011].mean() if code in pres.index else np.nan,
             "presc_2007": pres.loc[code, 2007] if code in pres.index else np.nan,
             "presc_2011": pres.loc[code, 2011] if code in pres.index else np.nan,
             "arrival": a, "arrival_noncannabis": arr_nc.get(dc), "arrival_with_analogues": arr_an.get(dc),
             "das_samples_2012_2016": int(das[(das.LOCATION == dc) & das.YEAR.between(2012, 2016)].TOTAL.sum())}
        for lab, s in (("all", s_all), ("acc", s_acc), ("asr", s_asr)):
            r[f"{lab}_2005_2011"] = s.loc[2005:2011].mean() if len(s) else np.nan
            r[f"{lab}_2007_2011"] = s.loc[2007:2011].mean() if len(s) else np.nan
            r[f"{lab}_2010_2012"] = s.loc[2010:2012].mean() if len(s) else np.nan
            r[f"{lab}_2019_2021"] = s.loc[2019:2021].mean() if len(s) else np.nan
            if a is not None and a <= 2021:
                r[f"{lab}_pre_arrival"] = s.loc[a - 3:a - 1].mean() if len(s) else np.nan
            else:
                r[f"{lab}_pre_arrival"] = r[f"{lab}_2010_2012"]
        hr = h[h.Region == ("Territories" if code == "TERR" else geo)].rate
        r["phac_opioid_2016_2021"] = hr.mean() if hr.notna().sum() == 6 else np.nan
        r["phac_years"] = int(hr.notna().sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    t["arrival_c"] = t.arrival.fillna(2022).clip(upper=2022)
    t["arrival_nc_c"] = t.arrival_noncannabis.fillna(2022).clip(upper=2022)
    t["arrival_an_c"] = t.arrival_with_analogues.fillna(2022).clip(upper=2022)
    for lab in ("all", "acc", "asr"):
        t[f"chg_{lab}"] = np.log(t[f"{lab}_2019_2021"] / t[f"{lab}_2010_2012"])
        t[f"chg_pre_{lab}"] = np.log(t[f"{lab}_2019_2021"] / t[f"{lab}_pre_arrival"])
    t["log_presc"] = np.log(t.presc_2007_2011)
    t["presc_rank"] = t.presc_2007_2011.rank(ascending=False)
    t["chg_all_rank"] = t.loc[t.province != "TERR", "chg_all"].rank(ascending=False)
    t["level_2019_2021_rank"] = t.loc[t.province != "TERR", "all_2019_2021"].rank(ascending=False)
    t["phac_rank"] = t.loc[t.province != "TERR", "phac_opioid_2016_2021"].rank(ascending=False)
    t.to_csv(out / "canada_provinces_table.csv", index=False, float_format="%.4f")

    # tests on the ten provinces
    p10 = t[t.province != "TERR"].copy()
    res = []

    def add_s(test, x, y, label_x, label_y, variant):
        r, lo, hi, p, k = spearman(p10[x], p10[y], rng)
        res.append({"test": test, "variant": variant, "method": "Spearman", "x": label_x, "y": label_y, "estimate": r, "lo": lo, "hi": hi,
                    "p": p, "n": k})

    def add_o(test, formula, term, variant, label_y):
        o = ols(formula, p10, term)
        iqr = p10.log_presc.quantile(0.75) - p10.log_presc.quantile(0.25)
        ratio = np.exp(o["estimate"] * iqr) if term == "log_presc" else np.nan
        rng_ = p10.log_presc.max() - p10.log_presc.min()
        isp = term == "log_presc"
        # for arrival terms: the ratio for a province with fentanyl from 2014 against one without it by 2021 (coded 2022)
        span = rng_ if isp else 2014 - 2022
        res.append({"test": test, "variant": variant, "method": "OLS HC3: " + formula, "x": term, "y": label_y, **o,
                    "ratio_iqr": ratio, "ratio_iqr_lo": np.exp(o["lo"] * iqr) if isp else np.nan,
                    "ratio_iqr_hi": np.exp(o["hi"] * iqr) if isp else np.nan,
                    "ratio_range_or_2014_vs_none": np.exp(o["estimate"] * span), "ratio_range_lo": np.exp(min(o["lo"] * span, o["hi"] * span)),
                    "ratio_range_hi": np.exp(max(o["lo"] * span, o["hi"] * span))})

    add_s("1 before fentanyl", "presc_2007_2011", "all_2007_2011", "prescribing 2007-11", "all poisoning 2007-11", "main")
    add_s("1 before fentanyl", "presc_2007_2011", "all_2005_2011", "prescribing 2007-11", "all poisoning 2005-11", "mortality 2005-11")
    add_s("1 before fentanyl", "presc_2007_2011", "acc_2007_2011", "prescribing 2007-11", "accidental poisoning 2007-11", "accidental only")
    add_s("1 before fentanyl", "presc_2007_2011", "asr_2007_2011", "prescribing 2007-11", "age-standardised poisoning 2007-11", "age-standardised")
    add_o("1 before fentanyl", "np.log(all_2007_2011) ~ log_presc", "log_presc", "main, log level on log prescribing", "log all poisoning 2007-11")

    add_s("2 after arrival", "presc_2007_2011", "chg_all", "prescribing 2007-11", "change 2010-12 to 2019-21", "main")
    add_s("2 after arrival", "arrival_c", "chg_all", "arrival year (none by 2021 = 2022)", "change 2010-12 to 2019-21", "main")
    add_s("2 after arrival", "presc_2007_2011", "arrival_c", "prescribing 2007-11", "arrival year", "prescribing and arrival")
    add_o("2 after arrival", "chg_all ~ log_presc", "log_presc", "main, prescribing alone", "change 2010-12 to 2019-21")
    add_o("2 after arrival", "chg_all ~ log_presc + arrival_c", "log_presc", "main, with arrival", "change 2010-12 to 2019-21")
    add_o("2 after arrival", "chg_all ~ log_presc + arrival_c", "arrival_c", "main, with arrival", "change 2010-12 to 2019-21")
    add_o("2 after arrival", "chg_all ~ arrival_c", "arrival_c", "arrival alone", "change 2010-12 to 2019-21")
    for lab, name in (("acc", "accidental only"), ("asr", "age-standardised")):
        add_s("2 after arrival", "presc_2007_2011", f"chg_{lab}", "prescribing 2007-11", "change", name)
        add_s("2 after arrival", "arrival_c", f"chg_{lab}", "arrival year", "change", name)
        add_o("2 after arrival", f"chg_{lab} ~ log_presc + arrival_c", "log_presc", name + ", with arrival", "change")
        add_o("2 after arrival", f"chg_{lab} ~ log_presc + arrival_c", "arrival_c", name + ", with arrival", "change")
    add_s("2 after arrival", "presc_2007_2011", "chg_pre_all", "prescribing 2007-11", "change from 3 years before arrival", "pre-arrival baseline")
    add_o("2 after arrival", "chg_pre_all ~ log_presc + arrival_c", "log_presc", "pre-arrival baseline, with arrival", "change")
    add_o("2 after arrival", "chg_pre_all ~ log_presc + arrival_c", "arrival_c", "pre-arrival baseline, with arrival", "change")
    for ac, name in (("arrival_nc_c", "arrival, non-cannabis denominator"), ("arrival_an_c", "arrival, fentanyl with analogues")):
        add_s("2 after arrival", ac, "chg_all", name, "change 2010-12 to 2019-21", name)
        add_o("2 after arrival", f"chg_all ~ log_presc + {ac}", "log_presc", name + ", with arrival", "change")
        add_o("2 after arrival", f"chg_all ~ log_presc + {ac}", ac, name + ", with arrival", "change")
    add_s("2 after arrival", "presc_2007_2011", "all_2019_2021", "prescribing 2007-11", "level 2019-21", "level 2019-21")
    add_s("2 after arrival", "arrival_c", "all_2019_2021", "arrival year", "level 2019-21", "level 2019-21")
    add_o("2 after arrival", "np.log(all_2019_2021) ~ log_presc + arrival_c", "log_presc", "level 2019-21, with arrival", "log level")
    add_o("2 after arrival", "np.log(all_2019_2021) ~ log_presc + arrival_c", "arrival_c", "level 2019-21, with arrival", "log level")
    add_s("2 after arrival", "presc_2007_2011", "phac_opioid_2016_2021", "prescribing 2007-11", "PHAC opioid deaths 2016-21", "PHAC")
    add_s("2 after arrival", "arrival_c", "phac_opioid_2016_2021", "arrival year", "PHAC opioid deaths 2016-21", "PHAC")
    add_o("2 after arrival", "np.log(phac_opioid_2016_2021) ~ log_presc + arrival_c", "log_presc", "PHAC, with arrival", "log PHAC rate")
    add_o("2 after arrival", "np.log(phac_opioid_2016_2021) ~ log_presc + arrival_c", "arrival_c", "PHAC, with arrival", "log PHAC rate")
    add_s("check", "all_2019_2021", "phac_opioid_2016_2021", "StatCan all poisoning 2019-21", "PHAC opioid deaths 2016-21", "measure check")
    add_s("check", "presc_2007", "presc_2011", "prescribing 2007", "prescribing 2011", "stability of prescribing ranks")
    pd.DataFrame(res).to_csv(out / "canada_provinces_results.csv", index=False, float_format="%.4f")
    for q, lab in ((0, "min"), (0.25, "25th percentile"), (0.75, "75th percentile"), (1, "max")):
        checks.append({"check": f"prescribing 2007-11, ten provinces, {lab}", "value": round(p10.presc_2007_2011.quantile(q), 3),
                       "detail": "DDD per 1000 population per day; quantiles by pandas' linear rule"})

    nat = load_national(deaths_dir, pop)
    nat.to_csv(out / "canada_provinces_national.csv", float_format="%.4f")
    for lo_, hi_ in ((2007, 2011), (2010, 2012), (2019, 2021)):
        s = nat.loc[lo_:hi_]
        checks.append({"check": f"Canada narcotic X42+X62+Y12 rate per 100 000, mean {lo_}-{hi_}", "value": round(s["narcotic_x42_x62_y12_rate"].mean(), 3), "detail": "13-10-0156 with 17-10-0005"})
        checks.append({"check": f"Canada narcotic share of all poisoning, {lo_}-{hi_}", "value": round(s.narcotic_x42_x62_y12.sum() / s.all_poisoning.sum(), 3), "detail": "pooled deaths"})
        checks.append({"check": f"Canada drug (X40-X44, X60-X64, Y10-Y14) share of all poisoning, {lo_}-{hi_}", "value": round(s.drug_x40_x44_x60_x64_y10_y14.sum() / s.all_poisoning.sum(), 3), "detail": "pooled deaths"})
        checks.append({"check": f"Canada X42 share of narcotic X42+X62+Y12, {lo_}-{hi_}", "value": round(s.x42.sum() / s.narcotic_x42_x62_y12.sum(), 3), "detail": "pooled deaths"})
    cn = n.loc["Canada"]
    agree = (nat["all_poisoning"] - cn["all"].reindex(nat.index)).abs().max()
    checks.append({"check": "max abs difference in Canada all-poisoning counts, 13-10-0156 vs 13-10-0932", "value": agree, "detail": "2000-2024"})
    pd.DataFrame(checks).to_csv(out / "canada_provinces_checks.csv", index=False)
    print(t[["province", "presc_2007_2011", "presc_rank", "arrival", "arrival_noncannabis", "arrival_with_analogues", "all_2007_2011", "all_2010_2012",
             "all_2019_2021", "chg_all", "phac_opioid_2016_2021"]].round(2).to_string())
    print(pd.DataFrame(res)[["test", "variant", "method", "x", "estimate", "lo", "hi", "p", "n"]].round(3).to_string())
    print(pd.DataFrame(checks).to_string())


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    a = [pathlib.Path(x).expanduser() for x in sys.argv[1:]]
    main(*a)
