"""Negative-control outcomes: does the prescribing-by-supply pattern also appear in deaths fentanyl cannot cause?

The country results are a contrast among the heaviest-prescribing quarter, with and without an illicit fentanyl market, in change from
2010-12 to 2019-21 and in 2019-21 levels, and an interaction of log prescribing with illicit supply. If the same patterns turn up in
deaths fentanyl cannot cause, they reflect how countries record deaths, their wealth or what sets North America apart, not fentanyl.
The three control outcomes were chosen before any result was seen:

- alcohol poisoning, X45, X65 and Y15: the nearest cousin of narcotic poisoning in how coroners code it;
- road traffic deaths, V01-V89: a general injury and recording control;
- all external causes other than drug poisoning: V01-Y89 less X40-X44, X60-X64, X85 and Y10-Y14.

Part 1, countries. The WHO extraction is code/02's, copied here and given the extra cause groups, with code/02 left as it is: underlying
cause, both sexes, all ages, detailed lists 103 and 104 only, WHO population with the World Bank's filling gaps. A country-year with no
rows for a cause group is missing, not zero, as in code/02; a variant fills those with zero. Each outcome then goes through code/06's own
frame function, loaded from code/06 without editing it as code/29 does, with the control rate in place of narcotic poisoning, so
prescribing, GDP, supply codes and the change measure are code/06's. Tests, on the main analysis's 60 countries and, beside them, on
every country that passes code/06's frame for that outcome:
  - change ~ log prescribing x supply + log GDP, OLS with HC3 errors, under code/06's coding (any illicit market by 2019) and under
    markets present in 2019-21 (code/25's `current`: USA, Canada, Estonia, Latvia, Lithuania);
  - Spearman correlation of prescribing with change;
  - among the heaviest-prescribing quarter, change and 2019-21 level with and without a market, two-sided exact Mann-Whitney.
Narcotic poisoning runs through the same path, and its results are checked against code/06's and code/25's outputs.

Part 2, US states. CDC WONDER county files for alcohol-induced deaths and suicide, 1999-2001 and 2010-20 pooled, summed to states with
suppressed counts at 1, 5 or 9 as code/07 does; overdose and synthetic-opioid deaths from the same folder run alongside for comparison.
Prescribing, arrival timing, distress and market type are taken from code/07's state frame (out/us_states_frame.csv). The pooled files
allow levels only: log rate 2010-20 on log prescribing, then with arrival timing, then with distress and market type, HC3 errors. A
change from 1999-2001 to 2010-20 is a variant; it spans the prescribing years and fentanyl's first years, so it is not the paper's
fentanyl-era change. Death counts are never written; only rates and estimates.


  python3 -I code/30_negative_controls.py <who_mortality_db date folder> <worldbank_population date folder> <incb csv> <arrival csv> \
      <worldbank_gdp_per_capita date folder> <wonder_pooled date folder> <out folder>

Writes, in the out folder: negative_controls_who_country_year.csv (the extraction), negative_controls_countries.csv (each outcome's
frame, rates only), negative_controls_results.csv (country tests), negative_controls_heavy.csv (the heaviest quarter's rates and ranks),
negative_controls_dropone.csv (the interaction without each market country in turn), negative_controls_net.csv (the interaction with
narcotic change less each control's change as the outcome), negative_controls_us_states.csv (state tests) and negative_controls_checks.csv (reproduction checks).
"""
import importlib.util
import io
import json
import pathlib
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr

HERE = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------- code/02's extraction, copied, with the control groups added
ALIAS = {"united states of america": "united states", "korea, rep.": "korea", "republic of korea": "korea", "iran, islamic rep.": "iran",
         "iran (islamic republic of)": "iran", "venezuela, rb": "venezuela", "venezuela (bolivarian republic of)": "venezuela", "egypt, arab rep.": "egypt",
         "slovak republic": "slovakia", "kyrgyz republic": "kyrgyzstan", "turkey": "turkiye", "czech republic": "czechia", "republic of moldova": "moldova",
         "hong kong sar, china": "hong kong", "china, hong kong sar": "hong kong", "bahamas, the": "bahamas", "gambia, the": "gambia",
         "netherlands (kingdom of the)": "netherlands", "russian federation": "russia", "the former yugoslav republic of macedonia": "north macedonia",
         "saint vincent and the grenadines": "st vincent", "st. vincent and the grenadines": "st vincent", "saint lucia": "st lucia", "st. lucia": "st lucia",
         "saint kitts and nevis": "st kitts", "st. kitts and nevis": "st kitts", "micronesia, fed. sts.": "micronesia", "micronesia (federated states of)": "micronesia"}

DRUG = ("X40", "X41", "X42", "X43", "X44", "X60", "X61", "X62", "X63", "X64", "X85", "Y10", "Y11", "Y12", "Y13", "Y14")
ROAD = tuple(f"V{i:02d}" for i in range(1, 90))
GROUPS = {
    "drug_poisoning": DRUG,
    "narcotics": ("X42", "X62", "Y12"),
    "alcohol_poisoning": ("X45", "X65", "Y15"),
    "road_traffic": ROAD,
    "external_nondrug": None,  # V01-Y89 less DRUG, chosen by range below
}
DETAILED = {"103", "104"}  # three- and four-character ICD-10 lists
OUTCOMES = {"narcotics": "narcotic poisoning (the paper's outcome)", "alcohol_poisoning": "alcohol poisoning",
            "road_traffic": "road traffic deaths", "external_nondrug": "external causes other than drug poisoning"}


def key(s):
    s = re.sub(r"\s+", " ", str(s).lower().replace("ü", "u")).strip()
    return ALIAS.get(s, s)


def worldbank(folder):
    d = json.loads((pathlib.Path(folder) / "SP.POP.TOTL.json").read_text())[1]
    w = pd.DataFrame([(key(x["country"]["value"]), int(x["date"]), x["value"]) for x in d if x["value"] is not None], columns=["key", "Year", "wb_population"])
    return w.drop_duplicates(["key", "Year"]).set_index(["key", "Year"]).wb_population


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        return pd.read_csv(io.BytesIO(z.read(name)), dtype=str, low_memory=False)


def in_group(cause3, g):
    if GROUPS[g] is not None:
        return cause3.isin(GROUPS[g])
    external = cause3.str[0].isin(list("VWXY")) & (cause3 >= "V01") & (cause3 <= "Y89")
    return external & ~cause3.isin(DRUG)


def extract(folder, out, wb_folder):
    """code/02's main(), with GROUPS widened. Also writes <group>_zf_per_100k, the same rate with missing country-years set to zero."""
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
        for g in GROUPS:
            k = d[in_group(d.cause3, g)]
            parts.append(k.groupby(["Country", "Year", "List"], as_index=False).deaths.sum().assign(group=g))
        print(f.name, len(d), "rows")
    long = pd.concat(parts, ignore_index=True)
    r = long.pivot_table(index=["Country", "Year", "List"], columns="group", values="deaths", aggfunc="sum").reset_index()
    r.columns.name = None
    r = r[r.List.isin(DETAILED)].drop(columns="List").groupby(["Country", "Year"]).sum(min_count=1).reset_index()
    r = r.join(pop, on=["Country", "Year"])
    r.insert(1, "country_name", r.Country.map(names))
    r["population_source"] = r.population.notna().map({True: "WHO", False: ""})
    r["Year"] = pd.to_numeric(r.Year)
    wb = worldbank(wb_folder)
    fill = pd.Series([wb.get((key(n), y)) for n, y in zip(r.country_name, r.Year)], index=r.index, dtype=float)
    use = r.population.isna() & fill.notna()
    r.loc[use, "population"] = fill[use]
    r.loc[use, "population_source"] = "World Bank"
    for g in GROUPS:
        r[f"{g}_per_100k"] = r[g] / r.population * 1e5
        r[f"{g}_zf_per_100k"] = r[g].fillna(0) / r.population * 1e5
    r.to_csv(out, index=False)
    print(f"{r.Country.nunique()} countries, years {r.Year.min()}-{r.Year.max()}")
    return r


# ---------------------------------------------------------------- code/06's frame and tests
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CURRENT = {"United States of America", "Canada", "Estonia", "Latvia", "Lithuania"}  # code/25's markets present in 2019-21


def exact_mw(a, b):
    return mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue


def tests(m, outcome, sample, variant, iqr):
    m = m.copy()
    m["current"] = m.country.isin(CURRENT).astype(int)
    base = dict(outcome=outcome, sample=sample, variant=variant)
    rows = []
    for coding in ("anyill", "current"):
        k = m.dropna(subset=["narc_change", "lrx_pre", "lgdp"])
        r = smf.ols(f"narc_change ~ lrx_pre * {coding} + lgdp", data=k).fit(cov_type="HC3")
        t = f"lrx_pre:{coding}"
        lo, hi = r.conf_int().loc[t]
        rows.append(dict(base, coding=coding, model="interaction, adjusted for log GDP", n=int(r.nobs), n_with=int(k[coding].sum()),
                         estimate=r.params[t], lo=lo, hi=hi, p=r.pvalues[t], lrx_iqr_main60=iqr, ratio_iqr=np.exp(r.params[t] * iqr),
                         ratio_iqr_lo=np.exp(lo * iqr), ratio_iqr_hi=np.exp(hi * iqr),
                         slope_without=r.params["lrx_pre"], p_slope_without=r.pvalues["lrx_pre"]))
    s = spearmanr(m.rx_pre, m.narc_change)
    rows.append(dict(base, coding="", model="Spearman, prescribing with change", n=len(m), estimate=s[0], p=s[1]))
    h = m[m.rx_pre >= m.rx_pre.quantile(0.75)]
    for coding in ("anyill", "current"):
        w, wo = h[h[coding] == 1], h[h[coding] == 0]
        for oc in ("narc_1921", "narc_change"):
            rows.append(dict(base, coding=coding, model=f"Mann-Whitney, heaviest quarter, {'2019-21 level' if oc == 'narc_1921' else 'change'}",
                             n=len(h), n_with=len(w), median_with=w[oc].median(), median_without=wo[oc].median(),
                             max_without=wo[oc].max(), min_with=w[oc].min(), p=exact_mw(w[oc], wo[oc]),
                             ranks_with=" ".join(str(int(x)) for x in sorted(h[oc].rank(ascending=False)[h[coding] == 1]))))
    return rows


def countries(who_csv, incb, arrival, gdp, out):
    t6 = load("t6", "06_tests.py")
    w = pd.read_csv(who_csv)
    tmp = out / "negative_controls_tmp_who.csv"
    rows, frames, base, iqr = [], [], None, None
    for variant, suffix in (("code/02's missing years", "_per_100k"), ("missing years as zero", "_zf_per_100k")):
        for g, label in OUTCOMES.items():
            w.assign(narcotics_per_100k=w[g + suffix]).to_csv(tmp, index=False)
            m = t6.frame(incb, tmp, arrival, gdp)
            tmp.unlink()
            if base is None:  # narcotic poisoning, code/02's way: the main analysis's 60
                base = set(m.key)
                iqr = m.lrx_pre.quantile(0.75) - m.lrx_pre.quantile(0.25)
                print(f"middle half of prescribing in the main 60: {np.expm1(m.lrx_pre.quantile(0.25)):,.0f} to "
                      f"{np.expm1(m.lrx_pre.quantile(0.75)):,.0f} S-DDD, a gap of {iqr:.2f} in log(1 + S-DDD)")
            rows += tests(m, label, f"all {len(m)} passing", variant, iqr)
            main60 = m[m.key.isin(base)]
            rows += tests(main60, label, "main 60", variant, iqr)
            if variant == "code/02's missing years":
                frames.append(m[["key", "country", "status", "anyill", "rx_pre", "lrx_pre", "lgdp", "narc_1012", "narc_1921", "narc_change"]]
                              .rename(columns={"narc_1012": "rate_1012", "narc_1921": "rate_1921", "narc_change": "change"})
                              .assign(outcome=label, in_main_60=m.key.isin(base)))
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True), base


def drop_one(f):
    """The any-illicit interaction on the main 60, refitted without each market country in turn and without the USA and Canada together."""
    rows = []
    for label in OUTCOMES.values():
        m = f[(f.outcome == label) & f.in_main_60].rename(columns={"change": "narc_change"})
        sets = [(c, {c}) for c in sorted(m.loc[m.anyill == 1, "country"])] + [("USA and Canada", {"United States of America", "Canada"})]
        for name, drop in [("none", set())] + sets:
            k = m[~m.country.isin(drop)].dropna(subset=["narc_change", "lrx_pre", "lgdp"])
            r = smf.ols("narc_change ~ lrx_pre * anyill + lgdp", data=k).fit(cov_type="HC3")
            lo, hi = r.conf_int().loc["lrx_pre:anyill"]
            rows.append(dict(outcome=label, dropped=name, n=int(r.nobs), estimate=r.params["lrx_pre:anyill"], lo=lo, hi=hi, p=r.pvalues["lrx_pre:anyill"]))
    return pd.DataFrame(rows)


def net_of_control(f):
    """The interaction on the main 60 with narcotic change less each control's change as the outcome: how much of the narcotic
    interaction is left once a country's general trend in the control is taken out."""
    rows = []
    m = f[f.in_main_60]
    nar = m[m.outcome == OUTCOMES["narcotics"]].set_index("key")
    for g, label in OUTCOMES.items():
        if g == "narcotics":
            continue
        c = m[m.outcome == label].set_index("key").change
        k = nar.assign(narc_change=nar.change - c, current=nar.country.isin(CURRENT).astype(int)).dropna(subset=["narc_change", "lgdp"])
        for coding in ("anyill", "current"):
            r = smf.ols(f"narc_change ~ lrx_pre * {coding} + lgdp", data=k).fit(cov_type="HC3")
            t = f"lrx_pre:{coding}"
            lo, hi = r.conf_int().loc[t]
            rows.append(dict(outcome=f"narcotic change less {label} change", coding=coding, n=int(r.nobs), estimate=r.params[t], lo=lo, hi=hi,
                             p=r.pvalues[t]))
    return pd.DataFrame(rows)


def heavy_table(f, base):
    m = f[f.in_main_60]
    nar = m[m.outcome == OUTCOMES["narcotics"]].set_index("key")
    ranks = nar.rx_pre.rank(ascending=False).astype(int)
    keep = nar[nar.rx_pre >= nar.rx_pre.quantile(0.75)].index
    t = nar.loc[keep, ["country", "status", "anyill", "rx_pre"]].assign(rx_rank=ranks.loc[keep],
                                                                       current=lambda d: d.country.isin(CURRENT).astype(int))
    for g, label in OUTCOMES.items():
        k = m[m.outcome == label].set_index("key")
        k = k[k.index.isin(keep)]
        t[f"{g}_1012"] = k.rate_1012
        t[f"{g}_1921"] = k.rate_1921
        t[f"{g}_change"] = k.change
        t[f"{g}_1921_rank"] = k.rate_1921.rank(ascending=False)
        t[f"{g}_change_rank"] = k.change.rank(ascending=False)
    return t.sort_values("rx_rank").reset_index()


# ---------------------------------------------------------------- US states
WONDER = {"overdose": ("ucd_od_1999_2001", "ucd_od_2010_2020"), "alcohol-induced": ("ucd_alcohol_1999_2001", "ucd_alcohol_2010_2020"),
          "suicide": ("ucd_suicide_1999_2001", "ucd_suicide_2010_2020"), "synthetic opioid (T40.4) 2014-20": (None, "mcd_t404_2014_2020")}


def read_wonder(path, bound):  # code/07's
    d = pd.read_csv(path, sep="\t", dtype={"fips": str})
    d["fips"] = d.fips.str.zfill(5)
    d["deaths"] = pd.to_numeric(d.deaths.replace("Suppressed", bound), errors="coerce")
    d["population"] = pd.to_numeric(d.population, errors="coerce")
    d["st"] = d.fips.str[:2].astype(int)
    return d


def state_rate(path, bound):
    d = read_wonder(path, bound).groupby("st")[["deaths", "population"]].sum()
    return d.deaths / d.population * 1e5


def states(wonder, out):
    wonder = pathlib.Path(wonder)
    s0 = pd.read_csv(out / "us_states_frame.csv")[["st", "state", "rx", "lrx", "za", "zx", "white", "t404_2014_20"]].set_index("st")
    res, check = [], []
    for bound in (1, 5, 9):
        s = s0.copy()
        for name, (early, late) in WONDER.items():
            s[f"late_{name}"] = state_rate(wonder / f"{late}_county.tsv", bound)
            if early:
                s[f"early_{name}"] = state_rate(wonder / f"{early}_county.tsv", bound)
        if bound == 5:
            check.append(dict(check="US: synthetic-opioid rate 2014-20 from this path against code/07's frame, max absolute difference",
                              value=float((s["late_synthetic opioid (T40.4) 2014-20"] - s.t404_2014_20).abs().max())))
        for name, (early, late) in WONDER.items():
            outs = [(f"log rate {'2010-20' if early else '2014-20'}", np.log(s[f"late_{name}"]))]
            if early:
                outs.append(("change, 1999-2001 to 2010-20", np.log(s[f"late_{name}"] / s[f"early_{name}"])))
            for olab, y in outs:
                k = s.assign(y=y).dropna(subset=["y", "lrx", "za"])
                rho = spearmanr(k.rx, k.y)
                res.append(dict(outcome=name, measure=olab, bound=bound, model="Spearman with prescribing", n=len(k), term="rho", b=rho[0], p=rho[1]))
                for formula, lab in (("y ~ lrx", "log prescribing"), ("y ~ lrx + za", "+ arrival timing"),
                                     ("y ~ lrx + za + zx + white", "+ arrival, distress, market type")):
                    r = smf.ols(formula, data=k).fit(cov_type="HC3")
                    for term in ("lrx", "za"):
                        if term in r.params:
                            lo, hi = r.conf_int().loc[term]
                            res.append(dict(outcome=name, measure=olab, bound=bound, model=lab, n=int(r.nobs), term=term, b=r.params[term],
                                            lo=lo, hi=hi, p=r.pvalues[term], r2=r.rsquared))
    return pd.DataFrame(res), check, len(s0)


# ---------------------------------------------------------------- main
def main(who, wb, incb, arrival, gdp, wonder, out):
    out = pathlib.Path(out)
    who_csv = out / "negative_controls_who_country_year.csv"
    extract(who, who_csv, wb)
    res, frames, base = countries(who_csv, incb, arrival, gdp, out)
    res.to_csv(out / "negative_controls_results.csv", index=False)
    frames.to_csv(out / "negative_controls_countries.csv", index=False)
    heavy = heavy_table(frames, base)
    heavy.to_csv(out / "negative_controls_heavy.csv", index=False)
    drops = drop_one(frames)
    drops.to_csv(out / "negative_controls_dropone.csv", index=False)
    net = net_of_control(frames)
    net.to_csv(out / "negative_controls_net.csv", index=False)

    checks = []
    lr = frames[(frames.outcome == OUTCOMES["narcotics"]) & frames.in_main_60].lrx_pre
    for q in (0.25, 0.75):
        checks.append(dict(check=f"prescribing at the {int(q * 100)}th percentile of the main 60, S-DDD", value=float(np.expm1(lr.quantile(q)))))
    w = pd.read_csv(who_csv)
    ref02 = out / "who_drug_deaths_country_year.csv"
    if ref02.exists():
        r2 = pd.read_csv(ref02)
        j = r2.merge(w, on=["Country", "Year"], suffixes=("_02", "_30"))
        for c in ("narcotics", "drug_poisoning", "population"):
            checks.append(dict(check=f"WHO extraction: {c} against code/02's file, rows matched {len(j)} of {len(r2)}, max absolute difference",
                               value=float((j[c + "_02"] - j[c + "_30"]).abs().max())))
    ref06 = out / "tests_results.csv"
    main_int = res[(res.outcome == OUTCOMES["narcotics"]) & (res["sample"] == "main 60") & (res.variant == "code/02's missing years")]
    if ref06.exists():
        t = pd.read_csv(ref06)
        t = t[(t.model == "P2 any illicit | narc_change | all") & (t.term == "lrx_pre:anyill")].iloc[0]
        mine = main_int[(main_int.coding == "anyill") & (main_int.model.str.startswith("interaction"))].iloc[0]
        checks.append(dict(check="interaction, narcotic poisoning, any illicit: this path minus code/06's estimate", value=mine.estimate - t.b))
        checks.append(dict(check="interaction p: this path minus code/06's", value=mine.p - t.p))
    ref25 = out / "outcome_window_results.csv"
    if ref25.exists():
        t = pd.read_csv(ref25)
        t = t[t.source == "WHO"]
        for coding in ("anyill", "current"):
            a = t[(t.coding == coding) & (t.outcome == "narc_change") & (t.model == "interaction, adjusted for log GDP")].iloc[0]
            b = main_int[(main_int.coding == coding) & main_int.model.str.startswith("interaction")].iloc[0]
            checks.append(dict(check=f"interaction p, {coding}: this path minus code/25's", value=b.p - a.p))
            for oc, lab in (("narc_1921", "2019-21 level"), ("narc_change", "change")):
                a = t[(t.coding == coding) & (t.outcome == oc) & (t.model == "Mann-Whitney, heaviest quarter")].iloc[0]
                b = main_int[(main_int.coding == coding) & (main_int.model == f"Mann-Whitney, heaviest quarter, {lab}")].iloc[0]
                checks.append(dict(check=f"Mann-Whitney p, heaviest quarter, {lab}, {coding}: this path minus code/25's", value=b.p - a.p))
    missing = []
    for g in OUTCOMES:
        k = w[w.country_name.notna() & w.Year.between(2010, 2021)]
        missing.append(dict(check=f"country-years 2010-21 with no {g} rows (missing, not zero, in code/02's logic), all countries",
                            value=int(k[g].isna().sum())))
    checks += missing

    us, uscheck, n_states = states(wonder, out)
    us.to_csv(out / "negative_controls_us_states.csv", index=False)
    checks += uscheck
    pd.DataFrame(checks).to_csv(out / "negative_controls_checks.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_columns", 40):
        print(pd.DataFrame(checks).to_string(index=False))
        cols = ["outcome", "sample", "variant", "coding", "model", "n", "n_with", "estimate", "lo", "hi", "p", "median_with", "median_without", "ranks_with"]
        print(res[cols].round(3).to_string(index=False))
        print(heavy.round(2).to_string(index=False))
        print(drops.round(3).to_string(index=False))
        print(net.round(3).to_string(index=False))
        print(us[us.bound == 5].round(3).to_string(index=False))
    print(f"{n_states} states")


if __name__ == "__main__":
    main(*sys.argv[1:8])
