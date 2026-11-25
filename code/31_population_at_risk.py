"""Population at risk: did prescribing before fentanyl build a larger population in opioid treatment or high-risk opioid use, and does
that population, more than prescribing itself, carry mortality once fentanyl arrived?

The paper concludes that prescribing may have built the population at risk that fentanyl reached. This tests the mechanism in two
steps: (1) prescribing predicts the population at risk; (2) the population at risk predicts mortality after fentanyl's arrival, where
prescribing does not.

Part 1, US states, the 48 contiguous states and DC, as in code/07.
- Population at risk: SAMHSA TEDS-A admissions to publicly funded treatment, read in chunks from the 2006-2023 zip (one 4.9 GB CSV),
  columns STFIPS, ADMYR, SUB1-SUB3 and METHUSE only. Three measures per 100 000 population: any opioid (code 5, 6 or 7 among SUB1-SUB3),
  primary heroin (SUB1 = 5), and primary other opiates and synthetics (SUB1 = 7), the iatrogenic marker. Main window 2009-11, the
  paper's prescribing window; 2006-08 as a check. Population: WONDER's state-year population (the file code/24 reads), summed over the
  years the state is present in TEDS. A state-year SAMHSA dropped has no records; it is left out of both numerator and denominator,
  never filled, and the state is flagged as incomplete. The codebook's list of dropped state-years is checked against the records.
- Prescribing, arrival, distress and market type: code/07's state_frame(), loaded unedited. Post-arrival outcome: code/24's
  synthetic-opioid death rate over the three calendar years after each state's arrival year, copied from code/24 with its frame
  (out/alpert_rerun_frame.csv) and checked against out/fixed_span.csv. Hidden WONDER counts at 1 and at 9 (S2) or 1, 5 and 9 (S3).
- S1: log admissions 2009-11 on log prescribing 2009-11, OLS with HC3 errors, and the Spearman correlation; for each measure.
- S2: code/24's main state model, log rate in years 1-3 after arrival on log prescribing and arrival timing, reproduced; then with log
  admissions added, and with admissions in place of prescribing.
- S3: code/07's state-by-window panel, overdose windows 2008-10 to 2017-19, state and window effects, errors clustered by state, with
  log admissions x after arrival beside code/07's prescribing terms.
- Checks: without states with incomplete TEDS years, and the paper's model without each of them in turn; 2006-08 in place of 2009-11;
  years 2-4 after arrival; admissions per 1,000 dispensed prescriptions; S1 with distress and market type; and treatment capacity:
  the share of opioid admissions with medication in the plan (METHUSE) and DEA ARCOS buprenorphine and methadone grams per 100 000,
  2009-11, from code/08's output (out/arcos_retail_state_year_drug.csv). The script also reads code/07's, code/24's and code/06's
  outputs in the out folder (us_states_frame.csv, us_states_results.csv, alpert_rerun_frame.csv, fixed_span.csv, tests_countries.csv)
  to check its reproductions against them.

Part 2, countries. EUDA Statistical Bulletin 2023 tables: PDU-06-2 and PDU-1-1, high-risk opioid use per 1,000 aged 15-64; HSR-3-1,
opioid agonist treatment clients; TDI-2004, treatment entrants with an opioid as primary drug. Clients and entrants are per 100 000
total population, World Bank SP.POP.TOTL of the same year. Each country contributes the estimate closest to 2010, earlier year on a tie,
from years before its illicit supply was first seen (code/05's first_seen); a variant drops that restriction. Prescribing, GDP, supply
codes and mortality change come from code/06's frame(), loaded unedited. C1: log measure on log prescribing and log GDP, HC3 errors,
with the Spearman correlation. C2: the market countries' levels and ranks, as description. The USA and Canada have no EUDA measure.

Death counts are never written; only rates of admissions, model estimates and the EUDA measures.

  python3 -I code/31_population_at_risk.py <TEDS-A 2006-2023 zip> <inputs/1a_out folder> <arcos_county_year.csv> <wonder_pooled date folder> \
      <wonder_state_year_t404 date folder> <euda_statsbulletin_2023 date folder> <incb csv> \
      <who deaths csv> <arrival csv> <worldbank_gdp_per_capita date folder> <worldbank_population date folder> <out folder>

Writes, in the out folder: population_at_risk_teds_state_year.csv (admission counts by state and year; reused if present, delete it
to recount), population_at_risk_states.csv (each state's measures and flags), population_at_risk_results.csv (every estimate),
population_at_risk_countries.csv (each European country's measures and years), population_at_risk_market_countries.csv (C2) and
population_at_risk_checks.csv (reproduction and coverage checks).
"""
import importlib.util
import json
import pathlib
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
OPIOIDS = [5, 6, 7]
WINDOWS = {"2009-11": (2009, 2011), "2006-08": (2006, 2008)}
# Codebook, "State Exclusions" (teds-a-2006_2023-ds0001-info-codebook_v1.pdf, p. 6), for the years used here
CODEBOOK_DROPPED = {2006: [11], 2007: [1], 2008: [28], 2009: [11, 28], 2010: [54], 2011: [1, 54], 2012: [54]}
MEASURES = {"anyop": "any opioid in SUB1-SUB3", "heroin1": "primary heroin", "othop1": "primary other opiates and synthetics"}


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- TEDS-A
def teds_counts(zip_path, out):
    path = out / "population_at_risk_teds_state_year.csv"
    if path.exists():
        print(f"reusing {path.name}")
        return pd.read_csv(path)
    cols = ["STFIPS", "ADMYR", "SUB1", "SUB2", "SUB3", "METHUSE"]
    parts, rows, t0 = [], 0, time.time()
    with zipfile.ZipFile(zip_path) as z:
        member = z.infolist()[0]
        print(f"reading {member.filename}, {member.file_size / 1e9:.1f} GB")
        with z.open(member) as fh:
            for ch in pd.read_csv(fh, usecols=cols, dtype="int16", chunksize=2_000_000):
                rows += len(ch)
                anyop = ch.SUB1.isin(OPIOIDS) | ch.SUB2.isin(OPIOIDS) | ch.SUB3.isin(OPIOIDS)
                k = pd.DataFrame(dict(STFIPS=ch.STFIPS, ADMYR=ch.ADMYR, n_all=1, anyop=anyop.astype("int32"),
                                      heroin1=(ch.SUB1 == 5).astype("int32"), othop1=(ch.SUB1 == 7).astype("int32"),
                                      op1=ch.SUB1.isin(OPIOIDS).astype("int32"),
                                      anyop_mat_yes=(anyop & (ch.METHUSE == 1)).astype("int32"),
                                      anyop_mat_known=(anyop & ch.METHUSE.isin([1, 2])).astype("int32")))
                parts.append(k.groupby(["STFIPS", "ADMYR"], as_index=False).sum())
                print(f"  {rows:,} rows, {time.time() - t0:.0f} s", flush=True)
    t = pd.concat(parts).groupby(["STFIPS", "ADMYR"], as_index=False).sum()
    t.to_csv(path, index=False)
    print(f"{rows:,} admissions counted")
    return t


def state_population(d77):
    a = pd.read_csv(pathlib.Path(d77) / "wonder_t404_state_year_1999_2020.tsv", sep="\t", dtype=str)
    a["st"] = a.state.str.extract(r"\((\d\d)\)")[0].astype(int)
    a["year"] = a.year.str[:4].astype(int)
    a["pop"] = a.population.str.replace(",", "").astype(float)
    return a.set_index(["st", "year"])["pop"]


def teds_windows(t, pop, states, checks):
    """Admissions per 100 000 in each window over the years present; flags; the codebook's drops checked against the records."""
    t = t.rename(columns={"STFIPS": "st", "ADMYR": "year"})
    rows = []
    for st in states:
        r = dict(st=st)
        for w, (y0, y1) in WINDOWS.items():
            k = t[(t.st == st) & t.year.between(y0, y1) & (t.n_all > 0)]
            years = sorted(k.year)
            p = sum(pop.get((st, y), np.nan) for y in years)
            tag = w[:4]
            r[f"years_{tag}"] = " ".join(str(y) for y in years)
            r[f"missing_{tag}"] = " ".join(str(y) for y in range(y0, y1 + 1) if y not in years)
            r[f"incomplete_{tag}"] = int(len(years) < y1 - y0 + 1)
            for m in list(MEASURES) + ["op1", "n_all"]:
                r[f"{m}_{tag}"] = k[m].sum() / p * 1e5 if years else np.nan
            r[f"mat_share_{tag}"] = k.anyop_mat_yes.sum() / k.anyop_mat_known.sum() if k.anyop_mat_known.sum() else np.nan
            r[f"mat_known_{tag}"] = k.anyop_mat_known.sum() / k.anyop.sum() if k.anyop.sum() else np.nan
            # the smallest year's all-admission count against the window's largest: a sign of partial reporting
            r[f"min_to_max_{tag}"] = k.n_all.min() / k.n_all.max() if years else np.nan
        rows.append(r)
    s = pd.DataFrame(rows)
    present = set(zip(t.st[t.n_all > 0], t.year[t.n_all > 0]))
    for y, sts in CODEBOOK_DROPPED.items():
        for st in sts:
            checks.append(dict(check=f"TEDS: codebook drops state {st} in {y}; records present in the file", value=int((st, y) in present)))
    for y in range(2006, 2013):
        absent = sorted(st for st in states if (st, y) not in present)
        checks.append(dict(check=f"TEDS: of the 49 states, FIPS codes with no records in {y}", value=" ".join(map(str, absent)) or "none"))
    return s


# ---------------------------------------------------------------- code/24's post-arrival rate, copied
def post_arrival(d77, s, bound):
    d77 = pathlib.Path(d77)
    a = pd.read_csv(d77 / "wonder_t404_state_year_1999_2020.tsv", sep="\t", dtype=str)
    a["st"] = a.state.str.extract(r"\((\d\d)\)")[0].astype(int)
    a["year"] = a.year.str[:4].astype(int)
    a["pop"] = a.population.str.replace(",", "").astype(float)
    a["hidden"] = a.deaths.eq("Suppressed")
    a["d"] = pd.to_numeric(a.deaths.str.replace(",", ""), errors="coerce")
    m = pd.read_csv(d77 / "wonder_t404_state_year_2018_2024.tsv", sep="\t", dtype=str)
    m = pd.DataFrame(dict(st=m.state.str.extract(r"\((\d\d)\)")[0].astype(int), year=m.year.str[:4].astype(int),
                          pop=m.population.str.replace(",", "").astype(float), hidden=m.deaths.eq("Suppressed"),
                          d=pd.to_numeric(m.deaths.str.replace(",", ""), errors="coerce")))
    ya = a.assign(y=a.d.fillna(bound).where(~a.hidden, bound))[["st", "year", "y", "pop"]]
    ym = m.assign(y=m.d.where(~m.hidden, bound))[["st", "year", "y", "pop"]]
    p = pd.concat([ya, ym[ym.year.between(2021, 2024)]]).set_index(["st", "year"]).sort_index()

    def rate(st, y0, y1):
        k = p.loc[(st, slice(y0, y1)), :]
        return k.y.sum() / k["pop"].sum() * 1e5

    Y = np.floor(s.arrival).astype(int)
    return {"years 1-3 after the arrival year": pd.Series([rate(st, y + 1, y + 3) for st, y in zip(s.st, Y)], index=s.index),
            "years 2-4 after the arrival year": pd.Series([rate(st, y + 2, min(y + 4, 2024)) for st, y in zip(s.st, Y)], index=s.index)}


# ---------------------------------------------------------------- fitting helpers
def ols(d, formula, terms, base, out):
    r = smf.ols(formula, data=d).fit(cov_type="HC3")
    ci = r.conf_int()
    for t in terms:
        if t in r.params:
            out.append(dict(base, formula=formula, n=int(r.nobs), term=t, b=r.params[t], lo=ci.loc[t, 0], hi=ci.loc[t, 1], p=r.pvalues[t],
                            r2=r.rsquared))
    return r


def spear(d, a, b, base, out):
    k = d.dropna(subset=[a, b])
    rho = spearmanr(k[a], k[b])
    out.append(dict(base, formula=f"spearman({a}, {b})", n=len(k), term="rho", b=rho[0], p=rho[1]))


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


# ---------------------------------------------------------------- US states
def us_states(teds_zip, onea, arcos, wonder, d77, out, res, checks):
    t7 = load("t7", "07_us_states.py")
    teds = teds_counts(teds_zip, out)
    pop = state_population(d77)
    frames = {b: t7.state_frame(onea, arcos, wonder, b) for b in (1, 5, 9)}
    s5 = frames[5]
    ref = pd.read_csv(out / "us_states_frame.csv")
    checks.append(dict(check="code/07 frame: states here and in out/us_states_frame.csv", value=f"{len(s5)} and {len(ref)}"))
    checks.append(dict(check="code/07 frame: max absolute difference in log prescribing against out/us_states_frame.csv",
                       value=float((s5.set_index("st").lrx - ref.set_index("st").lrx).abs().max())))
    adm = teds_windows(teds, pop, list(s5.st), checks)

    # capacity: ARCOS retail buprenorphine and methadone grams per 100 000, 2009-11, as code/07's retail() averages them
    r = pd.read_csv(out / "arcos_retail_state_year_drug.csv")
    r = r[r.year.between(2009, 2011)]
    r["st"] = r.state.map(t7.FIPS)
    pick = lambda pat: r[r.drug.str.contains(pat, regex=True)].groupby(["st", "year"]).grams_per_100k.sum().groupby("st").mean()
    cap = pd.DataFrame({"bup_g": pick(r"^BUPRENORPHINE"), "methadone_g": pick(r"^METHADONE")})

    def add(s):
        s = s.merge(adm, on="st", how="left").merge(cap, left_on="st", right_index=True, how="left")
        for tag in ("2009", "2006"):
            for m in MEASURES:
                s[f"l{m}_{tag}"] = np.log(s[f"{m}_{tag}"])
        s["ladm"] = s.lanyop_2009
        s["lbup"] = np.log(s.bup_g)
        s["adm_per_1000_rx"] = s.anyop_2009 / s.rx  # rx is prescriptions per 100 persons, so this is admissions per 1,000 prescriptions
        s["othop_per_1000_rx"] = s.othop1_2009 / s.rx
        return s

    S = {b: add(f) for b, f in frames.items()}
    s = S[5]
    keep = ["st", "state", "rx", "arrival", "white", "years_2009", "missing_2009", "incomplete_2009", "years_2006", "missing_2006", "incomplete_2006",
            "anyop_2009", "heroin1_2009", "othop1_2009", "op1_2009", "n_all_2009", "anyop_2006", "heroin1_2006", "othop1_2006", "min_to_max_2009",
            "min_to_max_2006", "mat_share_2009", "mat_known_2009", "adm_per_1000_rx", "othop_per_1000_rx", "bup_g", "methadone_g"]
    s[keep].to_csv(out / "population_at_risk_states.csv", index=False)
    complete = s[(s.incomplete_2009 == 0)]
    complete06 = s[(s.incomplete_2006 == 0)]

    # S1
    for tag, win in (("2009", "2009-11"), ("2006", "2006-08")):
        for sname, d in (("49 states", s), ("complete TEDS years", s[s[f"incomplete_{tag}"] == 0])):
            for m, lab in MEASURES.items():
                base = dict(part="US", test="S1", outcome=f"log admissions per 100 000, {lab}, {win}", sample=sname, bound=np.nan)
                spear(d, "rx", f"{m}_{tag}", base, res)
                ols(d, f"l{m}_{tag} ~ lrx", ["lrx"], base, res)
    base = dict(part="US", test="S1 capacity", outcome="log admissions per 100 000, any opioid, 2009-11", sample="49 states", bound=np.nan)
    ols(s.dropna(subset=["lbup"]), "ladm ~ lrx + lbup", ["lrx", "lbup"], dict(base, model="with log buprenorphine grams"), res)
    ols(s, "ladm ~ lrx + zx + white", ["lrx", "zx", "white"], dict(base, model="with distress and market type"), res)
    q1, q4 = s.rx.quantile(0.25), s.rx.quantile(0.75)
    for lab, k in (("lowest quarter", s[s.rx <= q1]), ("highest quarter", s[s.rx >= q4])):
        checks.append(dict(check=f"US: median any-opioid admissions per 1,000 prescriptions, 2009-11, {lab} of prescribing ({len(k)} states)",
                           value=float(k.adm_per_1000_rx.median())))
        checks.append(dict(check=f"US: median primary other-opiate admissions per 1,000 prescriptions, 2009-11, {lab} of prescribing",
                           value=float(k.othop_per_1000_rx.median())))
    checks.append(dict(check="US: median MAT-planned share of opioid admissions with METHUSE known, 2009-11, 47 states", value=float(s.mat_share_2009.median())))
    for a, b, lab in (("anyop_2009", "bup_g", "admissions with buprenorphine grams"), ("rx", "bup_g", "prescribing with buprenorphine grams"),
                      ("anyop_2009", "methadone_g", "admissions with methadone grams"), ("anyop_2009", "mat_share_2009", "admissions with share planned on medication"),
                      ("rx", "mat_share_2009", "prescribing with share planned on medication"), ("rx", "adm_per_1000_rx", "prescribing with admissions per 1,000 prescriptions"),
                      ("rx", "othop_per_1000_rx", "prescribing with other-opiate admissions per 1,000 prescriptions")):
        spear(s, a, b, dict(part="US", test="capacity and ratio", outcome=lab, sample="49 states", bound=np.nan), res)

    # S2: code/24's frame and outcome
    f24 = pd.read_csv(out / "alpert_rerun_frame.csv")
    f24 = f24.merge(s[["st", "ladm", "lanyop_2006", "lheroin1_2009", "lothop1_2009", "incomplete_2009", "incomplete_2006"]], on="st", how="left")
    checks.append(dict(check="S2: states in code/24's frame without an admissions rate", value=int(f24.ladm.isna().sum())))
    fs = pd.read_csv(out / "fixed_span.csv")
    for bound in (1, 9):
        rates = post_arrival(d77, f24, bound)
        for span, y in rates.items():
            d = f24.assign(ly=np.log(y.astype(float)))
            main = span.startswith("years 1-3")
            if main:
                r = smf.ols("ly ~ lrx + za", data=d).fit(cov_type="HC3")
                ref = fs[(fs.span == span) & (fs.hidden == bound) & (fs.model == "dispensing, arrival date")].iloc[0]
                checks.append(dict(check=f"S2 reproduction, hidden at {bound}: lrx here minus out/fixed_span.csv", value=r.params["lrx"] - ref.b))
                checks.append(dict(check=f"S2 reproduction, hidden at {bound}: se here minus out/fixed_span.csv", value=r.bse["lrx"] - ref.se))
            samples = [("49 states", d)] + ([("complete TEDS years", d[d.incomplete_2009 == 0])] if main else [])
            for sname, k in samples:
                specs = [("prescribing, arrival (the paper's model)", "ly ~ lrx + za", ["lrx", "za"]),
                         ("prescribing and admissions, arrival", "ly ~ lrx + ladm + za", ["lrx", "ladm", "za"]),
                         ("admissions, arrival", "ly ~ ladm + za", ["ladm", "za"]),
                         ("admissions alone", "ly ~ ladm", ["ladm"])]
                if main and sname == "49 states":
                    specs += [("prescribing and admissions, arrival, distress, market", "ly ~ lrx + ladm + za + zx + white", ["lrx", "ladm"]),
                              ("admissions, arrival, distress, market", "ly ~ ladm + za + zx + white", ["ladm"]),
                              ("prescribing and admissions 2006-08, arrival", "ly ~ lrx + lanyop_2006 + za", ["lrx", "lanyop_2006"]),
                              ("prescribing and primary heroin admissions, arrival", "ly ~ lrx + lheroin1_2009 + za", ["lrx", "lheroin1_2009"]),
                              ("prescribing and primary other-opiate admissions, arrival", "ly ~ lrx + lothop1_2009 + za", ["lrx", "lothop1_2009"])]
                if main and sname == "complete TEDS years":
                    k6 = d[d.incomplete_2006 == 0]
                    ols(k6, "ly ~ lrx + lanyop_2006 + za", ["lrx", "lanyop_2006"],
                        dict(part="US", test="S2", outcome=f"log synthetic-opioid rate, {span}", sample="complete TEDS years 2006-08", bound=bound,
                             model="prescribing and admissions 2006-08, arrival"), res)
                for lab, formula, terms in specs:
                    ols(k, formula, terms, dict(part="US", test="S2", outcome=f"log synthetic-opioid rate, {span}", sample=sname, bound=bound, model=lab), res)
                if main and sname == "complete TEDS years":
                    # which of the incomplete states moves the paper's own model
                    for st in sorted(d.loc[d.incomplete_2009 == 1, "state"]):
                        ols(d[d.state != st], "ly ~ lrx + za", ["lrx"], dict(part="US", test="S2 drop one", outcome=f"log synthetic-opioid rate, {span}",
                                                                          sample=f"without {st}", bound=bound, model="prescribing, arrival (the paper's model)"), res)
    iq = dict(lrx=iqr(f24.lrx), ladm=iqr(f24.ladm))
    checks.append(dict(check="US: middle half of log prescribing, 49 states (75th less 25th percentile)", value=iq["lrx"]))
    checks.append(dict(check="US: middle half of log any-opioid admissions 2009-11, 49 states", value=iq["ladm"]))
    for q in (0.25, 0.75):
        checks.append(dict(check=f"US: any-opioid admissions per 100 000, 2009-11, {int(q * 100)}th percentile", value=float(s.anyop_2009.quantile(q))))
        checks.append(dict(check=f"US: prescribing per 100, 2009-11, {int(q * 100)}th percentile", value=float(s.rx.quantile(q))))

    # S3: code/07's panel with admissions x after arrival
    MID = t7.MID
    for bound, sb in S.items():
        for sname, d in (("49 states", sb), ("complete TEDS years", sb[sb.incomplete_2009 == 0])):
            long = d.melt(id_vars=["st", "lrx", "ladm", "arrival"], value_vars=[f"od_{k}" for k in t7.OD], var_name="window", value_name="od")
            long["window"] = long.window.str[3:]
            long["post"] = (long.window.map(MID) >= long.arrival).astype(int)
            long["lod"] = np.log(long.od)
            for w in ("2011-13", "2014-16", "2017-19"):
                long[f"lrx_{w[:4]}"] = long.lrx * (long.window == w)
            trend = "lrx_2011 + lrx_2014 + lrx_2017"
            specs = [("code/07's gradient after arrival", "lod ~ C(st) + C(window) + post + lrx:post"),
                     ("with admissions x after arrival", "lod ~ C(st) + C(window) + post + lrx:post + ladm:post"),
                     ("admissions x after arrival alone", "lod ~ C(st) + C(window) + post + ladm:post"),
                     ("with admissions x after arrival, beyond the calendar change in the prescribing gradient",
                      "lod ~ C(st) + C(window) + " + trend + " + post + lrx:post + ladm:post")]
            for lab, formula in specs:
                r = smf.ols(formula, data=long).fit(cov_type="cluster", cov_kwds={"groups": long.st})
                ci = r.conf_int()
                for term in ("lrx:post", "ladm:post", "post"):
                    if term in r.params:
                        res.append(dict(part="US", test="S3", outcome="log overdose rate, four windows", sample=sname, bound=bound, model=lab, formula=formula,
                                        n=int(r.nobs), term=term, b=r.params[term], lo=ci.loc[term, 0], hi=ci.loc[term, 1], p=r.pvalues[term], r2=r.rsquared))
                if lab.startswith("code/07") and sname == "49 states":
                    ref = pd.read_csv(out / "us_states_results.csv")
                    ref = ref[(ref.model == "P2 steepening | gradient after the state's arrival") & (ref.bound == bound) & (ref.term == "lrx:post")].iloc[0]
                    checks.append(dict(check=f"S3 reproduction, bound {bound}: lrx:post here minus out/us_states_results.csv", value=r.params["lrx:post"] - ref.b))
    return s, iq


# ---------------------------------------------------------------- countries
def euda_table(path, first_col_is_country=True):
    x = pd.read_excel(path, header=None)
    hdr = x.index[x.iloc[:, 0].astype(str).str.strip() == "Country"][0]
    cols = [str(c).strip() for c in x.iloc[hdr]]
    body = x.iloc[hdr + 1:].copy()
    body.columns = cols
    body = body[body.Country.notna() & ~body.Country.astype(str).str.startswith("(")]
    body["Country"] = body.Country.astype(str).str.replace("*", "", regex=False).str.strip()
    return body


def long_years(body, source, unit):
    yc = [c for c in body.columns if c.replace(".0", "").isdigit()]
    k = body.melt(id_vars=["Country"], value_vars=yc, var_name="year", value_name="value")
    k["year"] = k.year.str.replace(".0", "", regex=False).astype(int)
    k["value"] = pd.to_numeric(k.value, errors="coerce")
    return k.dropna(subset=["value"]).assign(source=source, unit=unit)


EUDA_NAMES = {"Netherlands": "Netherlands (Kingdom of the)"}


def countries(euda, incb, who, arrival, gdp, wbpop, out, res, checks):
    t6 = load("t6", "06_tests.py")
    m = t6.frame(incb, who, arrival, gdp)
    ref = pd.read_csv(out / "tests_countries.csv")
    checks.append(dict(check="code/06 frame: countries here and in out/tests_countries.csv", value=f"{len(m)} and {len(ref)}"))
    checks.append(dict(check="code/06 frame: max absolute difference in log prescribing against out/tests_countries.csv",
                       value=float((m.set_index("country").lrx_pre - ref.set_index("country").lrx_pre).abs().max())))
    iq_c = iqr(m.lrx_pre)
    checks.append(dict(check="countries: middle half of log(1 + S-DDD) in code/06's 60 (75th less 25th percentile)", value=iq_c))
    euda = pathlib.Path(euda)
    # high-risk opioid use: trends 2009-21 and the most recent estimates
    pdu = long_years(euda_table(euda / "PDU-06-2.xlsx"), "PDU-06-2", "per 1,000 aged 15-64")
    recent = euda_table(euda / "PDU-1-1.xlsx")
    central = [c for c in recent.columns if c.startswith("Central")][0]
    def end_year(v):  # "2021" or a span such as "2014-15", dated to its last year as PDU-06-2 places it
        parts = str(v).replace(".0", "").split("-")
        return int(parts[0][:4 - len(parts[1])] + parts[1]) if len(parts) == 2 else int(parts[0])

    recent = recent[recent.Year.notna()].assign(value=lambda d: pd.to_numeric(d[central], errors="coerce"), year=lambda d: d.Year.map(end_year))
    recent = recent.dropna(subset=["value"])[["Country", "year", "value"]].assign(source="PDU-1-1", unit="per 1,000 aged 15-64")
    pdu = pd.concat([pdu, recent[~recent.set_index(["Country", "year"]).index.isin(pdu.set_index(["Country", "year"]).index)]])
    oat = long_years(euda_table(euda / "HSR-3-1.xlsx"), "HSR-3-1", "per 100 000")
    tdi = long_years(euda_table(euda / "TDI-2004.xlsx"), "TDI-2004", "per 100 000")
    # World Bank population, keyed as code/06 keys GDP
    g = json.loads((pathlib.Path(wbpop) / "SP.POP.TOTL.json").read_text())[1]
    g = pd.DataFrame([(t6.key(r["country"]["value"]), int(r["date"]), r["value"]) for r in g if r["value"] is not None], columns=["key", "year", "pop"])
    g["key"] = g.key.replace(t6.WB_ALIAS)
    wb = g.drop_duplicates(["key", "year"]).set_index(["key", "year"])["pop"]
    first = m.set_index("country").first_seen
    keymap = m.set_index("country").key
    rows, chosen = [], {}
    for name, d in (("hr", pdu), ("oat", oat), ("tdi", tdi)):
        d = d.assign(country=d.Country.replace(EUDA_NAMES))
        unmatched = sorted(set(d.country) - set(m.country))
        checks.append(dict(check=f"EUDA {name}: countries with data but not in code/06's 60", value=" ".join(unmatched) or "none"))
        d = d[d.country.isin(m.country)].copy()
        if name != "hr":
            d["value"] = [v / wb.get((keymap[c], y), np.nan) * 1e5 for c, v, y in zip(d.country, d.value, d.year)]
        d = d.dropna(subset=["value"])
        d["dist"] = (d.year - 2010).abs()
        for rule in ("before arrival", "any year"):
            k = d if rule == "any year" else d[d.country.map(first).isna() | (d.year < d.country.map(first))]
            pick = k.sort_values(["country", "dist", "year"]).groupby("country").head(1)
            chosen[(name, rule)] = pick.set_index("country")
            for _, p in pick.iterrows():
                rows.append(dict(measure=name, rule=rule, country=p.country, year=p.year, value=p.value, source=p.source, unit=p.unit))
    cm = pd.DataFrame(rows)
    allc = sorted(set(cm.country))
    wide = m[m.country.isin(allc)][["country", "status", "first_seen", "anyill", "est", "rx_pre", "lrx_pre", "lgdp", "narc_1012", "narc_1921", "narc_change"]].copy()
    for (name, rule), p in chosen.items():
        tag = f"{name}_{'pre' if rule == 'before arrival' else 'any'}"
        wide[tag] = wide.country.map(p.value)
        wide[f"{tag}_year"] = wide.country.map(p.year)
        wide[f"{tag}_source"] = wide.country.map(p.source)
    wide.to_csv(out / "population_at_risk_countries.csv", index=False)
    labels = {"hr": "high-risk opioid use per 1,000 aged 15-64", "oat": "opioid agonist treatment clients per 100 000",
              "tdi": "opioid treatment entrants per 100 000"}
    for name, lab in labels.items():
        for rule, tag in (("before arrival", "pre"), ("any year", "any")):
            col = f"{name}_{tag}"
            d = wide.dropna(subset=[col, "lrx_pre", "lgdp"]).assign(y=lambda t: np.log(t[col]))
            base = dict(part="countries", test="C1", outcome=f"log {lab}, closest to 2010, {rule}", sample=f"European countries with data", bound=np.nan)
            spear(d, "rx_pre", col, base, res)
            ols(d, "y ~ lrx_pre + lgdp", ["lrx_pre", "lgdp"], base, res)
            if rule == "before arrival":
                ols(d, "y ~ lrx_pre", ["lrx_pre"], dict(base, sample="European countries with data, no GDP"), res)
                y0 = d[f"{col}_year"]
                ols(d[y0.between(2008, 2012)], "y ~ lrx_pre + lgdp", ["lrx_pre"], dict(base, sample="estimate dated 2008-12 only"), res)
    # C2: description, the market countries' levels and ranks among the European countries with data
    mk = []
    for name, lab in labels.items():
        col = f"{name}_pre"
        d = wide.dropna(subset=[col]).copy()
        d["rank"] = d[col].rank(ascending=False, method="min").astype(int)
        for _, r in d[d.anyill == 1].iterrows():
            mk.append(dict(measure=lab, country=r.country, status=r.status, first_seen=r.first_seen, value=r[col], year=r[f"{col}_year"],
                           rank=r["rank"], of=len(d), median_all=d[col].median(), narc_change=r.narc_change))
        for _, r in wide[(wide.anyill == 1) & wide[col].isna()].iterrows():
            anyv = r[f"{name}_any"]
            mk.append(dict(measure=lab, country=r.country, status=r.status, first_seen=r.first_seen, value=np.nan, year=np.nan, rank=np.nan, of=len(d),
                           median_all=d[col].median(), narc_change=r.narc_change, note=(f"only after arrival: {anyv:.2f} in {int(r[f'{name}_any_year'])}"
                                                                                       if pd.notna(anyv) else "no estimate in the table")))
    mk = pd.DataFrame(mk)
    mk.to_csv(out / "population_at_risk_market_countries.csv", index=False)
    return wide, mk, iq_c


# ---------------------------------------------------------------- main
def main(teds_zip, onea, arcos, wonder, d77, euda, incb, who, arrival, gdp, wbpop, out):
    out = pathlib.Path(out)
    warnings.filterwarnings("ignore", category=UserWarning)
    res, checks = [], []
    s, iq = us_states(teds_zip, onea, arcos, wonder, d77, out, res, checks)
    wide, mk, iq_c = countries(euda, incb, who, arrival, gdp, wbpop, out, res, checks)
    r = pd.DataFrame(res)
    r["model"] = r.model.fillna("")
    # plain arithmetic: the ratio of the outcome at the 75th to the 25th percentile of the regressor, from its slope
    span = {"lrx": iq["lrx"], "lrx:post": iq["lrx"], "ladm": iq["ladm"], "ladm:post": iq["ladm"], "lrx_pre": iq_c}
    r["ratio_middle_half"] = [np.exp(b * span[t]) if t in span else np.nan for b, t in zip(r.b, r.term)]
    r["ratio_lo"] = [np.exp(b * span[t]) if t in span else np.nan for b, t in zip(r.lo, r.term)]
    r["ratio_hi"] = [np.exp(b * span[t]) if t in span else np.nan for b, t in zip(r.hi, r.term)]
    r.to_csv(out / "population_at_risk_results.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "population_at_risk_checks.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_rows", 600, "display.max_columns", 30, "display.max_colwidth", 80):
        print(pd.DataFrame(checks).to_string(index=False))
        print(s[["state", "rx", "anyop_2009", "heroin1_2009", "othop1_2009", "missing_2009", "missing_2006", "min_to_max_2009", "mat_share_2009",
                 "adm_per_1000_rx", "bup_g"]].round(2).to_string(index=False))
        cols = ["test", "outcome", "sample", "bound", "model", "n", "term", "b", "lo", "hi", "p", "r2", "ratio_middle_half"]
        print(r[cols].round(3).to_string(index=False))
        print(wide.round(3).to_string(index=False))
        print(mk.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:13])
