"""Further checks, set after the paper's results and a review were seen; every variant fitted is written out.

Items:
  1   The absolute scale. Every US test in the paper is on the log or ratio scale. Here: deaths per 100 000 a year gained at the
      75th against the 25th percentile of dispensing, for states (synthetic-opioid rate over the three years after arrival, arrival
      held, with and without the District of Columbia) and for counties within states (synthetic-opioid rate 2014-20, and the change
      in all drug poisoning from 2008-10 to 2017-19), population-weighted least squares with state effects.
  3b  The county-year model of synthetic-opioid deaths (code/33's 5a) on the states reached in 2019 or later only, the states that
      supply its reference years before arrival.
  5   The heroin route. Does dispensing in 2009-11 predict the growth of heroin between 2009-11 and the three years before each
      state's arrival, in primary-heroin treatment admissions (TEDS-A) and in heroin laboratory reports per resident (NFLIS)? And
      does heroin just before arrival carry the rise after it?
  6   Arrival dated without the top-50 denominator: the first year with one fentanyl report per 100 000 residents, and the first
      year fentanyl made up 5%, 10% or 20% of heroin plus fentanyl reports, each required to hold the next year. Then dispensing
      against each date, the triplicate gap adjusted for each, and the state model on each.
  7   What the 15 heaviest prescribers consumed, by substance, and the heaviest-quarter contrasts with countries ranked on oxycodone
      plus hydrocodone, on morphine, and on everything but fentanyl.
  15  Prescribing through 2012-14 as the exposure.
  20  2017-19 as the end window, before the pandemic years.

Inputs: this paper's out/ folder; the two CDC WONDER state-by-year files (wonder_state_year_t404); the county files (wonder_pooled) and
the county-year file (wonder_county_year), read with code/33's own functions; the derived inputs folder (county_base.csv and the
half-year clock); the ARCOS county file; the NFLIS state-by-year counts built for the arrival clock (nflis_dqs_state_year.csv) and
the NFLIS heroin file. Writes out/further_checks.csv and out/composition_sddd.csv: estimates and consumption figures only, no
count of deaths.

  python3 code/39_further_checks.py <out> <wonder_state_year_t404 date folder> <wonder_pooled date folder>
      <wonder_county_year date folder> <derived inputs folder> <arcos_county_year.csv> <nflis_dqs_state_year.csv> <NFLIS heroin csv>
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr

HERE = pathlib.Path(__file__).resolve().parent
PRE = ["2000-2002", "2003-2005", "2006-2008", "2009-2011"]
USA, CAN = "United States of America", "Canada"
CURRENT = {USA, CAN, "Estonia", "Latvia", "Lithuania"}


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row(res, item, check, sample, term, r=None, **kw):
    d = dict(item=item, check=check, sample=sample, term=term)
    if r is not None:
        ci = r.conf_int().loc[term]
        d.update(b=r.params[term], lo=ci[0], hi=ci[1], p=r.pvalues[term], n=int(r.nobs))
    d.update(kw)
    res.append(d)


def state_rates(d77):
    """Synthetic-opioid deaths and population by state and year: the final file to 2020 and the provisional file for 2021-24, as
    code/24 reads them. No count is hidden in any state's years after arrival; a hidden count elsewhere enters at 5."""
    d77 = pathlib.Path(d77)
    out = []
    for f, years in (("wonder_t404_state_year_1999_2020.tsv", (1999, 2020)), ("wonder_t404_state_year_2018_2024.tsv", (2021, 2024))):
        a = pd.read_csv(d77 / f, sep="\t", dtype=str)
        k = pd.DataFrame(dict(st=a.state.str.extract(r"\((\d\d)\)")[0].astype(int), year=a.year.str[:4].astype(int),
                              pop=a.population.str.replace(",", "").astype(float), hidden=a.deaths.eq("Suppressed"),
                              d=pd.to_numeric(a.deaths.str.replace(",", ""), errors="coerce")))
        out.append(k[k.year.between(*years)])
    p = pd.concat(out)
    p["y"] = np.where(p.hidden, 5, p.d)
    return p.set_index(["st", "year"]).sort_index()


def span_rate(p, st, y0, y1, strict=True):
    k = p.loc[(st, slice(y0, y1)), :]
    assert not (strict and k.hidden.any()), f"hidden count in state {st}, {y0}-{y1}"
    return k.y.sum() / k["pop"].sum() * 1e5


def item1_states(s, p, res):
    s = s.copy()
    s["rate13"] = [span_rate(p, st, r.Y + 1, r.Y + 3) for st, r in s.iterrows()]
    q25, q75 = np.percentile(s.rx, [25, 75])
    iqr = np.log(q75 / q25)
    for name, k in (("49 states", s), ("48 states, without DC", s[s.state != "DC"])):
        r = smf.ols("rate13 ~ lrx + za", data=k).fit(cov_type="HC3")
        ci = r.conf_int().loc["lrx"]
        row(res, "1", "synthetic-opioid deaths per 100 000 a year, years 1-3 after arrival, on log dispensing with arrival held; "
            "b is per log unit, diff is 75th against 25th percentile of state dispensing", name, "lrx", r,
            diff=r.params.lrx * iqr, diff_lo=ci[0] * iqr, diff_hi=ci[1] * iqr, mean_outcome=k.rate13.mean(), p25=q25, p75=q75)
        r = smf.ols("Q('od_2008-10') ~ lrx", data=k).fit(cov_type="HC3")
        ci = r.conf_int().loc["lrx"]
        row(res, "1", "context: all drug poisoning per 100 000 a year, 2008-10, on log dispensing (hidden county counts at 5)", name, "lrx", r,
            diff=r.params.lrx * iqr, diff_lo=ci[0] * iqr, diff_hi=ci[1] * iqr, mean_outcome=k["od_2008-10"].mean(), p25=q25, p75=q75)
    return s


def wls_within(d, y, res, item, check, sample, iqr, **kw):
    """Population-weighted least squares of a county rate on log dispensing with state effects, errors clustered by state."""
    d = d.dropna(subset=[y, "lrx", "w"])
    r = smf.wls(f"{y} ~ lrx + C(st)", data=d, weights=d.w).fit(cov_type="cluster", cov_kwds={"groups": d.st})
    ci = r.conf_int().loc["lrx"]
    row(res, item, check, sample, "lrx", r, diff=r.params.lrx * iqr, diff_lo=ci[0] * iqr, diff_hi=ci[1] * iqr,
        mean_outcome=float(np.average(d[y], weights=d.w)), states=d.st.nunique(), **kw)


def item1_counties(c33, onea, arcos, wonder, res):
    cb = c33.counties(onea, arcos)
    cb = cb[cb.rx_missing == 0].copy()
    p25, p75 = np.percentile(cb.rx, [25, 75])
    iqr = np.log(p75 / p25)
    for bound in c33.BOUNDS:
        t = c33.read_wonder(wonder / "mcd_t404_2014_2020_county.tsv", bound).merge(cb[["fips", "st", "lrx", "A"]], on="fips")
        t["rate"] = t.deaths / t.population * 1e5  # the file's population is person-years over 2014-20
        t["w"] = t.population
        wls_within(t, "rate", res, "1", "counties within states: synthetic-opioid deaths per 100 000 a year, 2014-20, on log dispensing; "
                   "diff is 75th against 25th percentile of county dispensing", "all states", iqr, bound=bound, p25=p25, p75=p75)
        wls_within(t[t.A <= 2017], "rate", res, "1", "counties within states: synthetic-opioid deaths per 100 000 a year, 2014-20, on log dispensing; "
                   "diff is 75th against 25th percentile of county dispensing", "states reached by 2017", iqr, bound=bound, p25=p25, p75=p75)
        w = {}
        for win in ("2008-10", "2017-19"):
            d = c33.read_wonder(wonder / f"{c33.WFILE[win]}_county.tsv", bound)[["fips", "deaths", "population"]]
            w[win] = d.assign(rate=d.deaths / d.population * 1e5).set_index("fips")
        k = cb[["fips", "st", "lrx", "A"]].copy()
        k["r0"], k["r1"] = k.fips.map(w["2008-10"].rate), k.fips.map(w["2017-19"].rate)
        k["w"] = k.fips.map(w["2017-19"].population)
        k["change"] = k.r1 - k.r0
        for name, kk in (("all states", k), ("states reached by 2017", k[k.A <= 2017])):
            wls_within(kk, "change", res, "1", "counties within states: change in all drug poisoning per 100 000 a year, 2008-10 to 2017-19, "
                       "on log dispensing", name, iqr, bound=bound, p25=p25, p75=p75)
            wls_within(kk, "r0", res, "1", "context, counties within states: all drug poisoning per 100 000 a year in 2008-10 on log dispensing",
                       name, iqr, bound=bound, p25=p25, p75=p75)
    return cb, iqr


def item3b(c33, cb, iqr, wcy, res):
    out = []
    for bound in c33.BOUNDS:
        cy = c33.read_county_year(wcy / "wonder_county_year_t404.tsv", bound).merge(cb[["fips", "st", "state", "lrx", "A"]], on="fips")
        cy["since"] = cy.year - cy.A
        cy = cy.rename(columns={"year": "yr"})
        cy = cy[cy.A >= 2019]
        g = cy.groupby("fips")
        cy = cy[(g.deaths.transform("sum") > 0) & (g.yr.transform("size") > 1)].copy()
        bins = [(f"lrx_x_since_{lab}", cy.lrx * cy.since.between(lo, hi)) for lab, lo, hi in c33.YEAR_BINS if (cy.since.between(lo, hi)).any()]
        c33.run_years(cy, sorted(cy.yr.unique()), bins, "5a, states reached in 2019 or later", bound, out, False, iqr)
        if bound == 5:
            c33.run_years(cy, sorted(cy.yr.unique()), bins, "5a, states reached in 2019 or later, censored Poisson", "censored 1-9", out, False, iqr, censored=True)
            info = dict(states=cy.st.nunique(), counties=cy.fips.nunique(), hidden_share=float(cy.hidden.mean()),
                        states_list=" ".join(sorted(cy.state.unique())))
    for r in out:
        res.append(dict(item="3b", check=r["model"], sample=f"{info['states']} states ({info['states_list']}), {r['counties']} counties; "
                        f"share of county-years hidden {info['hidden_share']:.2f}", term=r["term"], b=r["b"], lo=r["lo"], hi=r["hi"], p=r["p"],
                        n=r["obs"], bound=r["bound"], ratio=r["ratio"], ratio_lo=r["ratio_lo"], ratio_hi=r["ratio_hi"], converged=r["converged"]))


def window_mean(series, st, y0, y1, need=2):
    """Mean of a state-year series over y0..y1, needing at least `need` years present."""
    v = [series.get((st, y)) for y in range(y0, y1 + 1)]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if len(v) >= need else np.nan


def item5(s, p, out, heroin_csv, res):
    pop = p["pop"]
    t = pd.read_csv(out / "population_at_risk_teds_state_year.csv")
    t = t[t.ADMYR.between(2006, 2023)]
    t["rate"] = [h / pop.get((st, y), np.nan) * 1e5 for h, st, y in zip(t.heroin1, t.STFIPS, t.ADMYR)]
    teds = t.set_index(["STFIPS", "ADMYR"]).rate
    h = pd.read_csv(heroin_csv, dtype=str)
    h = h[h.STATE_FIPS.notna()]
    h = h.assign(st=h.STATE_FIPS.astype(int), year=h.YYYY.astype(int), n=h.DRUG_REPORTS.astype(float)).groupby(["st", "year"]).n.sum()
    nflis = pd.Series({k: v / pop.get(k, np.nan) * 1e5 for k, v in h.items()})
    s = s.copy()
    for tag, ser, base in (("teds", teds, (2009, 2011)), ("nflis", nflis, (2010, 2011))):
        s[f"{tag}_base"] = [window_mean(ser, st, *base) for st in s.index]
        s[f"{tag}_pre"] = [window_mean(ser, st, r.Y - 3, r.Y - 1) for st, r in s.iterrows()]
        s[f"l{tag}_pre"] = np.log(s[f"{tag}_pre"].where(s[f"{tag}_pre"] > 0))
        s[f"{tag}_growth"] = np.log((s[f"{tag}_pre"] / s[f"{tag}_base"]).where((s[f"{tag}_pre"] > 0) & (s[f"{tag}_base"] > 0)))
    s["lrate13"] = np.log(s.rate13)
    names = {"teds": "primary-heroin treatment admissions per 100 000 (TEDS-A)", "nflis": "heroin laboratory reports per 100 000 (NFLIS)"}
    for tag, lab in names.items():
        # states reached in 2014 have a window before arrival, 2011-13, that overlaps the base years; shown with and without them
        for sname, k in (("all states with data", s), ("states reached in 2016 or later", s[s.Y >= 2016])):
            k = k.dropna(subset=[f"{tag}_growth"])
            for f, term in ((f"{tag}_growth ~ lrx", "lrx"), (f"{tag}_growth ~ lrx + white", "lrx"), (f"l{tag}_pre ~ lrx", "lrx"),
                            (f"l{tag}_pre ~ lrx + white", "lrx")):
                r = smf.ols(f, data=k).fit(cov_type="HC3")
                oc = "growth from the base years to the three years before arrival, log ratio" if "growth" in f else "level in the three years before arrival, log"
                row(res, "5", f"{lab}: {oc}; model {f}", sname, term, r)
            rho, pv = spearmanr(k.rx, k[f"{tag}_growth"])
            row(res, "5", f"{lab}: Spearman of dispensing with growth to the three years before arrival", sname, "rho", b=rho, p=pv, n=len(k))
        k = s.dropna(subset=[f"l{tag}_pre"])
        for f, terms in ((f"lrate13 ~ lrx + za + l{tag}_pre", ["lrx", f"l{tag}_pre"]), (f"lrate13 ~ l{tag}_pre + za", [f"l{tag}_pre"]),
                         ("lrate13 ~ lrx + za", ["lrx"])):
            for sname, kk in (("states with data", k), ("states with data, without DC", k[k.state != "DC"])):
                r = smf.ols(f, data=kk).fit(cov_type="HC3")
                for term in terms:
                    row(res, "5", f"log synthetic-opioid rate, years 1-3 after arrival; heroin before arrival is {lab}; model {f}", sname, term, r)
    return s


def item6(s, p, nflis_year, heroin_csv, res):
    n = pd.read_csv(nflis_year)
    n = n[n.st_fips.notna()].assign(st=lambda d: d.st_fips.astype(int))
    h = pd.read_csv(heroin_csv, dtype=str)
    h = h[h.STATE_FIPS.notna()]
    h = h.assign(st=h.STATE_FIPS.astype(int), year=h.YYYY.astype(int), n=h.DRUG_REPORTS.astype(float)).groupby(["st", "year"]).n.sum().rename("heroin")
    n = n.merge(h.reset_index(), on=["st", "year"], how="left").fillna({"heroin": 0})
    n = n[n.year <= 2025]  # 2026 is a half-year in both pulls
    n["opioid_share"] = n.fent_category_reports / (n.fent_category_reports + n.heroin).where((n.fent_category_reports + n.heroin) > 0)

    def first_held(col, cut):
        o = {}
        for st, g in n.sort_values("year").groupby("st"):
            ok = (g[col] >= cut).to_numpy()
            yrs = g.year.to_numpy()
            hit = [yrs[i] for i in range(len(yrs) - 1) if ok[i] and ok[i + 1]]
            o[st] = hit[0] if hit else np.nan
        return pd.Series(o)

    clocks = {"one fentanyl report per 100 000 residents": first_held("per100k", 1.0),
              "fentanyl 5% of heroin plus fentanyl reports": first_held("opioid_share", 0.05),
              "fentanyl 10% of heroin plus fentanyl reports": first_held("opioid_share", 0.10),
              "fentanyl 20% of heroin plus fentanyl reports": first_held("opioid_share", 0.20)}
    s = s.copy()
    for name, c in clocks.items():
        k = s.assign(alt=s.index.map(c)).dropna(subset=["alt"])
        k["zalt"] = (k.alt - k.alt.mean()) / k.alt.std()
        k["Yalt"] = k.alt.astype(int)
        base = f"arrival dated as the first year with {name}, held the next year"
        row(res, "6", f"{base}: years, minimum to maximum, and states dated", "49 states", "range", b=k.alt.min(), hi=k.alt.max(), lo=k.alt.min(), n=len(k))
        rho, pv = spearmanr(k.alt, k.arrival)
        row(res, "6", f"{base}: Spearman with the paper's half-year date", "states dated", "rho", b=rho, p=pv, n=len(k))
        rho, pv = spearmanr(k.alt, k.rx)
        row(res, "6", f"{base}: Spearman with dispensing 2009-11", "states dated", "rho", b=rho, p=pv, n=len(k))
        row(res, "6", f"{base}: mean date, triplicate states against the rest (difference in years)", "states dated", "triplicate",
            smf.ols("alt ~ triplicate", data=k).fit(cov_type="HC3"))
        row(res, "6", f"{base}: date on log dispensing (years per log unit)", "states dated", "lrx", smf.ols("alt ~ lrx", data=k).fit(cov_type="HC3"))
        for f in ("growth_since_launch ~ triplicate", "growth_since_launch ~ triplicate + zalt"):
            row(res, "6", f"{base}: triplicate gap in overdose growth, 1995-98 to 2017-19; model {f}", "states dated", "triplicate",
                smf.ols(f, data=k).fit(cov_type="HC3"))
        # the state model on this date: years 1-3 after it, where those years end by 2024
        k2 = k[k.Yalt + 3 <= 2024].copy()
        k2["lr"] = np.log([span_rate(p, st, r.Yalt + 1, r.Yalt + 3, strict=False) for st, r in k2.iterrows()])  # a hidden count enters at 5
        for sname, kk in (("states dated", k2), ("states dated, without DC", k2[k2.state != "DC"])):
            row(res, "6", f"{base}: log synthetic-opioid rate, years 1-3 after it, on log dispensing with the date held", sname, "lrx",
                smf.ols("lr ~ lrx + zalt", data=kk).fit(cov_type="HC3"))
            row(res, "6", f"{base}: log synthetic-opioid rate, years 1-3 after it, triplicate with the date held", sname, "triplicate",
                smf.ols("lr ~ triplicate + zalt", data=kk).fit(cov_type="HC3"))
    # the paper's own date, for reference on the same lines
    for f in ("growth_since_launch ~ triplicate", "growth_since_launch ~ triplicate + za"):
        row(res, "6", f"the paper's half-year date: triplicate gap in overdose growth, 1995-98 to 2017-19; model {f}", "49 states", "triplicate",
            smf.ols(f, data=s).fit(cov_type="HC3"))


def mw(a, b):
    return mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue if len(a) and len(b) else np.nan


def contrasts(m, rank_col, label, res, item, change="narc_change", level="narc_1921", market="anyill"):
    k = m.dropna(subset=[rank_col]).nlargest(15, rank_col)
    w, wo = k[k[market] == 1], k[k[market] == 0]
    row(res, item, f"15 heaviest on {label}: median change in narcotic-poisoning mortality (log ratio), with a market by 2019 against without",
        "; ".join(sorted(k.country)), "Mann-Whitney", b=w[change].median(), lo=wo[change].median(), p=mw(w[change], wo[change]), n=len(k),
        n_with=len(w), note="b is the median with a market, lo the median without")
    cur = k[k.country.isin(CURRENT)]
    rest = k[~k.country.isin(CURRENT)]
    row(res, item, f"15 heaviest on {label}: narcotic-poisoning mortality per 100 000 at the end window, markets present in 2019-21 against the rest",
        "with: " + ", ".join(sorted(cur.country)), "Mann-Whitney", b=cur[level].min(), lo=rest[level].min(), hi=rest[level].max(), p=mw(cur[level], rest[level]),
        n=len(k), n_with=len(cur), note="b is the lowest rate with a market, lo and hi the range without")


def item7(out, res):
    m = pd.read_csv(out / "tests_countries.csv")
    x = pd.read_csv(out / "incb_sddd_2000_2014.csv")
    import re
    x["key"] = [re.sub(r"\s+", " ", re.sub(r"[^a-z ,()']", " ", str(c).lower().replace("’", "'").replace("ü", "u").replace("ç", "c")
                                              .replace("é", "e").replace("ô", "o"))).strip() for c in x.country]
    subs = ["codeine", "fentanyl", "hydrocodone", "hydromorphone", "morphine", "oxycodone", "pethidine", "tilidine", "others"]
    g = x[x.window.isin(PRE)].groupby("key")[subs + ["analgesic"]].mean()
    m = m.merge(g, left_on="key", right_index=True, how="left")
    m["oxy_hydro"] = m.oxycodone + m.hydrocodone
    m["not_fentanyl"] = m.analgesic - m.fentanyl
    m["other"] = m.analgesic - m.oxy_hydro - m.fentanyl - m.morphine
    assert np.allclose(m.analgesic, m.rx_pre, equal_nan=True), "the substance means do not rebuild the paper's exposure"
    top = m.nlargest(15, "rx_pre")
    comp = top[["country", "status", "rx_pre", "oxy_hydro", "fentanyl", "morphine", "other", "narc_1921"]].copy()
    for c in ("oxy_hydro", "fentanyl", "morphine", "other"):
        comp[c + "_share"] = comp[c] / comp.rx_pre
    comp.to_csv(out / "composition_sddd.csv", index=False)
    for c, lab in (("rx_pre", "all opioid analgesics, the paper's ranking"), ("oxy_hydro", "oxycodone plus hydrocodone"), ("morphine", "morphine"),
                   ("not_fentanyl", "all opioid analgesics but fentanyl")):
        contrasts(m, c, lab, res, "7")
    rho, pv = spearmanr(top.fentanyl / top.rx_pre, top.narc_1921)
    row(res, "7", "15 heaviest: Spearman of fentanyl's share of consumption with narcotic-poisoning mortality 2019-21", "15 countries", "rho", b=rho, p=pv, n=15)
    return m, x


def item15(m, x, res):
    for lab, wins in (("2000-11 and 2012-14, five windows", PRE + ["2012-2014"]), ("2012-14 alone", ["2012-2014"]), ("2009-11 and 2012-14", ["2009-2011", "2012-2014"])):
        g = x[x.window.isin(wins)].groupby("key").analgesic.agg(["mean", "count"])
        k = m.assign(rx_alt=m.key.map(g["mean"]), n_alt=m.key.map(g["count"])).dropna(subset=["rx_alt", "narc_change", "lgdp"])
        k["lrx_alt"] = np.log1p(k.rx_alt)
        r = smf.ols("narc_change ~ lrx_alt * anyill + lgdp", data=k).fit(cov_type="HC3")
        row(res, "15", f"interaction of log prescribing with an illicit fentanyl market, prescribing averaged over {lab}", f"{len(k)} countries", "lrx_alt:anyill", r)
        contrasts(k, "rx_alt", f"prescribing in {lab}", res, "15")
        rho, pv = spearmanr(k.rx_alt, k.rx_pre)
        row(res, "15", f"Spearman of prescribing in {lab} with the paper's 2000-11 exposure", f"{len(k)} countries", "rho", b=rho, p=pv, n=len(k))


def item20(m, out, res):
    w = pd.read_csv(out / "who_drug_deaths_country_year.csv")
    w["Year"] = pd.to_numeric(w.Year, errors="coerce")
    name_to_key = dict(zip(m.country, m.key))
    w["key"] = w.country_name.map(name_to_key)
    k = w[w.Year.between(2017, 2019) & w.key.notna()].groupby("key").narcotics_per_100k.agg(["mean", "count"])
    m = m.assign(narc_1719=m.key.map(k["mean"].where(k["count"] >= 2)))
    m["change_1719"] = np.log((m.narc_1719 + 0.05) / (m.narc_1012 + 0.05))
    d = m.dropna(subset=["change_1719", "lrx_pre", "lgdp"])
    row(res, "20", "countries of the 60 with at least two years of deaths in 2017-19", "", "n", n=len(d), b=len(d),
        note="missing: " + ", ".join(sorted(set(m.country) - set(d.country))))
    r = smf.ols("change_1719 ~ lrx_pre * anyill + lgdp", data=d).fit(cov_type="HC3")
    row(res, "20", "interaction of log prescribing with an illicit fentanyl market, change in narcotic-poisoning mortality 2010-12 to 2017-19",
        f"{len(d)} countries", "lrx_pre:anyill", r)
    r = smf.ols("narc_change ~ lrx_pre * anyill + lgdp", data=d).fit(cov_type="HC3")
    row(res, "20", "the paper's interaction, 2010-12 to 2019-21, on the same countries", f"{len(d)} countries", "lrx_pre:anyill", r)
    contrasts(d, "rx_pre", "prescribing, end window 2017-19", res, "20", change="change_1719", level="narc_1719")
    # in 2017-19 Sweden's market was still present for part of the window; the by-2019 coding is the level contrast that fits it
    top = d.nlargest(15, "rx_pre")
    a, b = top[top.anyill == 1], top[top.anyill == 0]
    row(res, "20", "15 heaviest prescribers: narcotic-poisoning mortality per 100 000 in 2017-19, with a market by 2019 against without",
        "with: " + ", ".join(sorted(a.country)), "Mann-Whitney", b=a.narc_1719.min(), lo=b.narc_1719.min(), hi=b.narc_1719.max(), p=mw(a.narc_1719, b.narc_1719),
        n=len(top), n_with=len(a), note="b is the lowest rate with a market, lo and hi the range without")


def main(out, d77, wonder, wcy, onea, arcos, nflis_year, heroin_csv):
    out, wonder, wcy = pathlib.Path(out), pathlib.Path(wonder), pathlib.Path(wcy)
    c33 = load("c33", "33_county_arrival.py")
    res = []
    p = state_rates(d77)
    s = pd.read_csv(out / "alpert_rerun_frame.csv").set_index("st")
    s["Y"] = np.floor(s.arrival).astype(int)
    s = item1_states(s, p, res)
    # the frame must rebuild the paper's main state estimate before anything else is read from it
    chk = smf.ols("np.log(rate13) ~ lrx + za", data=s).fit(cov_type="HC3").params.lrx
    paper = pd.read_csv(out / "fixed_span.csv").query("span == 'years 1-3 after the arrival year' and model == 'dispensing, arrival date' and hidden == 1").b.iloc[0]
    assert np.isclose(chk, paper), (chk, paper)
    cb, iqr = item1_counties(c33, onea, arcos, wonder, res)
    item3b(c33, cb, iqr, wcy, res)
    s = item5(s, p, out, heroin_csv, res)
    item6(s, p, nflis_year, heroin_csv, res)
    m, x = item7(out, res)
    item15(m, x, res)
    item20(m, out, res)
    r = pd.DataFrame(res)
    r.to_csv(out / "further_checks.csv", index=False)
    with pd.option_context("display.width", 320, "display.max_rows", 600, "display.max_colwidth", 150):
        cols = [c for c in ["item", "check", "sample", "term", "bound", "b", "lo", "hi", "p", "n", "diff", "diff_lo", "diff_hi", "ratio", "ratio_lo", "ratio_hi", "mean_outcome"] if c in r]
        print(r[cols].round(3).to_string(index=False))
        print(pd.read_csv(out / "composition_sddd.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 9:
        sys.exit(__doc__)
    main(*sys.argv[1:])
