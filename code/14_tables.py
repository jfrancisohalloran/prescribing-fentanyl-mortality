"""Tables 1 and 2 for the manuscript (paper/TABLES.md), built from the outputs so every cell is traced.

  python3 code/14_tables.py <out folder> <paper folder>
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def ci(b, se):
    return f"{b:.2f} ({b - 1.96 * se:.2f} to {b + 1.96 * se:.2f})"


def main(out, paper):
    out = pathlib.Path(out)
    a = pd.read_csv(out / "synthetic_arrival.csv")
    t = pd.read_csv(out / "tests_countries.csv")
    u = pd.read_csv(out / "unodc_fentanyl_seizures.csv")
    t["rank"] = t.rx_pre.rank(ascending=False).astype(int)
    order = {"established": 0, "episodic": 1, "diverted only": 2, "unclear": 3}
    k = a[a.status.isin(order) | a.transit].copy()
    k["o"] = k.status.map(order).fillna(4)
    k = k.sort_values(["o", "first_seen", "country"])
    rows = ["| Country | Status by 2019 | Lasting market first seen | At scale | Product | UNODC fentanyl-type seizures, g per million a year, 2015–21 | Prescribing rank among 60 | Narcotic-poisoning deaths per 100 000, 2019–21 |",
            "|" + "|".join("-" * w for w in (14, 16, 7, 7, 34, 14, 9, 11)) + "|"]
    for _, r in k.iterrows():
        status = "transit only" if r.transit and r.status == "none known" else r.status
        sz = u.loc[u.country == r.country, "g_per_million_per_year"]
        tt = t[t.country == r.country]
        fmt = lambda v: "" if pd.isna(v) else f"{int(v)}"
        rows.append(f"| {r.country.replace('United States of America', 'USA').replace('China, Hong Kong SAR', 'Hong Kong')} | {status} | {fmt(r.first_seen)} | {fmt(r.at_scale)} | "
                    f"{(r['product'] if isinstance(r['product'], str) else '')} | {'' if sz.empty else f'{sz.iloc[0]:,.0f}'} | "
                    f"{'' if tt.empty else int(tt['rank'].iloc[0])} | {'' if tt.empty else f'{tt.narc_1921.iloc[0]:.2f}'} |")
    t1 = ("**Table 1: Illicit fentanyl markets by country, 2019.** Countries other than these are coded as having no illicit fentanyl market known. "
          "Sources for each code are in the appendix. The first year dates a lasting market; earlier episodes, such as the USA's in 2005–07 "
          "and Sweden's in the 1990s and early 2000s, are noted there. Sweden's market ended in 2018. The alternative coding counts Lithuania as an established market. Seizure rates are calculated from UNODC's World Drug Report statistical annex^17^ and "
          "death rates from the WHO Mortality Database;^11^ prescribing ranks are from INCB's technical reports.^10^ "
          "Transit-only countries count as supply only in a sensitivity analysis. A blank cell "
          "means that no seizure was reported or that the country lacks WHO mortality data.\n\n" + "\n".join(rows))

    r = pd.read_csv(out / "tests_results.csv")
    us = pd.read_csv(out / "us_states_results.csv"); us = us[us.bound == 5]
    al = pd.read_csv(out / "alpert_rerun_results.csv"); al = al[al.bound == 5]
    g = lambda d, m, term: d[(d.model == m) & (d.term == term)].iloc[0]
    q = t.rx_pre.quantile(0.75); hi = t[t.rx_pre >= q]
    w, wo = hi.loc[hi.any_illicit_by_2019, "narc_change"], hi.loc[~hi.any_illicit_by_2019, "narc_change"]
    # Table 2 gives each model in full, in the same words row to row: units, outcome, exposure, what it is
    # adjusted for, the model, and the estimate.
    lines = ["| Units | Outcome | Exposure | Adjusted for | Model | Estimate (95% CI) |",
             "|" + "|".join("-" * w for w in (16, 22, 22, 18, 14, 20)) + "|"]
    row = lambda *cells: lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    REG, CHANGE = "Linear regression", "Change in narcotic-poisoning mortality, 2010–12 to 2019–21"
    lines.append("| **Countries** | | | | | |")
    x = g(r, "P1 | narc_change | all", "lrx_pre"); row(f"{x.n} countries", CHANGE, "Log prescribing, 2000–11", "GDP", REG, ci(x.b, x.se))
    x = g(r, "P2 any illicit | narc_change | all", "lrx_pre:anyill")
    row(f"{x.n} countries", CHANGE, "Log prescribing × illicit fentanyl market, established or episodic", "GDP", REG, ci(x.b, x.se))
    x = g(r, "P2 established | narc_change | all", "lrx_pre:est")
    row(f"{x.n} countries", CHANGE, "Log prescribing × established market", "GDP", REG, ci(x.b, x.se))
    x = g(r, "P2 no illicit supply | narc_change | all", "rho")
    row(f"{x.n} countries without an illicit fentanyl market", CHANGE, "Log prescribing, 2000–11", "None", "Spearman correlation", f"{x.b:.2f} (p={x.p:.2f})")
    heavy = f"{len(w) + len(wo)} heaviest prescribers, {len(w)} with an illicit fentanyl market and {len(wo)} without"
    row(heavy, CHANGE, "Illicit fentanyl market", "None", "Mann–Whitney test",
        f"median {np.expm1(w.median()) * 100:+.0f}% vs {np.expm1(wo.median()) * 100:+.0f}% (p={mannwhitneyu(w, wo).pvalue:.2f})")
    lw, lwo = hi.loc[hi.any_illicit_by_2019, "narc_1921"], hi.loc[~hi.any_illicit_by_2019, "narc_1921"]
    row(heavy, "Narcotic-poisoning mortality per 100 000, 2019–21", "Illicit fentanyl market", "None", "Mann–Whitney test",
        f"{lw.min():.2f}–{lw.max():.2f} vs {lwo.min():.2f}–{lwo.max():.2f} (p={mannwhitneyu(lw, lwo).pvalue:.3f})")
    # markets present in 2019-21 (code/25): among the heaviest prescribers, the USA and Canada
    cur = hi.country.isin(["United States of America", "Canada", "Estonia", "Latvia", "Lithuania"])
    cw, cwo = hi.loc[cur, "narc_1921"], hi.loc[~cur, "narc_1921"]
    row(f"{len(hi)} heaviest prescribers, {len(cw)} with an illicit fentanyl market in 2019–21 and {len(cwo)} without",
        "Narcotic-poisoning mortality per 100 000, 2019–21", "Illicit fentanyl market in 2019–21", "None", "Exact Mann–Whitney test",
        f"{' and '.join(f'{v:.2f}' for v in sorted(cw))} vs {cwo.min():.2f}–{cwo.max():.2f} "
        f"(p={mannwhitneyu(cw, cwo, alternative='two-sided', method='exact').pvalue:.3f})")
    SYN, OD = "Synthetic-opioid mortality, pooled over 2014–20", "Change in overdose mortality, 2011–13 to 2017–19"
    lines.append("| **US states** | | | | | |")
    x = g(us, "P2 + arrival | log synthetic-opioid rate 2014-20", "lrx"); row(f"{x.n} states", SYN, "Log dispensing, 2009–11", "Arrival date", REG, ci(x.b, x.se))
    x = g(us, "P2 + arrival, distress, market | log synthetic-opioid rate 2014-20", "lrx")
    row(f"{x.n} states", SYN, "Log dispensing, 2009–11", "Arrival date, distress, market type", REG, ci(x.b, x.se))
    fx = pd.read_csv(out / "fixed_span.csv"); fx = fx[fx.span == "years 1-3 after the arrival year"]
    span = lambda mod: "; ".join(dict.fromkeys(ci(k.b, k.se) for k in fx[fx.model == mod].sort_values("hidden").itertuples()))
    SPAN = "Synthetic-opioid mortality, three years after arrival"
    row("49 states", SPAN, "Log dispensing, 2009–11", "Arrival date", REG, span("dispensing, arrival date"))
    row("49 states", SPAN, "White-powder heroin market", "None", REG, span("market type"))
    x = g(us, "P2 + arrival | overdose rise 2011-13 to 2017-19", "lrx"); row(f"{x.n} states", OD, "Log dispensing, 2009–11", "Arrival date", REG, ci(x.b, x.se))
    x = g(us, "P2 steepening | after arrival, beyond the calendar change in the gradient", "lrx:post"); assert x.n == 49 * 4, x.n
    row("49 states, 4 windows each", "Overdose mortality", "Log dispensing in windows after arrival",
        "State, window, the dispensing slope in each window, windows after arrival",
        "Panel regression, errors clustered by state", ci(x.b, x.se))
    # The county and provincial tests (code/33 and code/32)
    lci = lambda b, lo, hi: f"{b:.2f} ({lo:.2f} to {hi:.2f})"
    ca = pd.read_csv(out / "county_arrival_results.csv")
    lines.append("| **US counties within states** | | | | | |")
    x = ca[(ca.model == "main") & (ca.bound.astype(str) == "5") & (ca.term == "lrx_x_after")].iloc[0]
    row(f"{int(x.counties)} counties, 5 windows each", "Overdose mortality, 2005–07 to 2017–19", "Log dispensing, 2009–11 × share of the window after the state's arrival",
        "County, state by window, the dispensing slope in each window", "Poisson regression, errors clustered by state", lci(x.b, x.lo, x.hi))
    k = ca[(ca.model == "5a T40.4 county-year: bins of years since arrival, censored Poisson") & ca.term.isin(["lrx_x_since_3-4", "lrx_x_since_5-6"])]
    row(f"{int(k.counties.iloc[0])} counties, 2018–25 by year", "Synthetic-opioid mortality", "Log dispensing, 2009–11 × 3–4 years since the state's arrival; × 5–6 years",
        "County, state by year", "Poisson regression, hidden counts censored between one and nine, errors clustered by state",
        "; ".join(lci(r.b, r.lo, r.hi) for r in k.sort_values("term").itertuples()))
    cp = pd.read_csv(out / "canada_provinces_results.csv")
    x = cp[(cp.variant == "main, with arrival") & (cp.x == "log_presc")].iloc[0]
    lines.append("| **Canadian provinces** | | | | | |")
    row(f"{int(x.n)} provinces", "Change in poisoning mortality, 2010–12 to 2019–21", "Log strong-opioid dispensing, fiscal 2007–08 to 2011–12",
        "Arrival year", REG, lci(x.estimate, x.lo, x.hi))
    lines.append("| **Triplicate states against the other states** | | | | | |")
    for mod, outcome, adjusted in (("growth 1995-98 to 2017-19 | triplicate", "Change in overdose mortality, 1995–98 to 2017–19", "None"),
                                   ("growth 1995-98 to 2017-19 | triplicate, arrival held", "Change in overdose mortality, 1995–98 to 2017–19", "Arrival date"),
                                   ("growth 2011-13 to 2017-19 | triplicate, arrival held", OD, "Arrival date"),
                                   ("log synthetic-opioid rate 2014-20 | triplicate", SYN, "None"),
                                   ("log synthetic-opioid rate 2014-20 | triplicate, arrival held", SYN, "Arrival date")):
        x = g(al, mod, "triplicate"); row(f"{x.n} states", outcome, "Triplicate state", adjusted, REG, ci(x.b, x.se))
    row("49 states", SPAN, "Triplicate state", "Arrival date", REG, span("triplicate, arrival date"))
    t2 = ("**Table 2: Design and main estimates.** Each row is one model. Mortality is on the log scale and changes are log ratios, except "
          "where rates per 100 000 are shown. Arrival date is when illicit fentanyl reached the state. Linear regressions use robust standard "
          "errors. A rate pooled over 2014–20 gives late-arriving states fewer years with fentanyl; the rate over the three calendar years after "
          "each state's arrival year does not. Suppressed US county counts enter at five, except in the censored model; the results at one and nine, and every other specification, are in the appendix.\n\n"
          + "\n".join(lines))
    (pathlib.Path(paper) / "TABLES.md").write_text(t1 + "\n\n" + t2 + "\n")
    print(t1[:1200]); print(); print(t2[:2400])


if __name__ == "__main__":
    main(*sys.argv[1:3])
