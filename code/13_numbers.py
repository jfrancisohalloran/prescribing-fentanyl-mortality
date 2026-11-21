"""Every number the manuscript quotes, with 95% CIs, traced to the output it came from (paper/NUMBER_TRACE.md).

Reads the outputs of code/06, 07, 09, 10 and 11. Estimates are OLS with HC3 errors (or clustered by state for panels); CI = b +/- 1.96 SE.

  python3 code/13_numbers.py <out folder> <paper folder>
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def ci(b, se):
    return f"{b:.2f} (95% CI {b - 1.96 * se:.2f} to {b + 1.96 * se:.2f})"


def main(out, paper):
    out, rows = pathlib.Path(out), []

    def add(key, value, source, note=""):
        rows.append((key, value, source, note))

    t = pd.read_csv(out / "tests_countries.csv")
    r = pd.read_csv(out / "tests_results.csv")
    add("countries in the analysis", len(t), "tests_countries.csv")
    add("established markets in the sample", int(t.established_by_2019.sum()), "tests_countries.csv")
    add("any illicit supply in the sample", int(t.any_illicit_by_2019.sum()), "tests_countries.csv")
    inc = pd.read_csv(out / "incb_sddd_2000_2014.csv")  # Methods: 886 of the 887 country figures add up; China 2006-08 is off by rounding
    ok = inc.adds_up.astype(bool)
    add("INCB country figures extracted (country by window, five editions)", len(inc), "incb_sddd_2000_2014.csv")
    add("INCB country figures whose substances sum to the printed total", int(ok.sum()), "incb_sddd_2000_2014.csv")
    add("INCB country figure that misses", "; ".join(f"{r.country} {r.window}" for r in inc[~ok].itertuples()), "incb_sddd_2000_2014.csv",
        "a rounding gap (appendix A1)")

    def est(model, term):
        k = r[(r.model == model) & (r.term == term)].iloc[0]
        return k

    k = est("P1 | narc_change | all", "rho"); add("P1 Spearman, narcotics change", f"{k.b:.2f} (p = {k.p:.2f})", "tests_results.csv", k.model)
    k = est("P1 | drug_change | all", "rho"); add("P1 Spearman, all drug poisoning change", f"{k.b:.2f} (p = {k.p:.2f})", "tests_results.csv", k.model)
    k = est("P1 | narc_change | all", "lrx_pre"); add("P1 OLS with GDP, narcotics", ci(k.b, k.se) + f", p = {k.p:.2f}", "tests_results.csv", k.model)
    k = est("P2 any illicit | narc_change | all", "lrx_pre:anyill"); add("P2 interaction, any illicit, narcotics", ci(k.b, k.se) + f", p = {k.p:.3f}", "tests_results.csv", k.model)
    k = est("P2 established | narc_change | all", "lrx_pre:est"); add("P2 interaction, established, narcotics", ci(k.b, k.se) + f", p = {k.p:.2f}", "tests_results.csv", k.model)
    k = est("P2 no illicit supply | narc_change | all", "rho"); add("P2 Spearman without illicit supply, narcotics", f"{k.b:.2f} (p = {k.p:.2f}, n = {k.n})", "tests_results.csv", k.model)
    k = est("P4 | narc_change | all", "loral_pre"); add("P4 oral, narcotics", ci(k.b, k.se) + f", p = {k.p:.2f}", "tests_results.csv", k.model)
    k = est("P4 | narc_change | all", "lpatch_pre"); add("P4 patch, narcotics", ci(k.b, k.se) + f", p = {k.p:.2f}", "tests_results.csv", k.model)

    q = t.rx_pre.quantile(0.75)
    hi = t[t.rx_pre >= q]
    a, b = hi.loc[hi.any_illicit_by_2019, "narc_change"], hi.loc[~hi.any_illicit_by_2019, "narc_change"]
    mw = mannwhitneyu(a, b)
    add("P3 top-quarter threshold (S-DDD)", f"{q:,.0f}", "tests_countries.csv")
    add("P3 median narcotics change, with supply", f"{a.median():.2f} (x{np.exp(a.median()):.2f}, n = {len(a)})", "tests_countries.csv")
    add("P3 median narcotics change, without supply", f"{b.median():.2f} (x{np.exp(b.median()):.2f}, n = {len(b)})", "tests_countries.csv")
    add("P3 Mann-Whitney, with against without", f"p = {mw.pvalue:.3f}", "tests_countries.csv")
    lw, lwo = hi.loc[hi.any_illicit_by_2019, "narc_1921"], hi.loc[~hi.any_illicit_by_2019, "narc_1921"]
    add("P3 levels 2019-21, with supply (range)", f"{lw.min():.2f} to {lw.max():.2f}", "tests_countries.csv")
    add("P3 levels Mann-Whitney, with against without (added after the change test)", f"p = {mannwhitneyu(lw, lwo).pvalue:.3f}", "tests_countries.csv")
    add("P3 heavy prescribers without supply, 2019-21 narcotics rate range", f"{hi.loc[~hi.any_illicit_by_2019, 'narc_1921'].min():.2f} to {hi.loc[~hi.any_illicit_by_2019, 'narc_1921'].max():.2f}", "tests_countries.csv")
    t["rank"] = t.rx_pre.rank(ascending=False).astype(int)
    add("prescribing ranks of established markets", ", ".join(f"{c} {n}" for c, n in t.loc[t.established_by_2019, ["country", "rank"]].sort_values("rank").values), "tests_countries.csv")
    s = t[t.in_review_scope]
    add("established vs not, prescribing, Mann-Whitney (review scope)", f"p = {mannwhitneyu(s.loc[s.established_by_2019, 'rx_pre'], s.loc[~s.established_by_2019, 'rx_pre']).pvalue:.2f}", "tests_countries.csv")
    us_ = t[t.country == "United States of America"].iloc[0]; ca = t[t.country == "Canada"].iloc[0]
    add("USA narcotics rate 2010-12 to 2019-21", f"{us_.narc_1012:.1f} to {us_.narc_1921:.1f}", "tests_countries.csv")
    add("Canada narcotics rate 2010-12 to 2019-21", f"{ca.narc_1012:.1f} to {ca.narc_1921:.1f}", "tests_countries.csv")
    add("USA all drug poisoning 2019-21", f"{us_.drug_1921:.1f}", "tests_countries.csv")
    est_ = t[t.country == "Estonia"].iloc[0]
    add("Estonia narcotics rate 2010-12 to 2019-21", f"{est_.narc_1012:.1f} to {est_.narc_1921:.1f}", "tests_countries.csv")
    for c in ("Estonia", "Latvia", "Germany", "Belgium", "Austria", "Denmark", "Switzerland"):  # Discussion: 4-5% and 64-97% of Canada's
        add(f"prescribing 2000-11 as a share of Canada's, {c}", f"{t.loc[t.country == c, 'rx_pre'].iloc[0] / ca.rx_pre:.0%}", "tests_countries.csv")

    aa = pd.read_csv(out / "tests_around_arrival.csv")
    for _, x in aa.iterrows():
        add(f"around arrival, {x.country}", f"{x.narc_before:.2f} to {x.narc_after:.2f} (x{np.exp(x.narc_change):.1f})", "tests_around_arrival.csv")

    us = pd.read_csv(out / "us_states_results.csv")
    us = us[us.bound == 5]
    fr = pd.read_csv(out / "us_states_frame.csv")

    def uest(model, term):
        return us[(us.model == model) & (us.term == term)].iloc[0]

    add("US states", len(fr), "us_states_frame.csv")
    add("US prescribing range, per 100", f"{fr.rx.min():.0f} to {fr.rx.max():.0f}", "us_states_frame.csv")
    add("US arrival range", f"{fr.arrival.min()} to {fr.arrival.max()}", "us_states_frame.csv")
    # Discussion, 2026-11-10: US mortality after China's class-wide control of 1 May 2019, and the states' overdose rate beside the
    # cardiac-coded poisonings (below 1 per 100 000) that underlying-cause counts leave out
    who = pd.read_csv(out / "who_drug_deaths_country_year.csv")
    usa = who[who.country_name == "United States of America"].set_index("Year").narcotics_per_100k
    add("US narcotic-poisoning mortality per 100 000, 2019 and 2021", f"{usa[2019]:.1f} and {usa[2021]:.1f}", "who_drug_deaths_country_year.csv")
    add("US states, median overdose rate per 100 000, 2017-19", f"{fr['od_2017-19'].median():.0f}", "us_states_frame.csv")
    tc = pd.read_csv(out / "tests_countries.csv")  # Discussion: all 11 of the heaviest prescribers without supply are in Europe or Australia
    top = tc[tc.rx_pre >= tc.rx_pre.quantile(0.75)]
    without = sorted(top.loc[~top.any_illicit_by_2019.astype(bool), "country"])
    add(f"Heaviest-prescribing quarter without illicit supply ({len(without)})", "; ".join(without), "tests_countries.csv")
    from scipy.stats import spearmanr
    add("US prescribing against arrival, Spearman", f"{spearmanr(fr.rx, fr.arrival)[0]:.2f} (p = {spearmanr(fr.rx, fr.arrival)[1]:.2f})", "us_states_frame.csv")
    k = uest("P2 + arrival | log synthetic-opioid rate 2014-20", "lrx"); add("US wave, prescribing with arrival held", ci(k.b, k.se) + f", p = {k.p:.2f}; R2 {k.r2:.2f}", "us_states_results.csv", k.model)
    k = uest("P2 + arrival, distress, market | log synthetic-opioid rate 2014-20", "lrx"); add("US wave, prescribing with arrival, distress, market", ci(k.b, k.se) + f", p = {k.p:.2f}", "us_states_results.csv", k.model)
    k = uest("P2 + arrival | overdose rise 2011-13 to 2017-19", "lrx"); add("US rise, prescribing with arrival held", ci(k.b, k.se) + f", p = {k.p:.2f}", "us_states_results.csv", k.model)
    for w in ("2008-10", "2011-13", "2014-16", "2017-19"):
        k = uest(f"P2 gradient across states | overdose rate {w}", "lrx"); add(f"US gradient {w}", ci(k.b, k.se) + f", p = {k.p:.3f}", "us_states_results.csv", k.model)
    k = uest("P2 steepening | after arrival, beyond the calendar change in the gradient", "lrx:post"); add("US gradient change at arrival, calendar held", ci(k.b, k.se) + f", p = {k.p:.2f}", "us_states_results.csv", k.model)
    k = uest("P4 oral oxycodone and hydrocodone against fentanyl (ARCOS grams) | log synthetic-opioid rate 2014-20", "loral_g"); add("US P4 oral, wave", ci(k.b, k.se) + f", p = {k.p:.2f}", "us_states_results.csv", k.model)
    k = uest("P4 oral oxycodone and hydrocodone against fentanyl (ARCOS grams) | log synthetic-opioid rate 2014-20", "lfent_g"); add("US P4 fentanyl, wave", ci(k.b, k.se) + f", p = {k.p:.2f}", "us_states_results.csv", k.model)
    import statsmodels.formula.api as smf
    add("US arrival alone, R2 for the wave", f"{smf.ols('lt404 ~ za', data=fr).fit().rsquared:.2f}", "us_states_frame.csv")
    # Limitations, 2026-11-10: distress measured in 2008-12 (code/19) and drug deaths coded under cardiovascular causes by state (code/20)
    hc = pd.read_csv(out / "hardship_2010_check.csv")
    for d in ("2008-2012 index", "both"):
        k = hc[(hc.bound == 5) & (hc.outcome == "log synthetic-opioid rate 2014-20") & (hc.model == "arrival, distress, market")
               & (hc.distress == d) & (hc.term == "lrx")].iloc[0]
        add(f"US wave, prescribing with arrival, distress ({d}), market", f"{k.b:.2f} (95% CI {k.lo:.2f} to {k.hi:.2f}), p = {k.p:.2f}",
            "hardship_2010_check.csv", k.formula)
    cs = pd.read_csv(out / "cardiac_coding_shares.csv")
    a, s = cs[cs.deaths.str.startswith("all")].iloc[0], cs[cs.deaths.str.startswith("synthetic")].iloc[0]
    add("Cardiovascular-coded share of drug deaths by state, 1999-2020", f"{100 * a.share_min:.1f}% to {100 * a.share_max:.1f}% ({int(a.states_listed)} states)",
        "cardiac_coding_shares.csv")
    add("Cardiovascular-coded share of synthetic-opioid deaths, 2014-20", f"{100 * s.share_listed_states:.1f}% in the {int(s.states_listed)} states listed",
        "cardiac_coding_shares.csv", "the others have fewer than 10 such deaths")
    rho = cs[cs.deaths.str.startswith("all") & (cs.against == "prescribing 2009-11")].iloc[0]
    add("Cardiovascular-coded share against prescribing, Spearman", f"{rho.spearman:.2f} (p = {rho.p:.2f})", "cardiac_coding_shares.csv")
    ab = pd.read_csv(out / "cardiac_coding_addback.csv")
    ab = ab[(ab.outcome == "with cardiovascular-coded deaths") & (ab.formula == "lt404_cv ~ lrx + za")]
    add("US wave, prescribing with arrival held, cardiovascular-coded deaths added back",
        "; ".join(f"{r.b:.2f} (95% CI {r.lo:.2f} to {r.hi:.2f}) with left-out states at {r.left_out_states_at}" for r in ab.itertuples()),
        "cardiac_coding_addback.csv")

    al = pd.read_csv(out / "alpert_rerun_results.csv")
    al = al[al.bound == 5]

    def aest(model, term):
        return al[(al.model == model) & (al.term == term)].iloc[0]

    for term, lab in (("trip_1990", "1990-94"), ("trip_2002", "2002-04"), ("trip_2011", "2011-13"), ("trip_2014", "2014-16"), ("trip_2017", "2017-19")):
        k = aest("gap by window, relative to 1995-98", term); add(f"triplicate gap {lab}", ci(k.b, k.se), "alpert_rerun_results.csv", k.model)
        k = aest("gap by window, arrival held", term); add(f"triplicate gap {lab}, arrival held", ci(k.b, k.se), "alpert_rerun_results.csv", k.model)
    for model in ("growth 1995-98 to 2017-19 | triplicate", "growth 1995-98 to 2017-19 | triplicate, arrival held",
                  "growth 2011-13 to 2017-19 | triplicate, arrival held", "log synthetic-opioid rate 2014-20 | triplicate",
                  "log synthetic-opioid rate 2014-20 | triplicate, arrival held", "growth 1995-98 to 2011-13 (before fentanyl) | triplicate"):
        k = aest(model, "triplicate"); add(model, ci(k.b, k.se) + f", p = {k.p:.3f}", "alpert_rerun_results.csv")

    e = pd.read_csv(out / "euda_drd_check.csv")
    add("EUDA DRD-33 countries", e.country.nunique(), "euda_drd_check.csv")
    u = pd.read_csv(out / "unodc_fentanyl_seizures.csv")
    add("UNODC countries reporting fentanyl-type seizures 2015-21", len(u), "unodc_fentanyl_seizures.csv")
    for c in ("United States of America", "Canada", "Estonia", "Latvia", "Mexico", "China, Hong Kong SAR"):
        x = u[u.Country == c]
        if len(x):
            add(f"UNODC grams per million per year, {c}", f"{x.g_per_million_per_year.iloc[0]:,.0f}", "unodc_fentanyl_seizures.csv")
    add("UNODC median by coding", "; ".join(f"{k}: {v:,.1f}" for k, v in u.groupby("status").g_per_million_per_year.median().items()), "unodc_fentanyl_seizures.csv")
    ac = pd.read_csv(out / "audit_checks.csv").set_index(["check", "model"])  # further checks (code/23)
    for chk, mod, lab in (("country interaction", "without the USA", "Interaction of prescribing and illicit supply, without the USA"),
                          ("triplicate growth_since_launch", "western region", "Triplicate growth gap, western region held"),
                          ("triplicate growth_since_launch", "arrival date and western region", "Triplicate growth gap, arrival and western region held")):
        k = ac.loc[(chk, mod)]
        add(lab, ci(k.b, k.se) + f", n = {int(k.n)}", "audit_checks.csv", "appendix A7 or A9")
    k = r[(r.model == "P1 | narc_change | review scope") & (r.term == "lrx_pre")].iloc[0]
    add("Change on log prescribing, GDP held, countries within the reviews' scope", ci(k.b, k.se) + f", n = {int(k.n)}", "tests_results.csv")
    us = pd.read_csv(out / "us_states_frame.csv")
    ratio = us.rx.quantile(0.75) / us.rx.quantile(0.25)
    add("US dispensing, 75th over 25th percentile, and the wave's ratio at the interval's upper bound, 0.89",
        f"{ratio:.2f}; {ratio ** 0.89:.2f}", "us_states_frame.csv")
    ta = pd.read_csv(out / "alpert_rerun_frame.csv")
    rise = ta["od_2017-19"] - ta["od_2011-13"]
    add("Overdose mortality rise, 2011-13 to 2017-19, per 100 000: triplicate median, other states' median",
        f"{rise[ta.triplicate == 1].median():.1f}, {rise[ta.triplicate == 0].median():.1f}", "alpert_rerun_frame.csv")
    fx = pd.read_csv(out / "fixed_span.csv")  # the same span after each state's arrival (code/24), hidden counts at 1 and 9
    f3 = fx[fx.span == "years 1-3 after the arrival year"].set_index(["model", "hidden"])
    add("Arrival date's share of variation in the pooled 2014-20 rate; in the rate over years 1-3 after arrival",
        f"{100 * fx[(fx.span == '2014-20 pooled') & (fx.model == 'arrival date')].r2.min():.0f}%; "
        f"{100 * f3.loc['arrival date'].r2.min():.0f}% to {100 * f3.loc['arrival date'].r2.max():.0f}%", "fixed_span.csv", "appendix A8")
    for mod, lab in (("dispensing, arrival date", "Prescribing with arrival held, years 1-3 after arrival"), ("market type", "White-powder market, years 1-3 after arrival"),
                     ("triplicate, arrival date", "Triplicate with arrival held, years 1-3 after arrival")):
        add(lab + ", hidden counts at 9; at 1", "; ".join(ci(k.b, k.se) for k in f3.loc[mod].sort_index(ascending=False).itertuples()), "fixed_span.csv", "appendix A8")
    fc = pd.read_csv(out / "fixed_span_certain.csv")
    add("Arrival dates: median standard deviation in years; states above one year", f"{fc.median_sd.iloc[0]:.2f}; {int(fc.states_sd_above_1.iloc[0])} ({fc.left_out.iloc[0]})",
        "fixed_span_certain.csv", "appendix A8")
    k = fc[(fc.span == "years 1-3 after the arrival year") & (fc.model == "dispensing, arrival date")].sort_values("hidden", ascending=False)
    add("Prescribing with arrival held, years 1-3 after arrival, without those states, hidden counts at 9; at 1",
        "; ".join(ci(x.b, x.se) for x in k.itertuples()), "fixed_span_certain.csv", "appendix A8")
    wd = pd.read_csv(out / "who_drug_deaths_country_year.csv")
    wu = wd[wd.country_name == "United States of America"].set_index("Year")
    add("US all drug-poisoning mortality per 100 000, 2019 and 2021", f"{wu.drug_poisoning_per_100k[2019]:.1f}, {wu.drug_poisoning_per_100k[2021]:.1f}",
        "who_drug_deaths_country_year.csv")
    eu = pd.read_csv(out / "europe_after_2019.csv").pivot(index="country", columns="year", values="narcotics_per_100k")
    for c in ("Estonia", "Sweden"):  # the Discussion: mortality fell when local supply was disrupted in 2017-18, before China's control
        add(f"Narcotic-poisoning mortality per 100 000, {c}: 2016-17, 2018, 2020-21",
            f"{eu.loc[c, [2016, 2017]].mean():.1f}, {eu.loc[c, 2018]:.1f}, {eu.loc[c, [2020, 2021]].mean():.1f}", "europe_after_2019.csv",
            "WHO; appendix A4")

    gm = pd.read_csv(out / "gbd_check_countries.csv")
    gr = pd.read_csv(out / "gbd_check_results.csv")
    add("GBD countries with INCB prescribing", len(gm), "gbd_check_countries.csv")
    gm["rank"] = gm.rx_pre.rank(ascending=False).astype(int)
    add("GBD prescribing ranks of established markets", ", ".join(f"{c} {n}" for c, n in gm.loc[gm.established_by_2019, ["country", "rank"]].sort_values("rank").values), "gbd_check_countries.csv")
    for c in ("United States of America", "Canada"):
        add(f"GBD opioid-use-disorder death rate 2019-21, {c}", f"{gm.loc[gm.country == c, 'oud_1921'].iloc[0]:.1f}", "gbd_check_countries.csv")
    wo = gm[~gm.any_illicit_by_2019.astype(bool)]
    top = wo.sort_values("oud_1921", ascending=False).iloc[0]
    add("GBD highest rate without illicit supply, 2019-21", f"{top.country} {top.oud_1921:.2f}", "gbd_check_countries.csv")
    for oc, term, lab in (("np.log(oud_1921)", "lrx", "GBD level on log prescribing, GDP held"), ("change", "lrx:anyill", "GBD change, interaction with illicit supply"),
                          ("change", "lrx", "GBD change on log prescribing, GDP held")):
        k = gr[(gr.outcome == oc) & (gr.term == term)].iloc[0]
        add(lab, ci(k.b, k.se) + f", p = {k.p:.3f}, n = {int(k.n)}", "gbd_check_results.csv")
    k = gr[(gr.outcome == "change") & gr.term.str.startswith("top quarter")].iloc[0]
    add("GBD top quarter with vs without supply, change, Mann-Whitney", f"p = {k.p:.2f}", "gbd_check_results.csv")
    k = gr[(gr.outcome == "level 2019-21") & gr.term.str.startswith("top quarter")].iloc[0]
    add("GBD top quarter with vs without supply, levels 2019-21, Mann-Whitney", f"p = {k.p:.3f}, n = {int(k.n)}", "gbd_check_results.csv")

    # Further checks: code/25 to code/29
    eci = lambda b, lo, hi: f"{b:.2f} (95% CI {lo:.2f} to {hi:.2f})"
    ow = pd.read_csv(out / "outcome_window_results.csv")
    for x in ow.itertuples():
        if x.model.startswith("interaction"):
            add(f"{x.source} interaction, {x.outcome}, coding {x.coding}", eci(x.estimate, x.lo, x.hi) + f", p = {x.p:.3f}", "outcome_window_results.csv", "code/25")
        else:
            add(f"{x.source} {x.model}, {x.outcome}, coding {x.coding}",
                f"p = {x.p:.3f}; medians {x.median_with:.2f} with, {x.median_without:.2f} without; ranks with {x.ranks_with}", "outcome_window_results.csv", "code/25")
    fc = pd.read_csv(out / "fcodes_results.csv")
    for x in fc.itertuples():
        v = eci(x.estimate, x.lo, x.hi) + f", p = {x.p:.3f}" if x.model.startswith("interaction") else f"p = {x.p:.3f}; highest without {x.max_without:.2f}"
        add(f"F codes: {x.outcome}, {x.model}, n = {x.n}", v, "fcodes_results.csv", "code/29; appendix A7")
    nw = pd.read_csv(out / "norway_national.csv").set_index("country")
    add("Norway, national drug-induced deaths per 100 000, all ages, 2010-12 and 2019-21", f"{nw.rate_2010_12['Norway']:.1f}, {nw.rate_2019_21['Norway']:.1f}",
        "norway_national.csv", "code/28; appendix A5")
    sc = pd.read_csv(out / "fixed_span_scaled.csv")
    k = sc[(sc.term == "lrx") & sc.scaled_by_national_rate & (sc.model == "dispensing, arrival date")]
    add("Prescribing with arrival held, scaled by the national rate, by span and bound", "; ".join(
        f"{x.span} at {x.hidden}: {eci(x.b, x.lo, x.hi)}" for x in k.itertuples()), "fixed_span_scaled.csv", "code/26; appendix A8")
    tt = pd.read_csv(out / "triplicate_trend.csv")
    k = tt[tt.term == "trip_trend"]
    add("Triplicate pre-launch trend per year, bounds 1, 5, 9", "; ".join(eci(x.b, x.lo, x.hi) for x in k.itertuples()), "triplicate_trend.csv", "code/27; appendix A9")
    k = tt[tt.term.isin(["trip_2002", "trip_2014"])]
    add("Triplicate gaps against the extended trend, 2002-04 and 2014-16, bounds 1, 5, 9", "; ".join(
        f"{x.term[5:]} at {x.bound}: {eci(x.b, x.lo, x.hi)}" for x in k.itertuples()), "triplicate_trend.csv", "code/27; appendix A9")

    # The county, negative-control, treatment and provincial tests: code/30 to code/35
    ca = pd.read_csv(out / "county_arrival_results.csv")
    cs = pd.read_csv(out / "county_arrival_sample.csv")
    k = ca[(ca.model == "main") & (ca.term == "lrx_x_after")]
    add("County test: log dispensing x after-share, by bound 1, 5, 9", "; ".join(f"{x.bound}: {eci(x.b, x.lo, x.hi)}" for x in k.itertuples()),
        "county_arrival_results.csv", "code/33; appendix A10")
    add("County test: mortality ratio, 75th against 25th percentile of dispensing, by bound 1, 5, 9",
        "; ".join(f"{x.bound}: {x.ratio:.2f} ({x.ratio_lo:.2f} to {x.ratio_hi:.2f})" for x in k.itertuples()), "county_arrival_results.csv", "code/33; appendix A10")
    k = ca[(ca.model == "main") & (ca.term == "lrx_x_after")].iloc[0]
    add("County test: counties, county-windows, states", f"{int(k.counties)}, {int(k.obs)}, {int(k.states)}", "county_arrival_results.csv", "code/33")
    k = ca[ca.term.str.startswith("Wald test")]
    add("County test: pre-arrival bins jointly zero, p, by bound and censored", ", ".join(f"{x.p:.2f}" for x in k.itertuples()),
        "county_arrival_results.csv", "code/33; appendix A10")
    k = ca[(ca.model == "2 rival: economic distress") & (ca.term == "lrx_x_after")]
    add("County test: dispensing ratio with distress as rival, by bound", "; ".join(f"{x.bound}: {x.ratio:.2f} ({x.ratio_lo:.2f} to {x.ratio_hi:.2f})" for x in k.itertuples()),
        "county_arrival_results.csv", "code/33; appendix A10")
    k = ca[ca.model.str.startswith("1 event time (reference e=-1)") & ca.term.str.startswith("lrx_x_e")]
    add("County test: event-time ratios, e<=-3, e=-2, e=0, e=+1, by bound and censored", "; ".join(
        f"{x.bound} {x.term[7:]}: {x.ratio:.2f} ({x.ratio_lo:.2f} to {x.ratio_hi:.2f})" for x in k.itertuples()), "county_arrival_results.csv", "code/33; appendix A10")
    k = ca[ca.model.str.startswith("5a T40.4 county-year: bins of years since arrival") & ca.term.str.startswith("lrx_x_since")]
    add("County T40.4 2018-25: ratios by years since arrival, by bound and censored", "; ".join(
        f"{x.bound} {x.term[12:]}: {x.ratio:.2f} ({x.ratio_lo:.2f} to {x.ratio_hi:.2f})" for x in k.itertuples()), "county_arrival_results.csv", "code/33; appendix A10")
    add("County T40.4 2018-25: counties", int(k.counties.iloc[0]), "county_arrival_results.csv", "code/33")
    q = pd.read_csv(out / "county_arrival_checks.csv")
    add("County test: dispensing at the 25th and 75th percentiles, per 100", "; ".join(f"{r.iloc[0]}: {r.iloc[1]}" for _, r in q.head(4).iterrows()),
        "county_arrival_checks.csv", "code/33")

    nc = pd.read_csv(out / "negative_controls_results.csv")
    nc = nc[(nc["sample"] == "main 60") & (nc.variant == "code/02's missing years")]
    for x in nc[nc.model.str.startswith("interaction")].itertuples():
        add(f"Negative control: {x.outcome}, interaction, coding {x.coding}, n = {x.n}", eci(x.estimate, x.lo, x.hi) + f", p = {x.p:.3f}",
            "negative_controls_results.csv", "code/30; appendix A11")
    for x in nc[nc.model.str.startswith("Mann-Whitney")].itertuples():
        add(f"Negative control: {x.outcome}, {x.model}, coding {x.coding}, n = {x.n}", f"p = {x.p:.3f}", "negative_controls_results.csv", "code/30; appendix A11")
    ei = pd.read_csv(out / "external_by_intent.csv")
    for x in ei.itertuples():
        add(f"By intent: {x.outcome}, interaction, {x.coding}, n = {x.n}", eci(x.b, x.lo, x.hi) + f", p = {x.p:.3f}", "external_by_intent.csv", "code/34; appendix A11")
    en = pd.read_csv(out / "external_net.csv")
    for x in en.itertuples():
        add(f"Narcotic interaction, {x.how}: {x.control}", eci(x.b, x.lo, x.hi) + f", p = {x.p:.3f}", "external_net.csv", "code/34; appendix A11")

    pr = pd.read_csv(out / "population_at_risk_results.csv")
    for x in pr[(pr.test == "S1") & (pr.term == "lrx")].itertuples():
        add(f"Population at risk S1: {x.outcome}, {x.sample}", eci(x.b, x.lo, x.hi) + f", p = {x.p:.3f}, n = {x.n}", "population_at_risk_results.csv", "code/31; appendix A12")
    k = pr[(pr.test == "S3") & (pr["sample"] == "49 states") & (pr.formula == "lod ~ C(st) + C(window) + post + lrx:post + ladm:post")]
    for t in ("ladm:post", "lrx:post"):
        add(f"Population at risk S3: {t} with both terms, by bound", "; ".join(f"{int(x.bound)}: {eci(x.b, x.lo, x.hi)}" for x in k[k.term == t].itertuples()),
            "population_at_risk_results.csv", "code/31; appendix A12")
    for x in pr[(pr.test == "C1") & (pr.term == "lrx_pre") & (pr["sample"] == "European countries with data") & ~pr.outcome.str.contains("any year")].itertuples():
        add(f"Population at risk C1: {x.outcome}", eci(x.b, x.lo, x.hi) + f", p = {x.p:.3f}, n = {x.n}", "population_at_risk_results.csv", "code/31; appendix A12")

    cp = pd.read_csv(out / "canada_provinces_results.csv")
    for x in cp[cp.method.str.startswith("OLS") & (cp.variant.isin(["main, prescribing alone", "main, with arrival", "arrival alone"]))].itertuples():
        add(f"Canada: {x.variant}, {x.x}", eci(x.estimate, x.lo, x.hi) + f", p = {x.p:.3f}, n = {x.n}", "canada_provinces_results.csv", "code/32; appendix A13")
    for x in cp[(cp.method == "Spearman") & (cp.variant == "main")].itertuples():
        add(f"Canada: Spearman, {x.x} with {x.y}", f"{x.estimate:.2f} ({x.lo:.2f} to {x.hi:.2f}), p = {x.p:.3f}", "canada_provinces_results.csv", "code/32; appendix A13")
    ct = pd.read_csv(out / "canada_provinces_table.csv")
    add("Canada: provinces in the test", int(ct.presc_2007_2011.notna().sum()), "canada_provinces_table.csv", "code/32")
    q = ct.dropna(subset=["presc_2007_2011"]).set_index("province")
    add("Canada: Quebec's and Ontario's strong-opioid DDD per 1000 a day, fiscal 2007-08 to 2011-12", f"{q.presc_2007_2011['QC']:.2f}, {q.presc_2007_2011['ON']:.2f}",
        "canada_provinces_table.csv", "code/32; appendix A13")
    add("Canada: Nova Scotia's poisoning rate 2010-12 and 2019-21, arrival", f"{q.all_2010_2012['NS']:.2f}, {q.all_2019_2021['NS']:.2f}, {int(q.arrival['NS'])}",
        "canada_provinces_table.csv", "code/32; appendix A13")

    lo1 = pd.read_csv(out / "state_leave_one_out.csv")
    for b in (1, 5, 9):
        x = lo1[(lo1.bound == b) & (lo1.left_out == "DC")].iloc[0]
        add(f"State model without DC, bound {b}", eci(x.b, x.lo, x.hi) + f", p = {x.p:.3f}; Cook's distance {x.cooks_distance:.2f}",
            "state_leave_one_out.csv", "code/35; appendix A8")
    x = lo1[(lo1.bound == 5) & (lo1.left_out == "DC")].iloc[0]
    add("DC: dispensing per 100, 2009-11", f"{x.dispensing:.1f}", "state_leave_one_out.csv", "code/35")

    il = pd.read_csv(out / "indiana_lab_results.csv").query("sample == 'all' and exposure == 'E1'").set_index("outcome")
    add("Indiana laboratory cases, per SD of log dispensing: fentanyl's share, first-year hazard, fentanyl cases, methamphetamine cases",
        "; ".join(f"{il.loc[k, 'ratio']:.2f} ({il.loc[k, 'ratio_lo']:.2f} to {il.loc[k, 'ratio_hi']:.2f})" for k in
                  ("O1 share", "O2 timing", "O4 fentanyl 2017-20", "other: methamphetamine 2017-20")), "indiana_lab_results.csv", "code/38; appendix A10")
    add("Indiana laboratory cases: counties with a dispensing rate", f"{int(il.loc['O2 timing', 'n'])}", "indiana_lab_results.csv", "code/38; appendix A10")

    lines = ["# Number trace: every number the manuscript quotes", "", "Generated by `code/13_numbers.py`; rerun it after any change upstream.", "",
             "| Key | Value | Source (out/) | Model |", "|---|---|---|---|"]
    lines += [f"| {k} | {v} | `{s}` | {n} |" for k, v, s, n in rows]
    (pathlib.Path(paper) / "NUMBER_TRACE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(f"{k}: {v}" for k, v, _, _ in rows))


if __name__ == "__main__":
    main(*sys.argv[1:3])
