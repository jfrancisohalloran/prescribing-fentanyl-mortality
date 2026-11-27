"""Tables for the appendix from the checks added after review (code/39, 40 and 41), in plain words. code/16 calls `sections(out)` and
places each block at the end of the appendix section it belongs to. No number is retyped: every cell comes from an output file.

Outputs read: further_checks.csv, scotland_who.csv, composition_mme.csv, composition_contrasts.csv, heroin_rx_deaths.csv.
"""
import math
import pathlib

import pandas as pd


def md(df):
    d = df.fillna("").astype(str)
    widths = [min(40, max(4, len(col), *(len(v) for v in d[col].head(200)))) for col in d.columns]
    head = "| " + " | ".join(d.columns) + " |\n|" + "|".join("-" * w for w in widths) + "|\n"
    return head + "\n".join("| " + " | ".join(v.replace("|", "/") for v in row) + " |" for row in d.values) + "\n"


def ci(b, lo, hi, nd=2):
    return f"{b:.{nd}f} ({lo:.{nd}f} to {hi:.{nd}f})".replace("-", "–")


def pv(p):
    return "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def pct(v):
    return f"{100 * (math.exp(v) - 1):+.0f}%".replace("-", "–")


def one(k, what):
    assert len(k) == 1, f"{what}: {len(k)} rows"
    return k.iloc[0]


def scotland(out):
    t = pd.read_csv(out / "scotland_who.csv")
    t = t[~((t.place == "Scotland") & (t.window == "2019-21"))]
    order = ["Scotland", "England and Wales", "United Kingdom", "USA", "Canada"]
    t = t.assign(o=t.place.map(order.index)).sort_values(["window", "o"])
    years = t.window.str.replace("-", "–")
    tab = pd.DataFrame({"Place": t.place, "Years": years, "Narcotic poisoning per 100 000": t.narcotics_per_100k.map("{:.1f}".format),
                        "All drug poisoning per 100 000": t.drug_poisoning_per_100k.map("{:.1f}".format),
                        "Coded to narcotics": t.narcotic_share.map("{:.0%}".format), "Coded to unspecified drugs": t.unspecified_share.map("{:.0%}".format)})
    e = t[t.window == "2019-20"].set_index("place")
    s, u = e.loc["Scotland"], e.loc["USA"]
    u21 = t[(t.window == "2019-21") & (t.place == "USA")].iloc[0]
    return ["The WHO file lists Scotland separately, and the table sets it beside the USA, Canada and the rest of the UK on this paper's "
            "two definitions. A death that names several drugs is coded to an unspecified-drug category in some countries and to the "
            f"narcotic category in others: the narcotic codes hold {s.narcotic_share:.0%} of Scotland's drug poisonings and "
            f"{u.narcotic_share:.0%} of the USA's, where the unspecified codes hold {u.unspecified_share:.0%}. On the narcotic codes "
            f"Scotland's rate in 2019–20, {s.narcotics_per_100k:.1f} per 100 000, is above the USA's {u.narcotics_per_100k:.1f}; on all "
            f"drug poisoning the two are close, {s.drug_poisoning_per_100k:.1f} against {u.drug_poisoning_per_100k:.1f}, and the USA's "
            f"rate over 2019–21 was {u21.drug_poisoning_per_100k:.1f}. The file holds Scotland to 2020.", "", md(tab), ""]


def countries(out, r):
    k15, k20 = r[r.item == "15"], r[r.item == "20"]
    rows = []

    def contrast(k, lab):
        c = one(k[k.check.str.contains(lab, regex=False) & k.check.str.contains("median change")], lab)
        lv = one(k[k.check.str.contains(lab, regex=False) & k.check.str.contains("at the end window")], lab)
        return f"{pct(c.b)} against {pct(c.lo)} (p={c.p:.3f})", f"p={lv.p:.3f}"

    for lab, plain in (("2000-11 and 2012-14, five windows", "Prescribing 2000–14, deaths to 2019–21"),
                       ("2009-11 and 2012-14", "Prescribing 2009–14, deaths to 2019–21"), ("2012-14 alone", "Prescribing 2012–14, deaths to 2019–21")):
        i = one(k15[k15.check.str.startswith("interaction") & k15.check.str.endswith(lab)], lab)
        ch, lv = contrast(k15, f"prescribing in {lab}:")
        rows.append({"Prescribing and deaths": plain, "Countries": int(i.n), "Interaction (95% CI)": ci(i.b, i.lo, i.hi),
                     "Heaviest quarter, median change with a market against without": ch, "Heaviest quarter, level contrast": lv})
    i = one(k20[k20.check.str.contains("2010-12 to 2017-19")], "2017-19")
    ch, lv = contrast(k20, "end window 2017-19:")
    rows.append({"Prescribing and deaths": "Prescribing 2000–11, deaths to 2017–19", "Countries": int(i.n), "Interaction (95% CI)": ci(i.b, i.lo, i.hi),
                 "Heaviest quarter, median change with a market against without": ch, "Heaviest quarter, level contrast": lv})
    i = one(k20[k20.check.str.contains("on the same countries")], "same countries")
    rows.append({"Prescribing and deaths": "Prescribing 2000–11, deaths to 2019–21, the same countries", "Countries": int(i.n),
                 "Interaction (95% CI)": ci(i.b, i.lo, i.hi), "Heaviest quarter, median change with a market against without": "",
                 "Heaviest quarter, level contrast": ""})
    miss = one(k20[k20.term == "n"], "missing").note.replace("missing: ", "")
    L = ["Two checks on the windows. Prescribing was measured to 2011 and deaths to 2019–21, so the first three rows extend the "
         "prescribing average to 2014, the last INCB edition extracted. The fourth row ends the deaths in 2017–19, before the pandemic "
         f"years; five of the 60 countries lack two years of deaths in that window ({miss}), and the last row gives the paper's "
         "interaction on the remaining 55. The level contrast compares the USA and Canada with the rest of the heaviest quarter.", "",
         md(pd.DataFrame(rows)), ""]

    c = pd.read_csv(out / "composition_mme.csv")
    comp = pd.DataFrame({"Country": c.country.replace({"United States of America": "USA", "United Kingdom": "UK", "Netherlands (Kingdom of the)": "Netherlands"}),
                         "Market by 2019": c.status, "Rank, S-DDD": c.rank_sddd.astype(int), "Rank, MME": c.rank_mme.astype(int),
                         "S-DDD, share of USA": c.sddd_of_usa.map("{:.0%}".format), "MME, share of USA": c.mme_of_usa.map("{:.0%}".format),
                         "Oxycodone, hydrocodone": c.sddd_share_oxy_hydro.map("{:.0%}".format),
                         "Fentanyl": c.sddd_share_fentanyl.map("{:.0%}".format), "No factor": c.left_out_share.map("{:.0%}".format)})
    cc = pd.read_csv(out / "composition_contrasts.csv")
    RANK = {"all opioid analgesics, the paper's ranking": "All opioid analgesics in S-DDD, as in the paper", "oxycodone plus hydrocodone": "Oxycodone and hydrocodone",
            "morphine": "Morphine", "all opioid analgesics but fentanyl": "All but fentanyl", "morphine milligram equivalents": "Morphine milligram equivalents"}
    rows = []
    for key, plain in RANK.items():
        ch = one(cc[cc.check.str.contains(f"15 heaviest on {key}:", regex=False) & cc.check.str.contains("median change")], key)
        lv = one(cc[cc.check.str.contains(f"15 heaviest on {key}:", regex=False) & cc.check.str.contains("at the end window")], key)
        rows.append({"Countries ranked on": plain, "Median change with a market against without": f"{pct(ch.b)} against {pct(ch.lo)}",
                     "p": pv(ch.p), "Level contrast, p": pv(lv.p)})
    g, ca = c.set_index("country").loc["Germany"], c.set_index("country").loc["Canada"]
    L += ["What the heaviest prescribers consumed differed. The table gives each country's consumption in 2000–11 in INCB's defined daily "
          "doses and in morphine milligram equivalents, converting each substance's defined dose with the factors of the CDC's 2022 "
          "prescribing guideline; fentanyl's defined dose, 0.6 mg, is a 25 microgram-an-hour patch for a day. Pethidine, tilidine and "
          "INCB's residual category have no factor and are left out, which lowers Germany, Belgium and Switzerland. In defined doses "
          f"Germany consumed {g.sddd_of_canada:.0%} of Canada's level and in morphine equivalents {g.mme_of_canada:.0%}; Canada consumed "
          f"{ca.sddd_of_usa:.0%} and {ca.mme_of_usa:.0%} of the USA's. In the table MME is morphine milligram equivalents, and the last "
          "three columns are shares of each country's defined doses: from oxycodone and hydrocodone, from fentanyl, and from the "
          "substances without a conversion factor.", "", md(comp), "",
          "The contrasts in the heaviest quarter with countries ranked on each measure. The level contrast compares the USA and Canada "
          "with the other 13 in 2019–21.", "", md(pd.DataFrame(rows)), ""]
    return L


def states_absolute(r):
    k = r[(r.item == "1") & r["sample"].isin(["49 states", "48 states, without DC"])]
    rows = []
    for x in k.itertuples():
        what = ("Synthetic-opioid deaths over the three years after arrival, arrival held" if x.check.startswith("synthetic")
                else "All drug poisoning in 2008–10, before fentanyl")
        rows.append({"Outcome, deaths per 100 000 a year": what, "States": x.sample.replace("48 states, without DC", "48, without the District of Columbia").replace("49 states", "49"),
                     "Difference, 75th against 25th percentile of dispensing (95% CI)": ci(x.diff, x.diff_lo, x.diff_hi, 1),
                     "Mean": f"{x.mean_outcome:.1f}"})
    return ["Every state model above is on the logarithmic scale. The table gives the main state estimate on the absolute scale: the "
            "difference in deaths per 100 000 a year between states at the 75th and the 25th percentile of dispensing, from a linear "
            "regression with heteroskedasticity-robust errors.", "", md(pd.DataFrame(rows)), ""]


def arrival_dates(r):
    k = r[r.item == "6"]
    LAB = [("one fentanyl report per 100 000 residents", "First year with one fentanyl report per 100 000 residents"),
           ("fentanyl 5% of heroin plus fentanyl reports", "First year fentanyl was 5% of heroin and fentanyl reports"),
           ("fentanyl 10% of heroin plus fentanyl reports", "First year fentanyl was 10% of heroin and fentanyl reports"),
           ("fentanyl 20% of heroin plus fentanyl reports", "First year fentanyl was 20% of heroin and fentanyl reports")]
    base = one(k[k.check.str.startswith("the paper's half-year date") & k.check.str.endswith("+ za")], "paper's date")
    rows = [{"Arrival dated as": "The paper's date: 1% of the 50 most reported drugs, by half-year", "Years": "2014–21",
             "Agreement with the paper's date, Spearman": "", "Dispensing 2009–11, Spearman (p)": "",
             "Triplicate states later by, years (95% CI)": "", "Triplicate growth gap with the date held (95% CI)": ci(base.b, base.lo, base.hi)}]
    for key, plain in LAB:
        g = k[k.check.str.contains(key, regex=False)]
        rng = one(g[g.term == "range"], key)
        agree, disp = one(g[g.check.str.contains("Spearman with the paper")], key), one(g[g.check.str.contains("Spearman with dispensing")], key)
        late, gap = one(g[g.check.str.contains("mean date, triplicate")], key), one(g[g.check.str.endswith("+ zalt")], key)
        rows.append({"Arrival dated as": plain, "Years": f"{int(rng.lo)}–{str(int(rng.hi))[2:]}",
                     "Agreement with the paper's date, Spearman": f"{agree.b:.2f}".replace("-", "–"),
                     "Dispensing 2009–11, Spearman (p)": f"{disp.b:.2f} ({disp.p:.2f})".replace("-", "–"),
                     "Triplicate states later by, years (95% CI)": ci(late.b, late.lo, late.hi, 1),
                     "Triplicate growth gap with the date held (95% CI)": ci(gap.b, gap.lo, gap.hi)})
    return ["The triplicate gap is adjusted for when fentanyl arrived, so the date matters. The table dates arrival four more ways from "
            "the same laboratory reports, by year and without the 50-drug denominator, each required to hold the next year, and gives for "
            "each how far it agrees with the paper's date, whether dispensing predicts it, how much later the triplicate states were, and "
            "the triplicate gap in overdose growth from 1995–98 to 2017–19 with that date held; unadjusted, the gap is –0.45. Dispensing "
            "predicts none of the dates. The 5% date does not agree with the others and is shown for completeness.", "", md(pd.DataFrame(rows)), ""]


def counties(r):
    k = r[r.item == "3b"].copy()
    k["bin"] = k.term.str.replace("lrx_x_since_", "", regex=False)
    k["how"] = k.bound.astype(str).map({"1": "Entered at one", "1.0": "Entered at one", "5": "Entered at five", "5.0": "Entered at five",
                                        "9": "Entered at nine", "9.0": "Entered at nine", "censored 1-9": "Modelled as between one and nine"})
    assert k.how.notna().all(), sorted(k.bound.astype(str).unique())
    rows = []
    for how in ("Modelled as between one and nine", "Entered at one", "Entered at five", "Entered at nine"):
        g = k[k.how == how].set_index("bin")
        rows.append({"Hidden counts": how, "Arrival year": ci(g.loc["0"].ratio, g.loc["0"].ratio_lo, g.loc["0"].ratio_hi),
                     "1–2 years after": ci(g.loc["1-2"].ratio, g.loc["1-2"].ratio_lo, g.loc["1-2"].ratio_hi),
                     "3–4 years after": ci(g.loc["3-4"].ratio, g.loc["3-4"].ratio_lo, g.loc["3-4"].ratio_hi),
                     "5–6 years after": ci(g.loc["5-6"].ratio, g.loc["5-6"].ratio_lo, g.loc["5-6"].ratio_hi)})
    L = ["The model of synthetic-opioid deaths by county and year starts in 2018, so only the 15 states reached from 2019 supply its "
         "years before arrival. Half of those states' county-years are hidden. Fitted on those states alone, the estimate turns on the "
         "hidden counts as it does in all states. Each cell is mortality "
         "at the 75th against the 25th percentile of dispensing, relative to the years before arrival.", "", md(pd.DataFrame(rows)), ""]
    k = r[(r.item == "1") & r.check.str.contains("counties within states")]
    rows = []
    for x in k.itertuples():
        what = ("Synthetic-opioid deaths, 2014–20" if "synthetic" in x.check else
                "All drug poisoning in 2008–10, before fentanyl" if x.check.startswith("context") else "Change in all drug poisoning, 2008–10 to 2017–19")
        rows.append({"Outcome, deaths per 100 000 a year": what, "States": "All 49" if x.sample == "all states" else "The 31 reached by 2017",
                     "Hidden counts at": str(int(float(x.bound))), "Difference, 75th against 25th percentile of dispensing (95% CI)": ci(x.diff, x.diff_lo, x.diff_hi, 1),
                     "Mean": f"{x.mean_outcome:.1f}", "Counties": int(x.n)})
    t = pd.DataFrame(rows).sort_values(["Outcome, deaths per 100 000 a year", "States", "Hidden counts at"], kind="stable")
    L += ["The county test above is on the ratio scale: counties that prescribed more rose in proportion to where they started. Because "
          "they started higher, the same proportional rise is a larger rise in deaths. The table gives the difference in deaths per "
          "100 000 a year between counties at the 75th and the 25th percentile of dispensing within the same state, from "
          "population-weighted least squares with state effects and errors clustered by state.", "", md(t), ""]
    return L


def heroin(out):
    h = pd.read_csv(out / "heroin_rx_deaths.csv")
    SPEC = [("Heroin deaths, growth from 2009–11 to the three years before arrival, on log dispensing", "heroin deaths: growth", "49 states", "her_growth ~ lrx", "lrx"),
            ("Heroin deaths in the three years before arrival, on log dispensing", "heroin deaths: level", "49 states", "lher_pre ~ lrx", "lrx"),
            ("Synthetic-opioid rate after arrival on heroin deaths before it, arrival held", "synthetic-opioid rate, years 1-3 after arrival (log), heroin deaths alone",
             "49 states", "lrate13 ~ za + lher_pre", "lher_pre"),
            ("Synthetic-opioid rate after arrival on log dispensing, arrival and heroin deaths held", "heroin deaths before arrival held", "49 states",
             "lrate13 ~ lrx + za + lher_pre", "lrx"),
            ("The same, without the District of Columbia", "heroin deaths before arrival held", "without DC", "lrate13 ~ lrx + za + lher_pre", "lrx"),
            ("Prescription-opioid deaths in 2008–10, on log dispensing", "prescription-opioid deaths: level 2008-10", "49 states", "np.log(rxo_0810) ~ lrx", "lrx"),
            ("Prescription-opioid deaths, change from the three years before arrival to the three after, on log dispensing",
             "prescription-opioid deaths: change, 3 years before", "31 states reached by 2017", "rxo_change ~ lrx", "lrx"),
            ("Heroin deaths, the same change, on log dispensing", "heroin deaths: change, 3 years before", "31 states reached by 2017", "her_change ~ lrx", "lrx")]
    rows = []
    for plain, key, sample, model, term in SPEC:
        g = h[h.check.str.contains(key, regex=False) & (h["sample"] == sample) & (h.model == model) & (h.term == term)]
        a, b = one(g[g.bound == 1], plain), one(g[g.bound == 9], plain)
        rows.append({"Model, logarithmic scale": plain, "States": int(a.n), "Hidden counts at one (95% CI)": ci(a.b, a.lo, a.hi),
                     "Hidden counts at nine (95% CI)": ci(b.b, b.lo, b.hi)})
    return ["If prescribing built the market fentanyl entered, heroin deaths before arrival should be higher where prescribing was "
            "higher. Deaths involving heroin (T40.1) and natural and semisynthetic opioids (T40.2) by state and year, 1999–2020, are from "
            "CDC WONDER, with hidden counts entered at one and at nine; a death naming both drugs is in both series. Heroin deaths grew "
            "faster before arrival where dispensing was higher, but their level in the three years before arrival was unrelated to "
            "dispensing, and that level predicts the synthetic-opioid rate after arrival. Prescription-opioid deaths were higher where "
            "dispensing was higher and did not clearly change across arrival with dispensing; that comparison uses the 31 states with "
            "three years after arrival by 2020.", "", md(pd.DataFrame(rows)), ""]


def sections(out):
    out = pathlib.Path(out)
    r = pd.read_csv(out / "further_checks.csv", dtype={"item": str})
    return {"A5": scotland(out), "A7": countries(out, r), "A8": states_absolute(r), "A9": arrival_dates(r), "A10": counties(r), "A12": heroin(out)}
