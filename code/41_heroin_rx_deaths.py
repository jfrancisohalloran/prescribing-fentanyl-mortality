"""Heroin and prescription-opioid deaths before and after fentanyl's arrival, by state. If prescribing built the market fentanyl
entered, heroin deaths before arrival should be higher where dispensing was higher, and a flat total after arrival could hide
prescription-opioid deaths falling under synthetic-opioid deaths rising. Deaths involving heroin (T40.1) and natural and
semisynthetic opioids (T40.2) by state and year, 1999-2020, are from CDC WONDER; counts of 1 to 9 that WONDER hides enter at 1 and
at 9. Set after the paper's results and a review were seen; every variant fitted is written out. The script first
rebuilds the paper's main state estimate and stops if it does not.

Writes out/heroin_rx_deaths.csv: estimates only, no count of deaths.

  python3 code/41_heroin_rx_deaths.py <folder holding code/ and out/> <wonder_state_year_t404 date folder>
      <wonder_state_year_t401_t402 date folder>
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr


def read(path, bound):
    a = pd.read_csv(path, sep="\t", dtype=str)
    k = pd.DataFrame(dict(st=a.state.str.extract(r"\((\d\d)\)")[0].astype(int), year=a.year.str[:4].astype(int),
                          pop=a.population.str.replace(",", "").astype(float), hidden=a.deaths.eq("Suppressed"),
                          d=pd.to_numeric(a.deaths.str.replace(",", ""), errors="coerce")))
    k["y"] = np.where(k.hidden, bound, k.d)
    return k.set_index(["st", "year"]).sort_index()


def span(p, st, y0, y1):
    """Rate per 100 000 a year over y0..y1, and how many of its years are hidden; missing if the span leaves the file."""
    try:
        k = p.loc[(st, slice(y0, y1)), :]
    except KeyError:
        return np.nan, 0
    if len(k) < y1 - y0 + 1:
        return np.nan, 0
    return k.y.sum() / k["pop"].sum() * 1e5, int(k.hidden.sum())


def fit(res, label, sample, formula, data, terms, bound):
    r = smf.ols(formula, data=data).fit(cov_type="HC3")
    for t in terms:
        ci = r.conf_int().loc[t]
        res.append(dict(check=label, sample=sample, model=formula, term=t, bound=bound, b=r.params[t], lo=ci[0], hi=ci[1], p=r.pvalues[t],
                        n=int(r.nobs)))


def main(paper, d77, dnew):
    paper, d77, dnew = pathlib.Path(paper), pathlib.Path(d77), pathlib.Path(dnew)
    spec = importlib.util.spec_from_file_location("c39", paper / "code" / "39_further_checks.py")
    c39 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c39)
    syn = c39.state_rates(d77)
    s0 = pd.read_csv(paper / "out" / "alpert_rerun_frame.csv").set_index("st")
    s0["Y"] = np.floor(s0.arrival).astype(int)
    s0["rate13"] = [c39.span_rate(syn, st, r.Y + 1, r.Y + 3) for st, r in s0.iterrows()]
    s0["lrate13"] = np.log(s0.rate13)
    main_est = smf.ols("lrate13 ~ lrx + za", data=s0).fit(cov_type="HC3").params.lrx
    assert abs(main_est - 0.369) < 0.001, main_est
    print(f"paper's main state estimate rebuilt: {main_est:.3f}; states {len(s0)}; arrival years {s0.Y.min()}-{s0.Y.max()}")
    q25, q75 = np.percentile(s0.rx, [25, 75])
    iqr = np.log(q75 / q25)
    res, notes = [], []
    for bound in (1, 9):
        her, rxo = read(dnew / "wonder_t401_state_year_1999_2020.tsv", bound), read(dnew / "wonder_t402_state_year_1999_2020.tsv", bound)
        s = s0.copy()
        for tag, p in (("her", her), ("rxo", rxo)):
            for w, f in (("base", lambda r: (2009, 2011)), ("pre", lambda r: (r.Y - 3, r.Y - 1)), ("post", lambda r: (r.Y + 1, r.Y + 3)),
                         ("0810", lambda r: (2008, 2010))):
                v = [span(p, st, *f(r)) for st, r in s.iterrows()]
                s[f"{tag}_{w}"] = [a for a, _ in v]
                s[f"{tag}_{w}_hid"] = [h for _, h in v]
            s[f"l{tag}_pre"] = np.log(s[f"{tag}_pre"])
            s[f"{tag}_growth"] = np.log(s[f"{tag}_pre"] / s[f"{tag}_base"])
            s[f"{tag}_change"] = np.log(s[f"{tag}_post"] / s[f"{tag}_pre"])
            s[f"{tag}_diff"] = s[f"{tag}_post"] - s[f"{tag}_pre"]
        s["syn_pre"] = [span(syn, st, r.Y - 3, r.Y - 1)[0] for st, r in s.iterrows()]
        s["syn_diff"] = s.rate13 - s.syn_pre
        notes.append(f"bound {bound}: heroin state-spans with a hidden year, base {int((s.her_base_hid > 0).sum())}, before arrival "
                     f"{int((s.her_pre_hid > 0).sum())}; prescription-opioid, before {int((s.rxo_pre_hid > 0).sum())}, after "
                     f"{int((s.rxo_post_hid > 0).sum())}; states with three years after arrival by 2020: {int(s.rxo_post.notna().sum())}")
        # item 5 with deaths: heroin before arrival
        for sname, k in (("49 states", s), ("reached 2016 or later", s[s.Y >= 2016]), ("without DC", s[s.state != "DC"])):
            fit(res, "heroin deaths: growth, 2009-11 to the 3 years before arrival (log ratio)", sname, "her_growth ~ lrx", k, ["lrx"], bound)
            fit(res, "heroin deaths: growth, 2009-11 to the 3 years before arrival (log ratio)", sname, "her_growth ~ lrx + white", k, ["lrx"], bound)
            fit(res, "heroin deaths: level in the 3 years before arrival (log)", sname, "lher_pre ~ lrx", k, ["lrx"], bound)
            rho, pv = spearmanr(k.rx, k.her_growth)
            res.append(dict(check="heroin deaths: Spearman, dispensing with growth", sample=sname, model="", term="rho", bound=bound, b=rho, p=pv, n=len(k)))
            rho, pv = spearmanr(k.rx, k.her_pre)
            res.append(dict(check="heroin deaths: Spearman, dispensing with level before arrival", sample=sname, model="", term="rho", bound=bound, b=rho, p=pv, n=len(k)))
        for sname, k in (("49 states", s), ("without DC", s[s.state != "DC"])):
            fit(res, "synthetic-opioid rate, years 1-3 after arrival (log), heroin deaths before arrival held", sname,
                "lrate13 ~ lrx + za + lher_pre", k, ["lrx", "lher_pre"], bound)
            fit(res, "synthetic-opioid rate, years 1-3 after arrival (log), heroin deaths alone", sname, "lrate13 ~ za + lher_pre", k, ["lher_pre"], bound)
            fit(res, "synthetic-opioid rate, years 1-3 after arrival (log), heroin and market type held", sname,
                "lrate13 ~ lrx + za + lher_pre + white", k, ["lrx", "lher_pre", "white"], bound)
        # item 4: prescription-opioid deaths before and after arrival
        fit(res, "prescription-opioid deaths: level 2008-10 (log)", "49 states", "np.log(rxo_0810) ~ lrx", s, ["lrx"], bound)
        fit(res, "prescription-opioid deaths: level in the 3 years before arrival (log)", "49 states", "lrxo_pre ~ lrx", s, ["lrx"], bound)
        k = s.dropna(subset=["rxo_post"])
        for sname, kk in ((f"{len(k)} states reached by 2017", k), ("the same, without DC", k[k.state != "DC"])):
            fit(res, "prescription-opioid deaths: change, 3 years before to 3 years after arrival (log ratio)", sname, "rxo_change ~ lrx", kk, ["lrx"], bound)
            fit(res, "prescription-opioid deaths: change per 100 000 a year", sname, "rxo_diff ~ lrx", kk, ["lrx"], bound)
            fit(res, "heroin deaths: change per 100 000 a year", sname, "her_diff ~ lrx", kk, ["lrx"], bound)
            fit(res, "synthetic-opioid deaths: change per 100 000 a year", sname, "syn_diff ~ lrx", kk, ["lrx"], bound)
            fit(res, "heroin deaths: change, 3 years before to 3 years after arrival (log ratio)", sname, "her_change ~ lrx", kk, ["lrx"], bound)
        if bound == 1:
            d = s[["rxo_0810", "rxo_pre", "rxo_post", "her_base", "her_pre", "her_post", "syn_pre", "rate13"]].describe().loc[["mean", "50%"]]
            print("state means and medians per 100 000 a year, hidden counts at 1 (rates only):")
            print(d.round(2).to_string())
    r = pd.DataFrame(res)
    r["per_iqr"] = r.b.where(r.term == "lrx") * iqr
    for n in notes:
        print(n)
    print(f"75th against 25th percentile of state dispensing: log ratio {iqr:.3f}")
    with pd.option_context("display.width", 300, "display.max_rows", 400, "display.max_colwidth", 95):
        print(r.round(3).to_string(index=False))
    r.to_csv(paper / "out" / "heroin_rx_deaths.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
