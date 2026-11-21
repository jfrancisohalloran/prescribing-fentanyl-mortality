"""Inside each state, did counties that prescribed more before fentanyl rise more once the state's fentanyl arrived?

The state panel of code/07 asks this across 49 states and finds it too imprecise to judge. Counties give about 2,700 units with
dispensing data, and state-by-window fixed effects hold each state's arrival timing, policy and everything else common to the state in
a window fixed, so only differences between a state's own counties are used. The main model was fixed before any result was
seen, and every variant is reported.

Data, loaded as code/07 and code/24 load them:
- Prescribing: CDC's county opioid dispensing rate per 100 persons, 2009-11, and the economic distress index of 2000, from the companion county study's
  `county_base.csv`. That file holds the 48 contiguous states and DC, counties with at least 1,000 non-Hispanic white residents in
  2000, without Broomfield. Counties whose dispensing rate CDC did not report (`rx_missing` = 1, filled there with the state median)
  are dropped, as code/07 drops them.
- Pills: oxycodone and hydrocodone dosage units per person, 2006-12, from the ARCOS county-year file (summed units over summed people).
- Arrival: the companion county study's half-year clock `arr_p1_persist` (`halfyear/analogue_state_timing.csv`); its bootstrap standard deviation from
  `halfyear/arrival_dates_uncertainty.csv`. The arrival year is the clock's year, so an arrival in the second half of a year counts
  that whole year as after arrival; a check counts half-years instead.
- Deaths: CDC WONDER county files of drug poisoning as underlying cause (X40-X44, X60-X64, X85, Y10-Y14) in 2005-07, 2008-10,
  2011-13, 2014-16 and 2017-19; synthetic opioids (T40.4 among multiple causes) pooled over 2014-20; and the county-by-year T40.4 file
  for 2018-2025, final through 2024 and provisional for 2025. WONDER hides counts of 1-9; they enter at 1, 5 and 9 in turn. Rows
  WONDER marks Missing, or with population Not Available, are dropped.
- NCHS's model-based county rates are used only as a level check, since their notes advise against using them for timing.

Main model: county-by-window Poisson pseudo-maximum likelihood of deaths with log person-years as offset; fixed effects for county,
state x window, and log dispensing x window; the term of interest is log dispensing x the share of the window's three years in or after
the state's arrival year. Errors clustered by state. The county effects are profiled out exactly (the Poisson likelihood conditional
on each county's total), so the other parameters are fitted by Newton's method on a dense design. Variants: event-time bins of windows
relative to arrival; economic distress as a rival; ARCOS pills in place of dispensing; without the states whose arrival date has a
bootstrap standard deviation above a year; the T40.4 county-year panel and the T40.4 pooled cross-section, beside the same
cross-section for all drug poisoning in 2008-10 and 2017-19; population weights; and the threshold of non-Hispanic white residents.
Because the bounds disagree in sign for the T40.4 county-year panel, the main, event-time and synthetic-opioid models are also fitted
by censored Poisson maximum likelihood, in which a hidden count is known only to lie between 1 and 9. Its county effects are estimated, not profiled, so it checks the bounds rather than replacing them.

Writes, all estimates or counts of counties, never a death count:
  out/county_arrival_results.csv   every non-fixed-effect coefficient, by model and bound
  out/county_arrival_sample.csv    counties and county-windows kept and dropped, with the reason
  out/county_arrival_checks.csv    dispensing percentiles, the NCHS level check, and a check of the fitting code against statsmodels

  /opt/homebrew/bin/python3.13 -I code/33_county_arrival.py <inputs/1a_out folder> <arcos_county_year.csv> <wonder_pooled date folder>
      <wonder_county_year date folder> <nchs_county_model date folder> <out folder> [<wonder_county_year date folder holding
      wonder_county_year_od.tsv, all drug poisoning 2018-25, for model 5c>]
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm, spearmanr

WINDOWS = {"2005-07": 2005, "2008-10": 2008, "2011-13": 2011, "2014-16": 2014, "2017-19": 2017}
WFILE = {w: f"ucd_od_{y}_{y + 2}" for w, y in WINDOWS.items()}
BOUNDS = (1, 5, 9)
EVENT_BINS = ["e<=-3", "e=-2", "e=0", "e=+1"]  # windows relative to the window holding the arrival year; e=-1 is the reference
YEAR_BINS = [("0", 0, 0), ("1-2", 1, 2), ("3-4", 3, 4), ("5-6", 5, 6), ("7+", 7, 99)]  # years since arrival; years before are the reference


# ---------------------------------------------------------------- loading, as code/07 and code/24 do it

def read_wonder(path, bound):
    """code/07's reader: hidden counts enter at the bound. Rows WONDER marks Missing are dropped."""
    d = pd.read_csv(path, sep="\t", dtype={"fips": str})
    d["fips"] = d.fips.str.zfill(5)
    d["hidden"] = d.deaths.eq("Suppressed")
    d["deaths"] = pd.to_numeric(d.deaths.replace("Suppressed", bound), errors="coerce")
    d["population"] = pd.to_numeric(d.population, errors="coerce")
    return d.dropna(subset=["deaths", "population"])


def read_county_year(path, bound):
    w = pd.read_csv(path, sep="\t", dtype=str)
    code = "Residence County (without Puerto Rico) Code"
    w = w[w[code].notna() & w["Year Code"].notna()]
    d = pd.DataFrame(dict(fips=w[code].str.zfill(5), year=w["Year Code"].astype(int), hidden=w.Deaths.eq("Suppressed"),
                          deaths=pd.to_numeric(w.Deaths.replace("Suppressed", bound), errors="coerce"),
                          population=pd.to_numeric(w.Population, errors="coerce")))
    return d.dropna(subset=["deaths", "population"])


def counties(onea, arcos):
    onea = pathlib.Path(onea)
    cb = pd.read_csv(onea / "county_base.csv", dtype={"fips": str})
    cb["fips"] = cb.fips.str.zfill(5)
    clock = pd.read_csv(onea / "halfyear/analogue_state_timing.csv")
    sd = pd.read_csv(onea / "halfyear/arrival_dates_uncertainty.csv").set_index("state").arr_p1_persist_sd
    clock["sd"] = clock.state.map(sd)
    cb = cb.merge(clock[["st_fips", "state", "arr_p1_persist", "sd"]].rename(columns={"st_fips": "st", "arr_p1_persist": "arr"}), on="st", how="left")
    a = pd.read_csv(arcos, dtype={"fips": str})
    a["fips"] = a.fips.str.zfill(5)
    a = a[a.year.between(2006, 2012) & a.population_wapo.notna()]
    p = a.groupby("fips")[["pills_oxy_hydro", "population_wapo"]].sum()
    cb["pills"] = cb.fips.map(p.pills_oxy_hydro / p.population_wapo)
    cb["lrx"] = np.log(cb.rx)
    cb["lpills"] = np.log(cb.pills.where(cb.pills > 0))
    cb["zx"] = cb.econ_distress
    cb["A"] = np.floor(cb.arr).astype(int)
    return cb


def after_share(A, arr, y0):
    """Share of the window's three years in or after the arrival year, and the half-year version."""
    yrs = np.arange(y0, y0 + 3)
    by_year = (yrs[None, :] >= A[:, None]).mean(axis=1)
    halves = np.arange(y0, y0 + 3, 0.5)
    by_half = (halves[None, :] >= (arr - 0.25)[:, None]).mean(axis=1)  # 2017.25 is the first half of 2017, 2017.75 the second
    return by_year, by_half


def panel(cb, wonder, bound):
    rows = []
    for w, y0 in WINDOWS.items():
        d = read_wonder(wonder / f"{WFILE[w]}_county.tsv", bound)[["fips", "deaths", "population", "hidden"]]
        k = cb.merge(d, on="fips", how="inner").assign(window=w, y0=y0)
        k["after"], k["after_half"] = after_share(k.A.to_numpy(), k.arr.to_numpy(), y0)
        rows.append(k)
    p = pd.concat(rows, ignore_index=True)
    p["t"] = p.window.map({w: i + 1 for i, w in enumerate(WINDOWS)})
    p["c"] = (p.A - 2005) // 3 + 1  # index of the window that holds the arrival year (6 = 2020-22, after the panel)
    e = p.t - p.c
    p["ebin"] = np.select([e <= -3, e == -2, e == -1, e == 0, e == 1], ["e<=-3", "e=-2", "e=-1", "e=0", "e=+1"], "other")
    return p


# ---------------------------------------------------------------- Poisson PML with one profiled fixed effect

def fit_ppml(y, off, X, group, cluster, weights=None, tol=1e-10, maxit=200):
    """Poisson PML with group fixed effects profiled out. X holds every other regressor, fixed-effect dummies included.

    Returns coefficients, the clustered covariance (sandwich with the profiled Hessian, times G/(G-1)), iterations and convergence."""
    group = pd.factorize(group)[0]
    order = np.argsort(group, kind="stable")
    y, off, X, group, cluster = y[order], off[order], X[order], group[order], cluster[order]
    w = np.ones(len(y)) if weights is None else weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(y)]))
    n = np.add.reduceat(y, starts)

    def probs(b):
        eta = X @ b + off
        m = np.maximum.reduceat(eta, starts)[gid]
        e = np.exp(eta - m)
        s = np.add.reduceat(e, starts)[gid]
        return e / s

    def loglik(b):
        p = probs(b)
        return np.sum(w * np.where(y > 0, y * np.log(np.maximum(p, 1e-300)), 0.0))

    b = np.zeros(X.shape[1])
    ll, converged, it = loglik(b), False, 0
    for it in range(1, maxit + 1):
        p = probs(b)
        mu = n[gid] * p
        score = X.T @ (w * (y - mu))
        M = np.add.reduceat(mu[:, None] * X, starts, axis=0)
        wg = w[starts]
        H = X.T @ ((w * mu)[:, None] * X) - M.T @ ((wg / np.maximum(n, 1e-300))[:, None] * M)
        step = np.linalg.solve(H, score)
        lam = 1.0
        while True:
            ll_new = loglik(b + lam * step)
            if ll_new >= ll - 1e-12 * abs(ll) or lam < 1e-8:
                break
            lam /= 2
        b, ll_old, ll = b + lam * step, ll, ll_new
        if np.max(np.abs(lam * step)) < tol or abs(ll - ll_old) < 1e-14 * abs(ll):
            break
    p = probs(b)
    mu = n[gid] * p
    M = np.add.reduceat(mu[:, None] * X, starts, axis=0)
    wg = w[starts]
    H = X.T @ ((w * mu)[:, None] * X) - M.T @ ((wg / np.maximum(n, 1e-300))[:, None] * M)
    Hi = np.linalg.inv(H)
    score = X.T @ (w * (y - mu))
    converged = bool(score @ Hi @ score < 1e-8 and it < maxit)  # Newton decrement at the solution
    sc = (w * (y - mu))[:, None] * X
    cl_codes, cl = np.unique(cluster, return_inverse=True)
    S = np.zeros((len(cl_codes), X.shape[1]))
    np.add.at(S, cl, sc)
    G = len(cl_codes)
    V = Hi @ (S.T @ S) @ Hi * G / (G - 1)
    return b, V, it, converged, G


def fit_censored(y, hidden, off, X, group, cluster, lo=1, hi=9, tol=1e-9, maxit=200):
    """Poisson maximum likelihood in which WONDER's hidden counts are known only to lie between lo and hi, with group fixed effects.

    Group effects are estimated alongside the other parameters; Newton steps use the Schur complement, since the group block of the
    information matrix is diagonal. Errors clustered as in fit_ppml. Unlike the Poisson PML the fixed-effect estimate is not exactly
    free of incidental-parameter bias, so it is a check on the bounds, not a replacement for them."""
    from scipy.special import gammaln, logsumexp
    group = pd.factorize(group)[0]
    order = np.argsort(group, kind="stable")
    y, hidden, off, X, group, cluster = y[order], hidden[order], off[order], X[order], group[order], cluster[order]
    starts = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(y)]))
    ks = np.arange(lo, hi + 1, dtype=float)
    yo = np.where(hidden, 0.0, y)

    def parts(alpha, b):
        eta = alpha[gid] + X @ b + off
        mu = np.exp(eta)
        ll = np.where(hidden, 0.0, yo * eta - mu - gammaln(yo + 1))
        g, h = yo - mu, mu.copy()
        if hidden.any():
            e, m = eta[hidden], mu[hidden]
            lp = -m[:, None] + ks[None, :] * e[:, None] - gammaln(ks + 1)[None, :]
            L = logsumexp(lp, axis=1)
            q = np.exp(lp - L[:, None])
            E = q @ ks
            var = q @ ks ** 2 - E ** 2
            ll[hidden], g[hidden], h[hidden] = L, E - m, m - var
        return ll.sum(), g, h

    tot = np.add.reduceat(np.where(hidden, (lo + hi) / 2, y), starts)
    pop = np.add.reduceat(np.exp(off), starts)
    alpha, b = np.log(tot / pop), np.zeros(X.shape[1])
    ll, g, h = parts(alpha, b)
    converged = False
    for it in range(1, maxit + 1):
        D = np.add.reduceat(h, starts)
        ga = np.add.reduceat(g, starts)
        M = np.add.reduceat(h[:, None] * X, starts, axis=0)
        S = X.T @ (h[:, None] * X) - M.T @ (M / D[:, None])
        rhs = X.T @ g - M.T @ (ga / D)
        db = np.linalg.solve(S, rhs)
        da = (ga - M @ db) / D
        dec = rhs @ db + np.sum(ga * ga / D)  # Newton decrement
        lam = 1.0
        while True:
            ll_new, g_new, h_new = parts(alpha + lam * da, b + lam * db)
            if ll_new >= ll - 1e-10 * abs(ll) or lam < 1e-8:
                break
            lam /= 2
        alpha, b, ll, g, h = alpha + lam * da, b + lam * db, ll_new, g_new, h_new
        if dec < tol:
            converged = True
            break
    D = np.add.reduceat(h, starts)
    M = np.add.reduceat(h[:, None] * X, starts, axis=0)
    S = X.T @ (h[:, None] * X) - M.T @ (M / D[:, None])
    Si = np.linalg.inv(S)
    cl_codes, cl = np.unique(cluster, return_inverse=True)
    Sc = np.zeros((len(cl_codes), X.shape[1]))
    np.add.at(Sc, cl, g[:, None] * X)
    G = len(cl_codes)
    return b, Si @ (Sc.T @ Sc) @ Si * G / (G - 1), it, converged, G


def independent_columns(X, group, names, protect):
    """Drop fixed-effect columns that are linear in the group effects and earlier columns; report any protected column dropped."""
    df = pd.DataFrame(X)
    Xd = (df - df.groupby(group).transform("mean")).to_numpy()
    keep, dropped = [], []
    Q = np.zeros((Xd.shape[0], 0))
    for j in range(Xd.shape[1]):
        v = Xd[:, j] - Q @ (Q.T @ Xd[:, j]) if Q.shape[1] else Xd[:, j].copy()
        v = v - Q @ (Q.T @ v) if Q.shape[1] else v
        nv = np.linalg.norm(v)
        if nv > 1e-8 * max(1.0, np.linalg.norm(Xd[:, j])):
            Q = np.column_stack([Q, v / nv])
            keep.append(j)
        else:
            dropped.append(names[j])
    lost = [d for d in dropped if d in protect]
    return keep, lost


def design(p, expo, terms, fe_extra=()):
    """Fixed-effect dummies (state x window, exposure x window, and any extra variable x window) followed by the terms of interest."""
    cols, names = [], []
    later = [w for w in WINDOWS][1:]
    for st in sorted(p.st.unique()):
        for w in later:
            cols.append(((p.st == st) & (p.window == w)).astype(float).to_numpy())
            names.append(f"fe_st{st}_{w}")
    for v in (expo,) + tuple(fe_extra):
        for w in later:
            cols.append((p[v] * (p.window == w)).to_numpy())
            names.append(f"{v}_x_{w}")
    for name, col in terms:
        cols.append(np.asarray(col, dtype=float))
        names.append(name)
    return np.column_stack(cols), names


def fit(p, X, group, censored=False, weights=None):
    """Poisson PML with hidden counts at the bound already in p.deaths, or the censored likelihood treating them as 1 to 9."""
    y, off = p.deaths.to_numpy(float), np.log(p.population.to_numpy(float))
    if censored:
        return fit_censored(y, p.hidden.to_numpy(bool), off, X, group, p.st.to_numpy())
    return fit_ppml(y, off, X, group, p.st.to_numpy(), weights)


def run(p, expo, terms, label, bound, out, extra=(), weights=None, iqr=None, note="", censored=False, iqr_extra=None):
    """Fit one county-by-window model and append its non-fixed-effect coefficients to out."""
    X, names = design(p, expo, terms, extra)
    protect = {t for t, _ in terms}
    keep, lost = independent_columns(X, p.fips.to_numpy(), names, protect)
    X, names = X[:, keep], [names[j] for j in keep]
    wts = None if weights is None else (weights / weights.mean()).to_numpy(float)
    b, V, it, conv, G = fit(p, X, p.fips.to_numpy(), censored, wts)
    rows = []
    for j, nm in enumerate(names):
        if nm.startswith("fe_"):
            continue
        se = np.sqrt(V[j, j])
        r = dict(model=label, bound=bound, term=nm, b=b[j], se=se, lo=b[j] - 1.96 * se, hi=b[j] + 1.96 * se, p=2 * norm.sf(abs(b[j] / se)),
                 counties=p.fips.nunique(), obs=len(p), states=G, iterations=it, converged=conv, not_identified=" ".join(lost), note=note)
        q = iqr if nm.startswith(expo) else (iqr_extra or {}).get(nm.split("_x_")[0])
        if q is not None and nm in protect:
            # ratio of after-arrival to before-arrival mortality, 75th-percentile county over 25th, for a window wholly after arrival
            r.update(iqr_log=q, ratio=np.exp(b[j] * q), ratio_lo=np.exp((b[j] - 1.96 * se) * q), ratio_hi=np.exp((b[j] + 1.96 * se) * q))
        rows.append(r)
    out.extend(rows)
    return dict(zip(names, b)), V, names


def informative(p):
    """Drop counties with no deaths in any window or with one window only: they carry no information once county effects are in."""
    g = p.groupby("fips")
    ok = (g.deaths.transform("sum") > 0) & (g.window.transform("size") > 1)
    return p[ok].copy(), p.loc[~ok, "fips"].nunique()


def wald(b, V, names, which):
    idx = [names.index(w) for w in which if w in names]
    bb = np.array([b[names[i]] for i in idx])
    VV = V[np.ix_(idx, idx)]
    stat = float(bb @ np.linalg.solve(VV, bb))
    return stat, len(idx), chi2.sf(stat, len(idx))


# ---------------------------------------------------------------- the analyses

def main(onea, arcos, wonder, wcy, nchs, outdir, wcy_od=None):
    wonder, wcy, nchs, outdir = pathlib.Path(wonder), pathlib.Path(wcy), pathlib.Path(nchs), pathlib.Path(outdir)
    cb0 = counties(onea, arcos)
    sample, res, checks = [], [], []
    sample.append(dict(step="county_base.csv: 48 contiguous states and DC, at least 1,000 non-Hispanic white residents, no Broomfield", bound="all", counties=len(cb0)))
    cb = cb0[cb0.rx_missing == 0].copy()
    sample.append(dict(step="dropped: CDC reported no dispensing rate for 2009-11 (rx_missing = 1)", bound="all", counties=len(cb0) - len(cb)))
    sample.append(dict(step="counties with a dispensing rate", bound="all", counties=len(cb)))
    unc = cb0.drop_duplicates("st")
    uncertain = sorted(unc.state[unc.sd > 1])
    assert set(uncertain) == {"NV", "ND", "SD", "UT"}, uncertain
    p25, p75 = np.percentile(cb.rx, [25, 75])
    iqr_rx = np.log(p75 / p25)
    q25, q75 = np.nanpercentile(cb.pills, [25, 75])
    iqr_p = np.log(q75 / q25)
    x25, x75 = np.percentile(cb.zx, [25, 75])
    checks += [dict(check="dispensing per 100, 25th percentile of counties", value=p25), dict(check="dispensing per 100, 75th percentile", value=p75),
               dict(check="log ratio of 75th to 25th percentile of dispensing", value=iqr_rx),
               dict(check="pills per person 2006-12, 25th percentile", value=q25), dict(check="pills per person 2006-12, 75th percentile", value=q75),
               dict(check="log ratio of 75th to 25th percentile of pills", value=iqr_p),
               dict(check="economic distress, 25th to 75th percentile difference (SD units)", value=x75 - x25),
               dict(check="Spearman, dispensing against pills across counties", value=spearmanr(cb.rx, cb.pills, nan_policy="omit")[0]),
               dict(check="Spearman, dispensing against economic distress across counties", value=spearmanr(cb.rx, cb.zx)[0]),
               dict(check="counties with dispensing below 10 per 100", value=int((cb.rx < 10).sum())),
               dict(check="states whose arrival date has bootstrap SD above a year, in the panel", value=" ".join(uncertain)),
               dict(check="states with an arrival year by 2019, so some after-arrival years in 2005-19", value=int((unc[unc.rx.notna()].A <= 2019).sum())),
               dict(check="states arriving in 2020-21, no after-arrival years in 2005-19", value=" ".join(sorted(unc.state[unc.A >= 2020]))),
               dict(check="states with a full window after arrival (arrival year by 2017)", value=int((unc.A <= 2017).sum()))]
    for bound in BOUNDS:
        p = panel(cb, wonder, bound)
        if bound == 1:
            miss = len(cb) * len(WINDOWS) - len(p)
            sample.append(dict(step="dropped county-windows WONDER marks Missing", bound="all", counties=np.nan, county_windows=miss))
            for w in WINDOWS:
                k = p[p.window == w]
                checks.append(dict(check=f"share of county-windows hidden by WONDER (1-9 deaths), {w}", value=k.hidden.mean()))
        p, nd = informative(p)
        sample.append(dict(step="dropped: no deaths in any window, or one window only", bound=bound, counties=nd))
        sample.append(dict(step="main panel", bound=bound, counties=p.fips.nunique(), county_windows=len(p), states=p.st.nunique()))

        # main model
        main_terms = [("lrx_x_after", p.lrx * p.after)]
        run(p, "lrx", main_terms, "main", bound, res, iqr=iqr_rx)
        run(p, "lrx", [("lrx_x_after", p.lrx * p.after_half)], "check: after-share counted in half-years", bound, res, iqr=iqr_rx)
        # 1. event time
        ev = [(f"lrx_x_{e}", p.lrx * (p.ebin == e)) for e in EVENT_BINS]
        b, V, names = run(p, "lrx", ev, "1 event time (reference e=-1)", bound, res, iqr=iqr_rx)
        stat, df, pv = wald(b, V, names, ["lrx_x_e<=-3", "lrx_x_e=-2"])
        res.append(dict(model="1 event time (reference e=-1)", bound=bound, term="Wald test, pre-arrival bins jointly zero", b=stat, p=pv, note=f"chi-square, {df} df"))
        if bound == 5:
            cnt = p.drop_duplicates(["st", "window"]).groupby("ebin").st.size()
            for e, n_ in cnt.items():
                checks.append(dict(check=f"state-windows in event bin {e}", value=int(n_)))
        # 2. distress as a rival
        run(p, "lrx", [("lrx_x_after", p.lrx * p.after), ("zx_x_after", p.zx * p.after)], "2 rival: economic distress", bound, res, extra=("zx",), iqr=iqr_rx,
            iqr_extra={"zx": x75 - x25})
        # 3. pills
        pp, _ = informative(p[p.lpills.notna()])
        run(pp, "lpills", [("lpills_x_after", pp.lpills * pp.after)], "3 pills (ARCOS) in place of dispensing", bound, res, iqr=iqr_p)
        if bound == 1:
            sample.append(dict(step="pills variant: counties without ARCOS pills 2006-12 dropped", bound="all", counties=p.fips.nunique() - pp.fips.nunique()))
        # 4. drop states with uncertain arrival
        pc = p[~p.state.isin(uncertain)]
        run(pc, "lrx", [("lrx_x_after", pc.lrx * pc.after)], "4 without NV, ND, SD, UT (arrival SD above a year)", bound, res, iqr=iqr_rx)
        # 6. population weights; the threshold
        popw = p.groupby("fips").population.transform("mean")
        run(p, "lrx", main_terms, "6 weighted by population (mean person-years)", bound, res, weights=popw, iqr=iqr_rx)
        big = p[p.nhw_n >= 2500]
        run(big, "lrx", [("lrx_x_after", big.lrx * big.after)], "6 substitute: counties with at least 2,500 non-Hispanic white residents", bound, res, iqr=iqr_rx,
            note="county_base.csv holds no county under 1,000; the raw CDC rates for those are not on this Mac")
        # 5a. T40.4 county-year, 2018-2025
        cy = read_county_year(wcy / "wonder_county_year_t404.tsv", bound).merge(cb[["fips", "st", "state", "lrx", "A"]], on="fips")
        cy["since"] = cy.year - cy.A
        cy = cy.rename(columns={"year": "yr"})
        g = cy.groupby("fips")
        cy = cy[(g.deaths.transform("sum") > 0) & (g.yr.transform("size") > 1)].copy()
        yrs = sorted(cy.yr.unique())
        lin = [("lrx_x_years_since", cy.lrx * cy.since)]
        bins = [(f"lrx_x_since_{lab}", cy.lrx * cy.since.between(lo, hi)) for lab, lo, hi in YEAR_BINS]
        for lab, terms, with_trend in (("5a T40.4 county-year: linear in years since arrival", lin, False),
                                       ("5a T40.4 county-year: bins of years since arrival", bins, False),
                                       ("5a T40.4 county-year: bins, plus dispensing x year", bins, True)):
            run_years(cy, yrs, terms, lab, bound, res, with_trend, iqr_rx)
        if bound == 5:
            sample.append(dict(step="5a county-year panel 2018-2025", bound=bound, counties=cy.fips.nunique(), county_windows=len(cy), states=cy.st.nunique()))
            for lab, lo, hi in [("before arrival", -99, -1)] + YEAR_BINS:
                k = cy[cy.since.between(lo, hi)]
                checks.append(dict(check=f"5a: states with county-years {lab}; their arrival years", value=f"{k.state.nunique()} states; {sorted(k.A.unique())}"))
        # 5c. the same county-year model for all drug poisoning, 2018-2025, when that file is given
        if wcy_od:
            co = read_county_year(pathlib.Path(wcy_od) / "wonder_county_year_od.tsv", bound).merge(cb[["fips", "st", "state", "lrx", "A"]], on="fips")
            co["since"] = co.year - co.A
            co = co.rename(columns={"year": "yr"})
            g = co.groupby("fips")
            co = co[(g.deaths.transform("sum") > 0) & (g.yr.transform("size") > 1)].copy()
            obins = [(f"lrx_x_since_{lab}", co.lrx * co.since.between(lo, hi)) for lab, lo, hi in YEAR_BINS]
            for lab, with_trend in (("5c all drug poisoning county-year: bins of years since arrival", False),
                                    ("5c all drug poisoning county-year: bins, plus dispensing x year", True)):
                run_years(co, sorted(co.yr.unique()), obins, lab, bound, res, with_trend, iqr_rx)
            if bound == 5:
                sample.append(dict(step="5c county-year panel 2018-2025, all drug poisoning", bound=bound, counties=co.fips.nunique(), county_windows=len(co),
                                   states=co.st.nunique()))
                checks.append(dict(check="5c: share of county-years hidden by WONDER (1-9 deaths)", value=co.hidden.mean()))
                co5 = co
        # 5b. T40.4 pooled 2014-20, cross-section within states
        t = read_wonder(wonder / "mcd_t404_2014_2020_county.tsv", bound).merge(cb[["fips", "st", "lrx", "nhw_n"]], on="fips")
        pos = t[t.deaths > 0].assign(lrate=lambda k: np.log(k.deaths / k.population))
        r = smf.ols("lrate ~ lrx + C(st)", data=pos).fit(cov_type="cluster", cov_kwds={"groups": pos.st})
        res.append(dict(model="5b T40.4 2014-20 cross-section: OLS of log rate, state effects, counties with deaths", bound=bound, term="lrx", b=r.params.lrx, se=r.bse.lrx,
                        lo=r.conf_int().loc["lrx", 0], hi=r.conf_int().loc["lrx", 1], p=r.pvalues.lrx, counties=len(pos), obs=len(pos), states=pos.st.nunique(),
                        iqr_log=iqr_rx, ratio=np.exp(r.params.lrx * iqr_rx), ratio_lo=np.exp(r.conf_int().loc["lrx", 0] * iqr_rx), ratio_hi=np.exp(r.conf_int().loc["lrx", 1] * iqr_rx),
                        note=f"{len(t) - len(pos)} counties with no T40.4 deaths left out"))
        cross_section(t, "5b T40.4 2014-20 cross-section: Poisson PML, state effects, all counties", bound, res, iqr_rx)
        for w in ("2008-10", "2017-19"):
            cross_section(p[p.window == w], f"5b context: all drug poisoning {w} cross-section, Poisson PML, state effects", bound, res, iqr_rx)
        if bound == 5:
            level_check(p, nchs, checks)
            self_check(p, checks)
            p5, cy5, t5 = p, cy, t
    # Censored Poisson: hidden counts known only to lie in 1-9, so no bound is needed. A check on the bounds, fitted here on the Mac.
    cl = "censored 1-9"
    run(p5, "lrx", [("lrx_x_after", p5.lrx * p5.after)], "main, censored Poisson", cl, res, iqr=iqr_rx, censored=True)
    ev = [(f"lrx_x_{e}", p5.lrx * (p5.ebin == e)) for e in EVENT_BINS]
    b, V, names = run(p5, "lrx", ev, "1 event time (reference e=-1), censored Poisson", cl, res, iqr=iqr_rx, censored=True)
    stat, df, pv = wald(b, V, names, ["lrx_x_e<=-3", "lrx_x_e=-2"])
    res.append(dict(model="1 event time (reference e=-1), censored Poisson", bound=cl, term="Wald test, pre-arrival bins jointly zero", b=stat, p=pv, note=f"chi-square, {df} df"))
    yrs = sorted(cy5.yr.unique())
    bins = [(f"lrx_x_since_{lab}", cy5.lrx * cy5.since.between(lo, hi)) for lab, lo, hi in YEAR_BINS]
    run_years(cy5, yrs, bins, "5a T40.4 county-year: bins of years since arrival, censored Poisson", cl, res, False, iqr_rx, censored=True)
    run_years(cy5, yrs, bins, "5a T40.4 county-year: bins, plus dispensing x year, censored Poisson", cl, res, True, iqr_rx, censored=True)
    cross_section(t5, "5b T40.4 2014-20 cross-section: censored Poisson, state effects, all counties", cl, res, iqr_rx, censored=True)
    if wcy_od:
        obins = [(f"lrx_x_since_{lab}", co5.lrx * co5.since.between(lo, hi)) for lab, lo, hi in YEAR_BINS]
        for lab, with_trend in (("5c all drug poisoning county-year: bins of years since arrival, censored Poisson", False),
                                ("5c all drug poisoning county-year: bins, plus dispensing x year, censored Poisson", True)):
            run_years(co5, sorted(co5.yr.unique()), obins, lab, cl, res, with_trend, iqr_rx, censored=True)
    for w in ("2008-10", "2017-19"):
        cross_section(p5[p5.window == w], f"5b context: all drug poisoning {w} cross-section, censored Poisson, state effects", cl, res, iqr_rx, censored=True)
    r = pd.DataFrame(res)
    r.to_csv(outdir / "county_arrival_results.csv", index=False)
    pd.DataFrame(sample).to_csv(outdir / "county_arrival_sample.csv", index=False)
    pd.DataFrame(checks).to_csv(outdir / "county_arrival_checks.csv", index=False)
    show = r[~r.term.str.contains("_x_20") & ~r.term.str.startswith("zx_x_20")]
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 70):
        print(show[["model", "bound", "term", "b", "lo", "hi", "p", "ratio", "ratio_lo", "ratio_hi", "counties", "obs", "converged"]].round(3).to_string(index=False))
        print(pd.DataFrame(sample).to_string(index=False))
        print(pd.DataFrame(checks).to_string(index=False))


def cross_section(t, label, bound, out, iqr, censored=False):
    """Within-state slope of a county death rate on log dispensing: Poisson with state effects, offset log person-years."""
    t = t[t.groupby("st").deaths.transform("sum") > 0]
    b, V, it, conv, G = fit(t, t[["lrx"]].to_numpy(float), t.st.to_numpy(), censored)
    se = np.sqrt(V[0, 0])
    out.append(dict(model=label, bound=bound, term="lrx", b=b[0], se=se, lo=b[0] - 1.96 * se, hi=b[0] + 1.96 * se, p=2 * norm.sf(abs(b[0] / se)),
                    counties=len(t), obs=len(t), states=G, iterations=it, converged=conv, iqr_log=iqr, ratio=np.exp(b[0] * iqr),
                    ratio_lo=np.exp((b[0] - 1.96 * se) * iqr), ratio_hi=np.exp((b[0] + 1.96 * se) * iqr)))


def run_years(cy, yrs, terms, label, bound, out, with_trend, iqr, censored=False):
    """County-year model: county effects, state x year effects, optionally dispensing x year, and the terms."""
    cols, names = [], []
    for st in sorted(cy.st.unique()):
        for y in yrs[1:]:
            cols.append(((cy.st == st) & (cy.yr == y)).astype(float).to_numpy())
            names.append(f"fe_st{st}_{y}")
    if with_trend:
        for y in yrs[1:]:
            cols.append((cy.lrx * (cy.yr == y)).to_numpy())
            names.append(f"lrx_x_{y}")
    for nm, col in terms:
        cols.append(np.asarray(col, dtype=float))
        names.append(nm)
    X = np.column_stack(cols)
    protect = {t for t, _ in terms}
    keep, lost = independent_columns(X, cy.fips.to_numpy(), names, protect)
    X, names = X[:, keep], [names[j] for j in keep]
    b, V, it, conv, G = fit(cy, X, cy.fips.to_numpy(), censored)
    for j, nm in enumerate(names):
        if nm in protect:
            se = np.sqrt(V[j, j])
            out.append(dict(model=label, bound=bound, term=nm, b=b[j], se=se, lo=b[j] - 1.96 * se, hi=b[j] + 1.96 * se, p=2 * norm.sf(abs(b[j] / se)),
                            counties=cy.fips.nunique(), obs=len(cy), states=G, iterations=it, converged=conv, not_identified=" ".join(lost),
                            iqr_log=iqr, ratio=np.exp(b[j] * iqr), ratio_lo=np.exp((b[j] - 1.96 * se) * iqr), ratio_hi=np.exp((b[j] + 1.96 * se) * iqr),
                            note="2025 provisional"))


def level_check(p, nchs, checks):
    """NCHS model-based rates as a level check only: Spearman of county crude rates (hidden counts at 5) against the model's rates."""
    n = pd.read_csv(nchs / "nchs_county_model_2003_2021.csv", usecols=["FIPS", "Year", "Model-based Death Rate"])
    n["fips"] = n.FIPS.astype(str).str.zfill(5)
    for w, y0 in WINDOWS.items():
        m = n[n.Year.between(y0, y0 + 2)].groupby("fips")["Model-based Death Rate"].mean()
        k = p[p.window == w].assign(crude=lambda d: d.deaths / d.population * 1e5)
        k = k.assign(model=k.fips.map(m)).dropna(subset=["model"])
        checks.append(dict(check=f"NCHS level check {w}: Spearman of crude rate (hidden at 5) with NCHS model rate, counties {len(k)}", value=spearmanr(k.crude, k.model)[0]))
        within = k.assign(rc=k.groupby("st").crude.rank(pct=True), rm=k.groupby("st").model.rank(pct=True))
        checks.append(dict(check=f"NCHS level check {w}: Spearman of within-state percentile ranks", value=spearmanr(within.rc, within.rm)[0]))


def self_check(p, checks):
    """The profiled fit against statsmodels' Poisson GLM with every county dummy, on six states (hidden counts at 5)."""
    k = p[p.st.isin([18, 21, 39, 47, 51, 54])].copy()
    k["lrx_x_after"] = k.lrx * k.after
    X, names = design(k, "lrx", [("lrx_x_after", k.lrx_x_after)])
    keep, _ = independent_columns(X, k.fips.to_numpy(), names, {"lrx_x_after"})
    X, names = X[:, keep], [names[j] for j in keep]
    b, V, it, conv, G = fit_ppml(k.deaths.to_numpy(float), np.log(k.population.to_numpy(float)), X, k.fips.to_numpy(), k.st.to_numpy())
    D = pd.get_dummies(k.fips, dtype=float).to_numpy()
    g = sm.GLM(k.deaths.to_numpy(float), np.column_stack([X, D]), family=sm.families.Poisson(), offset=np.log(k.population.to_numpy(float))).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(k.st)[0], "use_correction": False}, maxiter=200, tol=1e-12)
    j = names.index("lrx_x_after")
    checks.append(dict(check="self-check against statsmodels GLM, six states: coefficient here", value=b[j]))
    checks.append(dict(check="self-check: statsmodels coefficient", value=g.params[j]))
    checks.append(dict(check="self-check: clustered SE here, without the G/(G-1) factor", value=np.sqrt(V[j, j] * (G - 1) / G)))
    checks.append(dict(check="self-check: statsmodels clustered SE, no correction", value=g.bse[j]))
    bc, Vc, _, _, _ = fit_censored(k.deaths.to_numpy(float), np.zeros(len(k), bool), np.log(k.population.to_numpy(float)), X, k.fips.to_numpy(), k.st.to_numpy())
    checks.append(dict(check="self-check: censored-Poisson code with nothing hidden, coefficient (should equal the first)", value=bc[j]))
    checks.append(dict(check="self-check: censored-Poisson code with nothing hidden, clustered SE without the G/(G-1) factor (should equal the third)",
                       value=np.sqrt(Vc[j, j] * (G - 1) / G)))


if __name__ == "__main__":
    if len(sys.argv) not in (7, 8):
        sys.exit(__doc__)
    main(*sys.argv[1:])
