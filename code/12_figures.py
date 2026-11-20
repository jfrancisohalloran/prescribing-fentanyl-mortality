"""Figures 1-4 for the paper (Lancet Public Health draft), from the outputs of code/06, 07 and 09 and the WHO file from code/02.

Palette from the dataviz reference, validated: supply status uses categorical slots 1-3 (blue, orange, aqua) with marker shape as a
second cue; countries with no known supply are light grey, behind. Prescribing tertiles use the ordinal blue ramp (steps 250, 450, 650).

Labels are placed by measurement, not by hand-set offsets, which had let names run over lines and dots.
place_labels puts each point's name at the nearest spot that clears every marker, line, leader and earlier name in its panel, and
only where it is clearly nearer its own point than any other; when it has to sit further away it gets a thin leader. column_labels stacks
names in one column with leaders: for the ends of lines, and for the cluster of heavy prescribers in figure 1A. check_overlaps then
measures every name, note, legend and leader in the finished figures; if any covers a marker, a line or another label, the script names
it and exits with an error after writing the files.

Numbers follow the Lancet's house style, as the text does: tick labels get midline decimal points (0·5) and en-dash minus signs (–0·2),
set in save() by wrapping each axis's own formatter. Names, legends and notes print no decimals.

Writes PNG (300 dpi) and PDF to paper/figures/.

  python3 code/12_figures.py <out folder> <figures folder>
"""
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.text import Annotation, Text

INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b68", "#e6e5e1", "#ffffff"
STATUS = {"established": ("#2a78d6", "o", "Established illicit fentanyl market by 2019"),
          "episodic": ("#eb6834", "s", "Episodic market"),
          "diverted": ("#1baf7a", "D", "Diverted pharmaceutical fentanyl, or unclear"),
          "none": ("#c9c8c3", "o", "No illicit market known")}
TERTILE = ["#86b6ef", "#2a78d6", "#104281"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "figure.facecolor": SURFACE})
LABEL = 7  # point size of the names inside the panels
PROBLEMS = []  # overlaps found by check_overlaps, and names place_labels could not fit


def group(status):
    return {"established": "established", "episodic": "episodic", "diverted only": "diverted", "unclear": "diverted"}.get(status, "none")


_DOT = re.compile(r"(?<=\d)\.(?=\d)")
_MINUS = re.compile(r"(?<![\w.])[-−](?=\d)")


def lancet(s):
    """A label in the Lancet's style: 0·5 and –0·2. Mathtext labels, such as the powers of ten on log axes, pass unchanged."""
    return s if not s or s.startswith("$") else _MINUS.sub("–", _DOT.sub("·", s))


class Midline(mticker.Formatter):
    """An axis's own tick formatter, its labels passed through lancet()."""

    def __init__(self, base):
        self.base = base

    def set_axis(self, axis):
        super().set_axis(axis)
        self.base.set_axis(axis)

    def set_locs(self, locs):
        self.base.set_locs(locs)

    def __call__(self, x, pos=None):
        return lancet(self.base(x, pos))

    def format_ticks(self, values):
        return [lancet(s) for s in self.base.format_ticks(values)]

    def get_offset(self):
        return lancet(self.base.get_offset())


def midline(fig):
    """Every tick label in the figure in the Lancet's style, once the scales and ticks are final."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for get, put in ((axis.get_major_formatter, axis.set_major_formatter), (axis.get_minor_formatter, axis.set_minor_formatter)):
                if not isinstance(get(), (mticker.NullFormatter, Midline)):
                    put(Midline(get()))


def save(fig, folder, name):
    midline(fig)
    for ext in ("png", "pdf"):
        fig.savefig(pathlib.Path(folder) / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# Label placement ---------------------------------------------------------------------------------------------------------------------
# Geometry is measured in display pixels once the layout is final. Offsets go to matplotlib in points, so it holds at any resolution.

def _k(fig):
    """Display pixels per point."""
    return fig.dpi / 72.0


def _shapes(ax):
    """Markers in ax as circles (x, y, r) and drawn lines as segments (x0, y0, x1, y1, half-width), in display pixels."""
    k = _k(ax.figure)
    circles, segs = [], []

    def add_path(xy, width):
        xy = np.asarray(xy, float)
        ok = np.isfinite(xy).all(axis=1)
        segs.extend((*xy[i], *xy[i + 1], width * k / 2) for i in range(len(xy) - 1) if ok[i] and ok[i + 1])

    for c in ax.collections:
        if not c.get_visible():
            continue
        if isinstance(c, PathCollection):  # scatter markers
            xy = c.get_offset_transform().transform(np.asarray(c.get_offsets(), float))
            sizes = np.resize(np.asarray(c.get_sizes(), float) if len(c.get_sizes()) else [36.0], len(xy))
            lw = np.asarray(c.get_linewidths(), float)
            r = (np.sqrt(sizes) + (lw[0] if len(lw) else 0)) / 2 * k
            circles.extend((x, y, rr) for (x, y), rr in zip(xy, r) if np.isfinite(x) and np.isfinite(y))
        elif isinstance(c, LineCollection):  # error bars
            lw = np.asarray(c.get_linewidths(), float)
            for seg in c.get_segments():
                add_path(c.get_transform().transform(seg), lw[0] if len(lw) else 1.0)
    for line in ax.lines:
        if not line.get_visible():
            continue
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), float))
        if line.get_linestyle() not in ("None", "none", "", " "):
            add_path(xy, line.get_linewidth())
        if line.get_marker() not in (None, "None", "none", "", " "):
            r = (line.get_markersize() + line.get_markeredgewidth()) / 2 * k
            circles.extend((x, y, r) for x, y in xy if np.isfinite(x) and np.isfinite(y))
    return np.array(circles, float).reshape(-1, 3), np.array(segs, float).reshape(-1, 5)


def _segs_cross(segs, box):
    """Which segments, widened by their half-widths, pass through box = (x0, y0, x1, y1) (Liang-Barsky clipping)."""
    if not len(segs):
        return np.zeros(0, bool)
    x0, y0, x1, y1, hw = segs.T
    dx, dy = x1 - x0, y1 - y0
    t_in, t_out, ok = np.zeros(len(segs)), np.ones(len(segs)), np.ones(len(segs), bool)
    for p, q in ((-dx, x0 - (box[0] - hw)), (dx, box[2] + hw - x0), (-dy, y0 - (box[1] - hw)), (dy, box[3] + hw - y0)):
        flat = p == 0
        ok &= ~(flat & (q < 0))
        t = q / np.where(flat, 1.0, p)
        t_in = np.where(~flat & (p < 0), np.maximum(t_in, t), t_in)
        t_out = np.where(~flat & (p > 0), np.minimum(t_out, t), t_out)
    return ok & (t_in <= t_out)


def _edge_gaps(circles, box):
    """Distance from each circle's edge to the box; negative where they overlap."""
    if not len(circles):
        return np.zeros(0)
    cx, cy, r = circles.T
    return np.hypot(cx - np.clip(cx, box[0], box[2]), cy - np.clip(cy, box[1], box[3])) - r


def _circles_hit(circles, box):
    return _edge_gaps(circles, box) < 0


def _seg_near_circles(seg, circles, pad):
    """Which circles, widened by pad, the segment ((x0, y0), (x1, y1)) passes through."""
    if not len(circles):
        return np.zeros(0, bool)
    (ax_, ay_), (bx, by) = seg
    cx, cy, r = circles.T
    vx, vy = bx - ax_, by - ay_
    t = np.clip(((cx - ax_) * vx + (cy - ay_) * vy) / max(vx * vx + vy * vy, 1e-9), 0, 1)
    return np.hypot(cx - ax_ - t * vx, cy - ay_ - t * vy) < r + pad


def _boxes_hit(box, boxes):
    return any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes)


def _grow(box, d):
    return (box[0] - d, box[1] - d, box[2] + d, box[3] + d)


def _text_size(fig, text, size):
    t = fig.text(0, 0, text, fontsize=size)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width, bb.height


def _words(t):
    """The extent of a name's words alone: an annotation's own extent includes its leader."""
    return Text.get_window_extent(t)


def _leaders(ax):
    """The leaders drawn in ax: (annotation, (x0, y0, x1, y1)) from the named point to where the leader meets the words."""
    out = []
    for t in ax.texts:
        if isinstance(t, Annotation) and t.arrow_patch is not None and t.get_visible():
            bb = _words(t)
            rx, ry = t.arrowprops.get("relpos", (0.5, 0.5))
            px, py = ax.transData.transform(t.xy)
            out.append((t, (px, py, bb.x0 + rx * bb.width, bb.y0 + ry * bb.height)))
    return out


# Eight directions from a point, each with the text alignment that puts the name's near side or corner there.
_DIRECTIONS = (((1, 0), "left", "center"), ((-1, 0), "right", "center"), ((0.7, 0.7), "left", "bottom"), ((-0.7, 0.7), "right", "bottom"),
               ((0.7, -0.7), "left", "top"), ((-0.7, -0.7), "right", "top"), ((0, 1), "center", "bottom"), ((0, -1), "center", "top"))


def place_labels(ax, items, size=LABEL, rings=(0, 2, 4, 7, 10, 14, 19, 25, 32, 40, 50, 60, 72), lead_from=4):
    """Name points, each at the nearest spot that clears every marker, line, leader and earlier name in the panel.

    items: (x, y, text) in data coordinates. Rings of growing distance (points beyond the marker's edge) are tried around each point,
    eight directions to a ring. A name without a leader must sit clearly nearer its own marker than any other, so it can't be read as
    another point's; a name set `lead_from` points or more away gets a thin leader from its marker, which must not cross another marker
    or name. Points in the most crowded places are named first. Call once the layout is final.
    """
    fig = ax.figure
    fig.canvas.draw()
    k = _k(fig)
    circles, segs = _shapes(ax)
    for _, seg in _leaders(ax):  # leaders already drawn, say by column_labels
        segs = np.vstack([segs, [*seg, 0.25 * k]])
    frame = _grow(ax.get_window_extent().extents, -2 * k)
    placed = [_words(t).extents for t in ax.texts if t.get_visible() and t.get_text().strip()]
    todo = []
    for x, y, text in items:
        px, py = ax.transData.transform((x, y))
        d = np.hypot(circles[:, 0] - px, circles[:, 1] - py)
        todo.append((int((d < 30 * k).sum()), x, y, text, px, py, int(d.argmin())))
    for _, x, y, text, px, py, own in sorted(todo, key=lambda p: -p[0]):
        r0 = circles[own, 2]
        d = np.hypot(circles[:, 0] - px, circles[:, 1] - py)
        others = circles[d >= 1.0]  # every marker but this point's own (drawn twice where it is redrawn on top)
        apart = circles[d > r0 + circles[:, 2]]  # markers not piled on this one
        w, h = _text_size(fig, text, size)
        best = None
        for ring in rings:
            for (ux, uy), ha, va in _DIRECTIONS:
                dist = r0 + (1.5 + ring) * k
                anchor = (px + ux * dist, py + uy * dist)
                x0 = {"left": anchor[0], "right": anchor[0] - w, "center": anchor[0] - w / 2}[ha]
                y0 = {"bottom": anchor[1], "top": anchor[1] - h, "center": anchor[1] - h / 2}[va]
                box = (x0, y0, x0 + w, y0 + h)
                if box[0] < frame[0] or box[1] < frame[1] or box[2] > frame[2] or box[3] > frame[3]:
                    continue
                padded = _grow(box, 1.0 * k)
                if _circles_hit(circles, padded).any() or _segs_cross(segs, padded).any() or _boxes_hit(padded, placed):
                    continue
                if ring < lead_from:  # no leader: clearly nearer its own marker than any other
                    if len(others) and _edge_gaps(others, box).min() < _edge_gaps(circles[own:own + 1], box)[0] + 3 * k:
                        continue
                    lead = None
                else:  # a leader from the marker's edge to the nearest point of the name
                    end = (float(np.clip(px, box[0], box[2])), float(np.clip(py, box[1], box[3])))
                    u = np.subtract(end, (px, py)) / max(np.hypot(end[0] - px, end[1] - py), 1e-9)
                    lead = ((px + u[0] * r0, py + u[1] * r0), end)
                    seg = np.array([[*lead[0], *lead[1], 0.0]])
                    if _seg_near_circles(lead, apart, 0.5 * k).any() or any(_segs_cross(seg, _grow(b, 0.5 * k))[0] for b in placed):
                        continue
                best = (anchor[0] - px, anchor[1] - py, ha, va, box, lead)
                break
            if best:
                break
        if best is None:
            PROBLEMS.append(f"no free spot for the name '{text}'")
            best = (r0 + 1.5 * k, 0.0, "left", "center", (px, py, px, py), None)
        dx, dy, ha, va, box, lead = best
        arrow = None
        if lead:  # drawn exactly as checked: from the marker's edge to that point of the name
            arrow = dict(arrowstyle="-", color=MUTED, lw=0.5, patchA=None, shrinkA=0, shrinkB=r0 / k,
                         relpos=((lead[1][0] - box[0]) / w, (lead[1][1] - box[1]) / h))
        ax.annotate(text, (x, y), xytext=(dx / k, dy / k), textcoords="offset points", ha=ha, va=va, fontsize=size, color=INK,
                    arrowprops=arrow)
        placed.append(box)
        if lead:
            segs = np.vstack([segs, [*lead[0], *lead[1], 0.25 * k]])


def _dodge(targets, gap, lo, hi):
    """Positions as near as possible (least squares) to the sorted targets, each at least `gap` above the one before, within [lo, hi]."""
    n = len(targets)
    q = np.asarray(targets, float) - gap * np.arange(n)
    blocks = []  # pool adjacent violators: [sum, count]
    for v in q:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    fit = np.concatenate([np.full(c, s / c) for s, c in blocks])
    return np.clip(fit, lo, max(lo, hi - gap * (n - 1))) + gap * np.arange(n)


def column_labels(ax, items, size=LABEL, gap=2.0, column=8):
    """Name points or line ends in one column, `column` points right of the rightmost of them, each name as near its point's height as
    the others allow and joined to it by a thin leader. For line ends, and for clusters too tight for names beside each point.
    items: (x, y, text) in data coordinates. Call once the layout is final."""
    fig = ax.figure
    fig.canvas.draw()
    k = _k(fig)
    circles, _ = _shapes(ax)

    def radius(px, py):  # of a marker drawn at the point, if there is one
        if not len(circles):
            return 0.0
        d = np.hypot(circles[:, 0] - px, circles[:, 1] - py)
        return float(circles[d.argmin(), 2]) if d.min() < 1 else 0.0

    pts = sorted(((*ax.transData.transform((x, y)), x, y, text) for x, y, text in items), key=lambda p: p[1])
    h = max(_text_size(fig, p[4], size)[1] for p in pts) + gap * k
    frame = ax.get_window_extent()
    ys = _dodge([p[1] for p in pts], h, frame.y0 + h / 2, frame.y1 - h / 2)
    xcol = max(p[0] + radius(p[0], p[1]) for p in pts) + column * k
    for (px, py, x, y, text), yy in zip(pts, ys):
        ax.annotate(text, (x, y), xytext=((xcol - px) / k, (yy - py) / k), textcoords="offset points", ha="left", va="center",
                    fontsize=size, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, relpos=(0, 0.5), patchA=None,
                                                              shrinkA=1, shrinkB=radius(px, py) / k + 0.5))


def check_overlaps(fig, name):
    """Record every name, note or legend in fig that covers a marker, a line or another label, and every leader that crosses a marker
    or another name. (A leader may cross a line: a line end that stops before the others needs one.)"""
    fig.canvas.draw()
    k = _k(fig)
    for ax in fig.axes:
        circles, segs = _shapes(ax)
        things = [(t.get_text().replace("\n", " "), _words(t), t) for t in ax.texts if t.get_visible() and t.get_text().strip()]
        if ax.get_legend() is not None:
            things.append(("legend", ax.get_legend().get_window_extent(), None))
        boxes = [_grow(bb.extents, -0.5 * k) for _, bb, _ in things]  # half a point's tolerance
        for i, ((label, _, _), b) in enumerate(zip(things, boxes)):
            hits = [what for what, hit in (("a marker", _circles_hit(circles, b).any()), ("a line", _segs_cross(segs, b).any()),
                                           ("another label", _boxes_hit(b, boxes[:i] + boxes[i + 1:]))) if hit]
            if hits:
                PROBLEMS.append(f"{name}: '{label}' covers {' and '.join(hits)}")
        for t, (px, py, ex, ey) in _leaders(ax):
            d = np.hypot(circles[:, 0] - px, circles[:, 1] - py) if len(circles) else np.zeros(0)
            r0 = float(circles[d.argmin(), 2]) if len(d) and d.min() < 1 else 0.0
            apart = circles[d > r0 + circles[:, 2]] if len(d) else circles
            u = np.subtract((ex, ey), (px, py)) / max(np.hypot(ex - px, ey - py), 1e-9)
            seg = ((px + u[0] * r0, py + u[1] * r0), (ex, ey))
            if _seg_near_circles(seg, apart, 0).any():
                PROBLEMS.append(f"{name}: the leader to '{t.get_text()}' crosses a marker")
            if any(_segs_cross(np.array([[*seg[0], *seg[1], 0.0]]), b)[0] for (_, _, o), b in zip(things, boxes) if o is not t):
                PROBLEMS.append(f"{name}: the leader to '{t.get_text()}' crosses another label")


# Figures ------------------------------------------------------------------------------------------------------------------------------

def figure1(out, folder):
    t = pd.read_csv(pathlib.Path(out) / "tests_countries.csv")
    t["g"] = t.status.map(group)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    panels = (("narc_1921", "Narcotic-poisoning deaths per 100 000", True), ("narc_change", "Log ratio, 2019–21 to 2010–12", False))
    for ax, (col, ylab, logy) in zip(axes, panels):
        for g in ("none", "diverted", "episodic", "established"):
            k = t[t.g == g]
            colour, marker, lab = STATUS[g]
            # no supply known: open circles, so the figure reads in greyscale against the filled circles of established markets
            paint = dict(facecolors="none", edgecolors="#8f8e89", linewidths=0.7) if g == "none" else dict(c=colour, edgecolors=SURFACE, linewidths=0.8)
            ax.scatter(k.rx_pre, k[col], s=26 if g != "none" else 18, marker=marker, zorder=3 if g != "none" else 2, label=lab, **paint)
        ax.set_xscale("log")
        ax.set_xlim(4, 2.5e5)  # room on the right for the names of the heavy prescribers
        if logy:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_xlabel("Opioid analgesic consumption, 2000–11\n(S-DDD per million inhabitants per day, log scale)")
        ax.set_ylabel(ylab)
        grid(ax, "both")
    axes[0].set_title("A  Level, 2019–21", loc="left", fontsize=9, color=INK, fontweight="bold")
    axes[1].set_title("B  Change, 2010–12 to 2019–21", loc="left", fontsize=9, color=INK, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    fig.tight_layout()
    short = {"United States of America": "USA", "United Kingdom": "UK"}
    heavy = ["Denmark", "Australia", "Switzerland", "Belgium", "Germany", "Austria"]  # too close together in A for names beside them
    column_labels(axes[0], [(r.rx_pre, r.narc_1921, r.country) for _, r in t[t.country.isin(heavy)].iterrows()])
    named = ({"United States of America", "Canada", "Estonia", "Latvia", "United Kingdom"},
             {"United States of America", "Canada", "Estonia", "Latvia", "Germany", "Austria", "United Kingdom"})
    renamed = ({}, {"Austria": "Austria (coding change)", "United Kingdom": "UK, Spain"})  # in B, Spain's point lies under the UK's
    for ax, (col, _, _), names, new in zip(axes, panels, named, renamed):
        place_labels(ax, [(r.rx_pre, r[col], new.get(r.country, short.get(r.country, r.country))) for _, r in t[t.country.isin(names)].iterrows()])
    check_overlaps(fig, "figure 1")
    save(fig, folder, "figure1_countries")


def figure2(out, folder):
    w = pd.read_csv(pathlib.Path(out) / "who_drug_deaths_country_year.csv")
    w["Year"] = pd.to_numeric(w.Year, errors="coerce")
    w = w[w.Year.between(2000, 2021)]
    est = {"United States of America": (2013, "USA"), "Canada": (2012, "Canada"), "Estonia": (2001, "Estonia"), "Latvia": (2012, "Latvia")}
    heavy = {"Germany": "Germany", "Belgium": "Belgium", "Austria": "Austria", "Denmark": "Denmark", "Switzerland": "Switzerland", "Australia": "Australia"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for name, (arr, lab) in est.items():
        k = w[w.country_name == name].sort_values("Year")
        axes[0].plot(k.Year, k.narcotics_per_100k, color=STATUS["established"][0], linewidth=1.6)
        a = k[k.Year == arr]
        if len(a):
            axes[0].plot(a.Year, a.narcotics_per_100k, marker="o", color=STATUS["established"][0], markersize=5, markeredgecolor=SURFACE)
    for name, lab in heavy.items():
        k = w[w.country_name == name].sort_values("Year")
        diverted = name in ("Germany", "Austria", "Belgium", "Australia")  # diverted or unclear: solid; none known: dashed, for greyscale
        axes[1].plot(k.Year, k.narcotics_per_100k, color=STATUS["diverted"][0] if diverted else "#8f8e89", linewidth=1.4,
                     linestyle="-" if diverted else "--")
    axes[0].set_title("A  Established illicit fentanyl markets", loc="left", fontsize=9, color=INK, fontweight="bold")
    axes[1].set_title("B  Heavy prescribers without an illicit market", loc="left", fontsize=9, color=INK, fontweight="bold")
    axes[0].set_ylabel("Narcotic-poisoning deaths per 100 000")
    for ax in axes:
        ax.set_xlim(2000, 2027.5)
        grid(ax)
    fig.tight_layout()
    for ax, names in ((axes[0], {n: v[1] for n, v in est.items()}), (axes[1], heavy)):
        items = []
        for name, lab in names.items():
            last = w[w.country_name == name].dropna(subset=["narcotics_per_100k"]).sort_values("Year").iloc[-1]
            items.append((last.Year, last.narcotics_per_100k, lab))
        column_labels(ax, items, column=14)
    check_overlaps(fig, "figure 2")
    save(fig, folder, "figure2_trajectories")


def figure3(out, folder):
    s = pd.read_csv(pathlib.Path(out) / "us_states_frame.csv")
    r = pd.read_csv(pathlib.Path(out) / "us_states_results.csv")
    s["tertile"] = pd.qcut(s.rx, 3, labels=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    names = ["Lowest third", "Middle third", "Highest third"]
    shapes = ["o", "s", "^"]  # shape as well as shade, so the thirds read in greyscale
    for q in range(3):
        k = s[s.tertile == q]
        axes[0].scatter(k.arrival, k.t404_2014_20, s=26, c=TERTILE[q], marker=shapes[q], edgecolors=SURFACE, linewidths=0.8,
                        label=f"{names[q]} of prescribing, 2009–11", zorder=3)
    named = s[s.state.isin(["WV", "DC", "NY", "VT", "OK", "AR", "MS", "CA", "TX"])]
    for q in range(3):  # the named states again, on top, so each name points at a visible marker
        k = named[named.tertile == q]
        axes[0].scatter(k.arrival, k.t404_2014_20, s=26, c=TERTILE[q], marker=shapes[q], edgecolors=SURFACE, linewidths=0.8, zorder=4)
    axes[0].set_yscale("log")
    axes[0].set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    axes[0].set_xlabel("Illicit fentanyl arrival (year)")
    axes[0].set_ylabel("Synthetic-opioid deaths per 100 000\na year, 2014–20 (log scale)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=1)
    grid(axes[0], "both")
    k = r[(r.bound == 5) & r.model.str.startswith("P2 gradient across states") & (r.term == "lrx")].copy()
    k["window"] = k.model.str.extract(r"rate (\d{4}-\d{2})$")[0].str.replace("-", "–")
    x = np.arange(len(k))
    axes[1].errorbar(x, k.b, yerr=1.96 * k.se, fmt="o", color=STATUS["established"][0], ecolor=STATUS["established"][0], elinewidth=1.4,
                     capsize=0, markersize=5, markeredgecolor=SURFACE)
    axes[1].axhline(0, color=MUTED, linewidth=0.6)
    axes[1].set_xticks(x, k.window)
    axes[1].set_xlabel("Overdose deaths in")
    axes[1].set_ylabel("Slope on log prescribing (95% CI)")
    grid(axes[1])
    axes[0].set_title("A  Arrival year and synthetic-opioid deaths", loc="left", fontsize=9, color=INK, fontweight="bold")
    axes[1].set_title("B  Slope on prescribing, by window", loc="left", fontsize=9, color=INK, fontweight="bold")
    fig.tight_layout()
    place_labels(axes[0], [(row.arrival, row.t404_2014_20, row.state) for _, row in named.iterrows()])
    check_overlaps(fig, "figure 3")
    save(fig, folder, "figure3_us_states")


def figure4(out, folder):
    r = pd.read_csv(pathlib.Path(out) / "alpert_rerun_results.csv")
    r = r[(r.bound == 5) & r.term.str.startswith("trip_")].copy()
    order = ["trip_1985", "trip_1990", "trip_1999", "trip_2002", "trip_2005", "trip_2008", "trip_2011", "trip_2014", "trip_2017"]
    labels = ["1985–89", "1990–94", "1999–01", "2002–04", "2005–07", "2008–10", "2011–13", "2014–16", "2017–19"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    # filled circles on solid bars against open squares on dashed bars, so the two series read in greyscale
    for i, (model, colour, lab, off, marker, face, style) in enumerate((
            ("gap by window, relative to 1995-98", "#2a78d6", "As in Alpert et al.", -0.12, "o", "#2a78d6", "-"),
            ("gap by window, arrival held", "#eb6834", "Adjusted for arrival date", 0.12, "s", SURFACE, "--"))):
        k = r[r.model == model].set_index("term").reindex(order)
        x = np.arange(len(order)) + off
        bars = ax.errorbar(x, k.b, yerr=1.96 * k.se, fmt=marker, color=colour, ecolor=colour, elinewidth=1.2, capsize=0, markersize=4.5,
                           markerfacecolor=face, markeredgecolor=colour if face == SURFACE else SURFACE, label=lab)
        bars[-1][0].set_linestyle(style)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.axvline(1.5, color=GRID, linewidth=6, zorder=0)
    ax.text(1.5, 1.02, "1995–98\nOxyContin launched 1996", transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", va="bottom", fontsize=6.5, color=MUTED)  # above the panel, over the reference window's band
    ax.set_xticks(np.arange(len(order)), labels, rotation=45, ha="right")
    ax.set_ylabel("Triplicate states' overdose rate,\nlog difference from 1995–98 (95% CI)")
    ax.legend(loc="upper right", fontsize=7)  # clear of the band and the bars
    grid(ax)
    fig.tight_layout()
    check_overlaps(fig, "figure 4")
    save(fig, folder, "figure4_triplicate")


def main(out, folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    figure1(out, folder)
    figure2(out, folder)
    figure3(out, folder)
    figure4(out, folder)
    print("written:", sorted(p.name for p in pathlib.Path(folder).glob("*.png")))
    if PROBLEMS:
        print("Labels to fix:", *PROBLEMS, sep="\n  ")
        sys.exit(1)
    print("checked: no name, note or legend covers a marker, a line or another label")


if __name__ == "__main__":
    main(*sys.argv[1:3])
