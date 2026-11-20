"""Extract INCB's table of average opioid consumption by country (S-DDD per million inhabitants per day) from five technical reports.

Editions and windows: 2003 (2000-02, Table XI.1), 2006 (2003-05), 2009 (2006-08), 2012 (2009-11) and 2015 (2012-14) (Table XIV.1, or XIV.1.a).
Every edition has 11 value columns ending in the total. 2003-2009: codeine, fentanyl, hydrocodone, hydromorphone, methadone, morphine,
oxycodone, pethidine, tilidine, others, total. 2012-2015: buprenorphine first, no tilidine. The total excludes buprenorphine in every edition.

A row is a line ending in 11 numeric cells (cells are separated by two or more spaces; "—" is zero). Its name is the text before them,
joined to any name lines just above, and the English name is the part before the first " — ". Each row's columns are checked against
its total (buprenorphine left out), and rows that don't add up are flagged, not dropped.
Analgesic consumption = total - methadone, the exposure.

  python3 code/04_incb_xiv.py <incb_narcotic_drugs date folder> <out csv>
"""
import pathlib
import re
import subprocess
import sys

import pandas as pd

OLD = ["codeine", "fentanyl", "hydrocodone", "hydromorphone", "methadone", "morphine", "oxycodone", "pethidine", "tilidine", "others", "total"]
NEW = ["buprenorphine", "codeine", "fentanyl", "hydrocodone", "hydromorphone", "methadone", "morphine", "oxycodone", "pethidine", "others", "total"]
EDITIONS = {  # file, window, columns, a line that starts the table, a line that ends it
    "2003": ("2003_full_report.pdf", "2000-2002", OLD, r"1\. Average consumption of narcotic drugs, in defined daily doses", r"2\. Global|TABLE XII|TABLEAU XII"),
    "2006": ("2006_full_report.pdf", "2003-2005", OLD, r"Table XIV\.1\.\s+Levels of consumption", r"Table XIV\.2|Tableau XIV\.2"),
    "2009": ("2009_part4_tables.pdf", "2006-2008", OLD, r"Table XIV\.1\.\s+Levels of consumption", r"Table XIV\.2|Tableau XIV\.2"),
    "2012": ("2012_part4_tables.pdf", "2009-2011", NEW, r"Table XIV\.1\.a\. All countries", r"Table XIV\.1\.b|Tableau XIV\.1\.b"),
    "2015": ("2015_full_report.pdf", "2012-2014", NEW, r"Table XIV\.1\.a\. All countries", r"Table XIV\.1\.b|Tableau XIV\.1\.b"),
}
NUM = re.compile(r"^(\d{1,3}(?: \d{3})*(?:\.\d+)?|—|-|<<|\.\.|\.)$")
HEADER = re.compile(r"Country or non|Pays ou territoire|País o territorio|Codeine|Codéine|Codeína|Fentanyl|Fentanil|Ranking|Classement|Clasificación|"
                    r"Table|Tableau|Cuadro|For the explanatory|defined daily doses|doses quotidiennes|dosis diarias|per million|par million|por millón", re.I)

JUNK = re.compile(r"Part four|Quatrième partie|Cuarta parte|Tables — Tableaux — Cuadros")


def value(tok):
    if tok in ("—", "-"):
        return 0.0
    if tok == "<<":
        return 0.25  # INCB's mark for less than 0.5
    if tok in ("..", "."):
        return float("nan")
    return float(tok.replace(" ", ""))


def parse(text, cols, start, end):
    lines = text.splitlines()
    def has_data(i):  # the contents page names the table too; the real one has rows soon after its title
        return any(len(re.findall(r"(?:^|\s{2,})(?:\d{1,3}(?: \d{3})*|—)(?=\s{2,}|$)", l)) >= len(cols) for l in lines[i:i + 60])
    i0 = next(i for i, l in enumerate(lines) if re.search(start, l) and has_data(i))
    i1 = next((i for i in range(i0 + 5, len(lines)) if re.search(end, lines[i])), len(lines))
    rows, name_buf, missed = [], [], []
    for l in lines[i0:i1]:
        l = JUNK.sub("  ", l)  # running page headers can share a line with a row
        if not l.strip():
            continue
        if HEADER.search(l):
            name_buf = []
            continue
        cells = [c.strip() for c in re.split(r"\s{2,}", l.strip()) if c.strip()]
        k = 0
        while k < len(cells) and NUM.match(cells[-1 - k]):
            k += 1
        if k >= len(cols):
            vals = cells[len(cells) - len(cols):]
            left = cells[:len(cells) - len(cols)]
            name = " ".join(name_buf + left)
            name_buf = []
            rows.append((name, [value(v) for v in vals]))
        else:
            if k >= 5:
                missed.append(l.strip())
            name_buf.append(" ".join(cells))
            name_buf = name_buf[-3:]
    return rows, missed


ALIAS = {"Swaziland": "Eswatini", "Cape Verde": "Cabo Verde", "Czech Republic": "Czechia", "Turkey": "Türkiye", "Netherlands": "Netherlands (Kingdom of the)",
         "The former Yugoslav Republic of Macedonia": "North Macedonia", "The former Yugoslav Rep. of Macedonia": "North Macedonia",
         "Hong Kong SAR of China": "China, Hong Kong SAR", "Macao SAR of China": "China, Macao SAR", "Dem. Rep. of the Congo": "Democratic Republic of the Congo",
         "Lao People's Dem. Rep.": "Lao People's Democratic Republic", "Libyan Arab Jamahiriya": "Libya", "Bolivia": "Bolivia (Plurinational State of)",
         "Venezuela": "Venezuela (Bolivarian Rep. of)", "Dem. People's Rep. of Korea": "Democratic People's Republic of Korea",
         "Dem. People's Republic of Korea": "Democratic People's Republic of Korea", "United States": "United States of America",
         "Islamic Republic of Iran": "Iran (Islamic Republic of)", "The form. Yug. Rep. of Macedonia": "North Macedonia", "Lao People's Dem. Rep": "Lao People's Democratic Republic",
         "Yugoslavia": "Serbia and Montenegro", "République-Unie de Tanzanie": "United Republic of Tanzania", "Îles Falkland (Malvinas)": "Falkland Islands (Malvinas)"}


def standard(raw, ref):
    """INCB's English name as the 2025 Excel file spells it, where it can be matched; otherwise the cleaned English name."""
    name = re.sub(r"\s+", " ", JUNK.sub(" ", raw)).replace("’", "'").strip()
    name = re.sub(r"^(\d+ )+", "", name)  # the global rank, where the edition has one (sometimes printed twice)
    name = re.split(r" —(?: |$)", name)[0].strip(" .")
    for cand in (name, re.sub(r"[a-z]$", "", name), re.sub(r"[a-z],[a-z]$", "", name), re.sub(r"[a-z]{2}$", "", name)):  # footnote letters stuck to the name
        cand = ALIAS.get(cand, cand)
        if cand in ref:
            return cand
    starts = [r for r in ref if name.startswith(r + " ")]  # the French name ran on without its dash
    return max(starts, key=len) if starts else name


def main(folder, out):
    folder, frames = pathlib.Path(folder), []
    ref = set(pd.read_excel(folder / "2025_2_Consumption_in_S-DDD_2015-2024.xlsx", sheet_name="Data", header=0).iloc[:, 1].dropna().astype(str))
    for ed, (f, window, cols, start, end) in EDITIONS.items():
        text = subprocess.run(["pdftotext", "-layout", str(folder / f), "-"], capture_output=True, text=True).stdout
        rows, missed = parse(text, cols, start, end)
        for l in missed:
            print(f"  {ed}: a line with numbers that isn't a row: {l[:150]}")
        d = pd.DataFrame([dict(zip(cols, v), raw_name=n) for n, v in rows])
        d.insert(0, "country", d.raw_name.map(lambda n: standard(n, ref)))
        d.insert(1, "in_2025_file", d.country.isin(ref))
        d.insert(0, "window", window)
        d.insert(0, "edition", ed)
        parts = [c for c in cols if c not in ("total", "buprenorphine")]
        d["sum_check"] = d[parts].sum(axis=1, min_count=1) - d.total
        d["adds_up"] = d.sum_check.abs() <= 2 + 0.002 * d.total  # rounding in the printed cells
        d["analgesic"] = d.total - d.methadone
        frames.append(d)
        print(f"{ed} ({window}): {len(d)} rows, {int(d.adds_up.sum())} add up; US {d.loc[d.country.str.startswith('United States'), 'analgesic'].round(0).tolist()}")
    r = pd.concat(frames, ignore_index=True)
    r.to_csv(out, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
