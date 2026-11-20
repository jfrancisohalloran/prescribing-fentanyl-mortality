"""Tidy DEA's ARCOS Retail Drug Summary, Report 4 (cumulative distribution by state in grams per 100,000 population), 2006-2015.

Each report lists, for every drug, each state's population, total grams distributed to retail registrants (pharmacies, hospitals,
practitioners and teaching institutions) and grams per 100,000 population for the year. This reads every drug and state into one long
file and checks each drug's rows against the report's US total line.

Fentanyl here is fentanyl base in pharmaceutical products (mostly transdermal patches at retail); oxycodone and hydrocodone are mostly
oral pills. Weights are grams of base drug, not doses, so compare drugs by their own variation, not by grams.

  python3 code/08_arcos_retail.py <dea_arcos_retail date folder> <out csv>
"""
import pathlib
import re
import subprocess
import sys

import pandas as pd

HEAD = re.compile(r"DRUG CODE:\s*(\S+)\s+DRUG NAME:\s*(.+?)\s*$")
HEAD2 = re.compile(r"^\s*DRUG:\s*(\S+)\s+-\s+(.+?)\s*$")  # the reissued 2011 and 2012 reports
ROW = re.compile(r"^\s*(\d+)\s+([A-Z][A-Z .'&-]*?)\s{2,}([\d,]+)\s+([\d,]+(?:\.\d+)?)\s+([\d,]+(?:\.\d+)?)\s*$")
TOTAL = re.compile(r"(?:U\.S\. GRAMS / PER 100K:|^US TOTAL)\s+([\d,]+)\s+([\d,]+(?:\.\d+)?)\s+([\d,]+(?:\.\d+)?)")


def num(x):
    return float(x.replace(",", ""))


def parse(path, year):
    text = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
    rows, totals, code, name = [], {}, None, None
    for line in text.splitlines():
        h = HEAD.search(line) or HEAD2.match(line)
        if h:
            code, name = h.group(1), h.group(2)
            continue
        t = TOTAL.search(line)
        if t and code:
            totals[code] = num(t.group(2))
            continue
        r = ROW.match(line)
        if r and code:
            rows.append(dict(year=year, drug_code=code, drug=name, state=r.group(2).strip(), population=num(r.group(3)),
                             grams=num(r.group(4)), grams_per_100k=num(r.group(5))))
    d = pd.DataFrame(rows).drop_duplicates(["drug_code", "state"])
    check = d.groupby("drug_code").grams.sum()
    bad = [c for c, g in totals.items() if c in check and abs(check[c] - g) > 0.01 * max(g, 1)]
    return d, len(totals), bad


def main(folder, out):
    frames = []
    for path in sorted(pathlib.Path(folder).glob("*_rpt4.pdf")):
        year = int(path.name[:4])
        d, n_tot, bad = parse(path, year)
        frames.append(d)
        print(f"{year}: {d.drug_code.nunique()} drugs, {d.state.nunique()} states, {len(d)} rows; "
              f"{n_tot - len(bad)} of {n_tot} drugs match the US total{'; off: ' + ', '.join(bad) if bad else ''}")
    r = pd.concat(frames, ignore_index=True)
    r.to_csv(out, index=False)
    for pat in ("FENTANYL", "OXYCODONE", "HYDROCODONE"):
        k = r[r.drug.str.contains(pat) & (r.state == "WEST VIRGINIA")]
        print(pat, "West Virginia grams/100k by year:", k.set_index("year").grams_per_100k.round(0).to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:3])
