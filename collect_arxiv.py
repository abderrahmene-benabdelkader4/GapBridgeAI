"""Collect a topic corpus from arXiv (metadata + PDFs) with a date window, for the time-split design.
Usage: python collect_arxiv.py --project mt --phrase "low-resource machine translation" --cat cs.CL --from 2019-01-01 --to 2023-12-31 --max 60
Writes projects/<project>/pdfs/*.pdf and manifest.csv. Respects arXiv's request to wait ~3 s between calls. NOT run in the build sandbox (no internet): the XML parsing is unit-tested, the network calls are not."""
import argparse, csv, re, time, datetime
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
NS = {"a": "http://www.w3.org/2005/Atom"}

def build_query(phrase, cat, date_from, date_to):
    d = lambda s, end: s.replace("-", "") + ("2359" if end else "0000")
    return f'cat:{cat} AND all:"{phrase}" AND submittedDate:[{d(date_from, False)} TO {d(date_to, True)}]'

def parse_feed(xml_text):
    out = []
    for e in ET.fromstring(xml_text).findall("a:entry", NS):
        full = e.findtext("a:id", "", NS).strip(); aid = re.sub(r"v\d+$", "", full.split("/abs/")[-1]); pdf = next((l.get("href") for l in e.findall("a:link", NS) if l.get("title") == "pdf"), f"https://arxiv.org/pdf/{aid}")
        out.append(dict(arxiv_id=aid, stem=aid.replace("/", "_"), title=" ".join(e.findtext("a:title", "", NS).split()), published=e.findtext("a:published", "", NS)[:10], pdf_url=pdf))
    return out

def collect(project, phrase, cat, date_from, date_to, max_results, root=Path(__file__).parent / "projects", delay=3.0):
    pdfs = root / project / "pdfs"; pdfs.mkdir(parents=True, exist_ok=True); H = {"User-Agent": "GapFinder-thesis/0.1 (research prototype)"}
    r = requests.get("https://export.arxiv.org/api/query", params=dict(search_query=build_query(phrase, cat, date_from, date_to), start=0, max_results=max_results, sortBy="relevance"), headers=H, timeout=60); r.raise_for_status()
    papers = parse_feed(r.text); rows = []
    for i, p in enumerate(papers, 1):
        f = pdfs / f"{p['stem']}.pdf"
        if not f.exists():
            time.sleep(delay); x = requests.get(p["pdf_url"], headers=H, timeout=120)
            if x.ok and x.content[:4] == b"%PDF": f.write_bytes(x.content)
            else: print("  skipped", p["arxiv_id"], x.status_code); continue
        rows.append(dict(p, downloaded=datetime.date.today().isoformat(), file=f.name)); print(f"[{i}/{len(papers)}] {p['arxiv_id']} {p['title'][:60]}")
    with open(root / project / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arxiv_id", "stem", "title", "published", "pdf_url", "downloaded", "file"]); w.writeheader(); w.writerows(rows)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--phrase", required=True); ap.add_argument("--cat", default="cs.CL")
    ap.add_argument("--from", dest="date_from", required=True); ap.add_argument("--to", dest="date_to", required=True); ap.add_argument("--max", type=int, default=60); a = ap.parse_args()
    collect(a.project, a.phrase, a.cat, a.date_from, a.date_to, a.max)
