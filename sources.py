"""Finding papers: OpenAlex search (titles, abstracts, open-access PDF links) and validated PDF download."""
import os, re, time
from pathlib import Path
import requests
from core import _abstract

class SourceError(Exception): pass
def base(): return os.getenv("OPENALEX_URL", "https://api.openalex.org").rstrip("/")
def _ua(): m = os.getenv("OPENALEX_MAILTO", ""); return {"User-Agent": f"GapFinder/1.0 (literature tool{'; mailto:' + m if m else ''})"}
_sleep = time.sleep

def _get(url, params=None, tries=5, stream=False, timeout=(10, 60)):
    for a in range(tries):
        try:
            r = requests.get(url, params=params, headers=_ua(), timeout=timeout, stream=stream)
            if r.status_code == 429 or r.status_code >= 500:
                _sleep(min(float(r.headers.get("retry-after", 2 ** a)), 30)); continue
            return r
        except (requests.ConnectionError, requests.Timeout): _sleep(min(2 ** a, 15))
    raise SourceError("Could not reach the paper service (network problem or rate limit). Try again in a minute.")

def _norm(w):
    locs = [w.get("best_oa_location") or {}, w.get("primary_location") or {}]; urls = []
    for u in [l.get("pdf_url") for l in locs] + [(w.get("open_access") or {}).get("oa_url")]:
        if u and u not in urls: urls.append(u)
    land = next((l.get("landing_page_url") for l in locs if l.get("landing_page_url")), None)
    return dict(id=(w.get("id") or "").rsplit("/", 1)[-1], title=" ".join((w.get("title") or "").split()), year=w.get("publication_year"), doi=w.get("doi"), cited_by=w.get("cited_by_count", 0),
                abstract=_abstract(w), pdf_urls=urls, url=w.get("doi") or land or w.get("id"), authors=[(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:3]],
                venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name"))

def search(query, year_from=None, year_to=None, n=50, sort="relevance"):
    flt = ["has_abstract:true"] + ([f"from_publication_date:{year_from}-01-01"] if year_from else []) + ([f"to_publication_date:{year_to}-12-31"] if year_to else [])
    out, page, per = [], 1, min(max(n, 10), 100)
    while len(out) < n:
        p = {"search": query, "filter": ",".join(flt), "per-page": per, "page": page}
        if sort == "cited": p["sort"] = "cited_by_count:desc"
        if os.getenv("OPENALEX_MAILTO"): p["mailto"] = os.environ["OPENALEX_MAILTO"]
        r = _get(base() + "/works", p)
        if r.status_code != 200: raise SourceError(f"Paper search failed (HTTP {r.status_code}). {r.text[:150]}")
        res = r.json().get("results", []); out += [_norm(w) for w in res if w.get("title")]
        if len(res) < per: break
        page += 1
    by_title, order = {}, []
    for p in out:                                                  # preprint + journal version share a title but not a DOI: keep one, preferring the copy with a PDF link
        k = re.sub(r"\W+", "", p["title"].lower())
        if k not in by_title: by_title[k] = p; order.append(k)
        elif p["pdf_urls"] and not by_title[k]["pdf_urls"]: by_title[k] = p
    return [by_title[k] for k in order][:n]

def fix_url(u): return re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", u or "")
def download_pdf(urls, dest, max_mb=40, delay=1.0):
    """Try each candidate URL; accept only real PDFs. Returns (ok, reason)."""
    why = "no open-access PDF link"
    for u in urls:
        u = fix_url(u)
        try:
            r = _get(u, stream=True, tries=2)
            if r.status_code != 200: why = f"HTTP {r.status_code}"; continue
            data = b""; 
            for chunk in r.iter_content(65536):
                data += chunk
                if len(data) > max_mb * 1e6: why = "file too large"; break
            else:
                if data[:5] == b"%PDF-": Path(dest).write_bytes(data); return True, "ok"
                why = "link is a web page, not a PDF (probably paywalled)"
        except SourceError: why = "download failed"
        _sleep(delay)
    return False, why
