import csv, io, datetime
def _rows(ranked, titles):
    for i, g in enumerate(ranked, 1):
        c = g.get("components", {}); v = g.get("verification", {}); cf = g.get("confidence") or {}
        yield dict(rank=i, condition=g.get("condition", "G"), type=g["type"], title=g["title"], score=round(g.get("score", 0), 3), evidence_strength=round(c.get("evidence", 0), 2), novelty=round(c.get("novelty", 0), 2),
                   relevance=round(c.get("relevance", 0), 2), feasibility=round(c.get("feasibility", 0), 2), external_check=v.get("status", ""), confidence=cf.get("conf", ""), confidence_basis=cf.get("basis", ""),
                   explanation=g.get("explanation", ""), evidence=" || ".join(f"{titles.get(e['paper'], e['paper'])} p.{e['page']}: {e['passage']}" for e in g["evidence"]))
def to_csv(ranked, titles):
    rows = list(_rows(ranked, titles)); buf = io.StringIO()
    if rows: w = csv.DictWriter(buf, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return buf.getvalue()
def to_markdown(ranked, titles, project="", topic="", papers=None):
    out = [f"# Candidate research gaps: {project}", f"Topic: {topic} | exported {datetime.date.today()}", "", "These are CANDIDATE gaps for researcher review. A gap means 'not found in this corpus', not 'never done'.", ""]
    for r, g in zip(_rows(ranked, titles), ranked):
        out += [f"## {r['rank']}. {g['title']}", f"*{g['type']}* | score {r['score']} | external check: {r['external_check'] or 'not checked'}" + (f" | confidence {r['confidence']} ({r['confidence_basis']})" if r["confidence"] != "" else ""), "", g.get("explanation", ""), "", "Evidence:"]
        out += [f"- {titles.get(e['paper'], e['paper'])}, p.{e['page']}: \"{e['passage']}\"" for e in g["evidence"]] + [""]
    cited = {x["paper"] for g in ranked for x in g["evidence"]}; refs = [p for p in (papers or []) if p["id"] in cited]
    if refs: out += ["## References", ""] + [f"- {p['title']}" + (f" ({p['year']})" if p.get("year") else "") + (f". {p['url']}" if p.get("url") else "") for p in refs]
    return "\n".join(out)

def to_bibtex(papers):
    import re as _re
    out = []
    for p in papers:
        key = _re.sub(r"\W+", "", (p.get("id") or p["title"]))[:30] or "paper"; f = [f"  title = {{{p['title']}}}"]
        if p.get("year"): f.append(f"  year = {{{p['year']}}}")
        if p.get("doi"): f.append(f"  doi = {{{str(p['doi']).replace('https://doi.org/', '')}}}")
        elif p.get("url"): f.append(f"  url = {{{p['url']}}}")
        out.append("@article{" + key + ",\n" + ",\n".join(f) + "\n}")
    return "\n\n".join(out)
