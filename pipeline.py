"""Glue used by both the CLI and the Streamlit app."""
import os, json, time, hashlib
from pathlib import Path
import core, llm

import settings
HERE = Path(__file__).parent
def pdir(project): d = settings.projects_dir() / project; d.mkdir(parents=True, exist_ok=True); return d

def get_store():
    """Neo4j store from env vars, or None when not configured/unreachable."""
    if not os.getenv("NEO4J_PASSWORD"): return None
    import graph
    s = graph.Store(os.getenv("NEO4J_URI", "bolt://localhost:7687"), os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"], os.getenv("NEO4J_DATABASE") or None); s.check(); return s

def extract_all(pdfs, mode, use_llm, log=print):
    """PDFs -> validated, normalized facts. mode: 'fixture' (offline demo) or 'llm' (Groq)."""
    papers, facts, dropped, stats, parsed = [], [], [], dict(calls=0, cached=0), []
    for i, pdf in enumerate(pdfs, 1):
        log(f"[{i}/{len(pdfs)}] {Path(pdf).name}"); p = core.parse_pdf(pdf); raw, st = core.extract_paper(p, mode, log=log)
        f, d = core.validate(raw, p); parsed.append(p); papers.append(dict(id=p["id"], title=p["title"])); facts += f; dropped += [(p["id"],) + x for x in d]
        stats["calls"] += st["calls"]; stats["cached"] += st["cached"]; log(f"    {len(f)} facts kept, {len(d)} rejected")
    facts, review, merges = core.normalize(facts, core.load_aliases(), core.llm_confirm if use_llm else None)
    for f in facts: f["id"] = fact_id(f)
    return dict(papers=papers, facts=facts, dropped=dropped, review=review, merges=merges, stats=stats, parsed=parsed)

def fact_id(f): return hashlib.sha1(json.dumps({k: v for k, v in f.items() if k != "id"}, sort_keys=True).encode()).hexdigest()[:10]
def save(project, res):
    d = pdir(project); (d / "extraction.json").write_text(json.dumps(dict(papers=res["papers"], facts=res["facts"], review=res.get("review", []), merges=res.get("merges", []), norm_error=res.get("norm_error", "")), indent=1), encoding="utf-8")
    if res.get("parsed"): (d / "corpus.json").write_text(json.dumps(res["parsed"]), encoding="utf-8")   # parsed text, needed by conditions V/H and provenance checks
def load(project):
    f = pdir(project) / "extraction.json"; return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
def load_corpus(project):
    f = pdir(project) / "corpus.json"; return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
def project_meta(project, **update):
    f = pdir(project) / "project.json"; m = json.loads(f.read_text(encoding="utf-8")) if f.exists() else dict(topic=project.replace("_", " "))
    if update: m.update({k: v for k, v in update.items() if v is not None}); f.write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m
def load_review(project):
    f = pdir(project) / "review.json"; return json.loads(f.read_text(encoding="utf-8")) if f.exists() else dict(rejected=[])
def save_review(project, review): (pdir(project) / "review.json").write_text(json.dumps(review, indent=1), encoding="utf-8")
def apply_review(facts, review): rej = set(review.get("rejected", [])); return [f for f in facts if f.get("id") not in rej]

def detect(res, project, store, use_llm):
    return store.detect_all(project, use_llm) if store else core.detect_all_py(res["facts"], use_llm)

def verify_all(gaps, project, log=print):
    cache = pdir(project) / "verify_cache.json"; c = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}; al = core.load_aliases(); mail = os.getenv("OPENALEX_MAILTO")
    search = lambda q: core.openalex_search(q, mailto=mail)
    for g in gaps:
        if g["key"] not in c or c[g["key"]]["status"] == "unavailable":
            c[g["key"]] = core.verify_gap(g, al, search); time.sleep(.3)
        g["verification"] = c[g["key"]]
    cache.write_text(json.dumps(c, indent=1), encoding="utf-8"); return gaps

def finish(gaps, titles, topic, weights, use_llm):
    for g in gaps: g.setdefault("verification", dict(status="not checked", papers=[], note="not checked"))
    ranked = core.rank(gaps, topic, titles, weights)
    for g in ranked: g["explanation"] = g.get("explanation") or core.explain(g, titles, use_llm)
    return ranked
