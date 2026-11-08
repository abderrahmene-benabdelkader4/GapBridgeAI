"""The three experimental conditions. All share: corpus, LLM, output schema, provenance gate.
  V = vector RAG: top-k retrieved chunks -> LLM proposes gaps
  G = graph: deterministic structural detectors (the LLM only explains)
  H = hybrid: top-k chunks + a graph-derived fact context (1-hop expansion over shared entities) -> LLM proposes gaps"""
import core, llm
from core import norm_ws, alnum

GAP_SYSTEM = "You propose research gaps from provided excerpts of papers. Reply with a single JSON object and nothing else."
GAP_USER = ('Topic: {topic}\nPropose up to {n} research gaps that the excerpts below suggest but do not resolve. A gap is something missing, untested, contradicted or left open in these papers.\n'
            'Return JSON: {{"gaps":[{{"title":str,"explanation":str,"evidence":[{{"paper":str,"passage":str}}]}}]}}\n'
            'Rules: "paper" is the id written before the "|" in the excerpt header. Each "passage" must be ONE sentence copied verbatim from the excerpts. Give 1 to 3 evidence passages per gap. '
            'Use no outside knowledge. If the excerpts support no gap, return {{"gaps":[]}}.\n\n{graph_context}EXCERPTS:\n{excerpts}')

GAPS_SCHEMA = {"type": "object", "required": ["gaps"], "properties": {"gaps": {"type": "array", "items": {"type": "object", "required": ["title", "explanation", "evidence"], "properties": {"title": {"type": "string"}, "explanation": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "object", "required": ["paper", "passage"], "properties": {"paper": {"type": "string"}, "passage": {"type": "string"}}}}}}}}}

def build_chunks(corpus, window=3, stride=2):
    out = []
    for p in corpus:
        s = p["sentences"]; pages = [core._page(x, p) for x in s]
        out += [dict(paper=p["id"], page=pages[i], text=" ".join(s[i:i + window])) for i in range(0, len(s), stride)]
    return out

def retrieve(index, chunks, topic, k=12):
    """Three queries (topic, limitations, future work) so retrieval is not limited to the topic's most generic passages."""
    qs = [(topic, k // 3), (topic + " limitations", k // 3), (topic + " future work open questions", k - 2 * (k // 3))]; seen, out = set(), []
    for q, kk in qs:
        got = 0
        for i, _ in index.search(q, k * 3):
            if i not in seen and got < kk: seen.add(i); out.append(i); got += 1
    return out

def excerpts(hits, chunks): return "\n".join(f"[{chunks[i]['paper']} | p.{chunks[i]['page']}] {chunks[i]['text']}" for i in hits)

def graph_context(facts, paper_ids, max_lines=80):
    paper_ids = sorted(paper_ids); ids = set(paper_ids); typed = ("problem", "method", "dataset", "context")
    ents = {f["name"] for f in facts if f["paper"] in ids and f["kind"] in typed}; shared = {}
    for f in facts:
        if f["paper"] not in ids and f["kind"] in typed and f["name"] in ents: shared[f["paper"]] = shared.get(f["paper"], 0) + 1
    extra = [p for p, _ in sorted(shared.items(), key=lambda kv: (-kv[1], kv[0]))[:6]]; lines = []
    for pid in paper_ids + extra:
        for f in facts:
            if f["paper"] != pid: continue
            k = f["kind"]; s = {"applied": lambda: f"{f['method']} applied to {f['problem']}", "context": lambda: f"studies {f['name']}", "dataset": lambda: f"uses dataset {f['name']}",
                                "result": lambda: f"{f['method']} on {f['dataset']}: {f['metric']} = {f['value']:g}" + (f" ({f['context']})" if f.get("context") else ""),
                                "future": lambda: f"future work: {f['text']}"}.get(k)
            if s: lines.append(f"- {pid}: {s()}")
    return "STRUCTURED FACTS from a knowledge graph of the same corpus (papers beyond the excerpts share entities with them):\n" + "\n".join(lines[:max_lines]) + "\n\n"

def gate(cands, by_id):
    """Provenance gate, identical for every condition: keep a candidate only if at least one cited passage is a real sentence of the cited paper."""
    ok = []
    for c in cands:
        ev = []
        for e in c.get("evidence") or []:
            pid = e.get("paper"); p = by_id.get(pid) or next((v for k, v in by_id.items() if pid and k.startswith(str(pid))), None)
            ps = core._passage(e.get("passage", ""), p) if p else None
            if ps: ev.append(dict(paper=p["id"], page=core._page(ps, p), passage=ps, note=e.get("note", "cited passage")))
        if ev: c = dict(c); c["evidence"] = ev; c["support_papers"] = len({e["paper"] for e in ev}); ok.append(c)
    return ok

def to_gap(c, cond):
    title = norm_ws(c.get("title", ""))[:200]; g = core._gap("llm_gap", f"{cond}|{alnum(title)[:60]}", title, [], c.get("evidence") or [], [], .5)
    g["explanation"] = norm_ws(c.get("explanation", "")) or title; g["condition"] = cond; return g

def run_cond(cond, *, corpus, chunks, index, facts, topic, n=8, use_llm=True, titles=None, store=None, project=None, k=12):
    """Returns dict(cands=[gap,...], n_raw=int, n_verified=int)."""
    by_id = {p["id"]: p for p in corpus}; titles = titles or {p["id"]: p["title"] for p in corpus}
    if cond == "G":
        gaps = store.detect_all(project, use_llm) if store else core.detect_all_py(facts, use_llm)
        for g in gaps: g.setdefault("verification", dict(status="not checked"))
        ranked = core.rank(gaps, topic, titles); ok = gate(ranked, by_id)
        for g in ok[:n]: g["explanation"] = core.explain(g, titles, use_llm); g["condition"] = "G"
        return dict(cands=ok[:n], n_raw=len(ranked), n_verified=len(ok))
    if cond not in ("V", "H"): raise ValueError(cond)
    if not use_llm: raise RuntimeError("conditions V and H need an LLM (set GROQ_API_KEY)")
    hits = retrieve(index, chunks, topic, k); ids = {chunks[i]["paper"] for i in hits}
    raw = llm.chat_json(GAP_SYSTEM, GAP_USER.format(topic=topic, n=n, graph_context=graph_context(facts, ids) if cond == "H" else "", excerpts=excerpts(hits, chunks)), max_tokens=max(1800, llm.output_tokens() // 2), schema=GAPS_SCHEMA)
    cands = [to_gap(c, cond) for c in (raw.get("gaps") or []) if isinstance(c, dict) and c.get("title")]; ok = gate(cands, by_id)
    return dict(cands=ok[:n], n_raw=len(cands), n_verified=len(ok))
