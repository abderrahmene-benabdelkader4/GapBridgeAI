"""Experiment harness: runs conditions V, G, H on every project (= topic corpus), pools and de-duplicates candidates,
and writes a BLINDED pool for the expert raters plus a separate key file that maps items back to conditions."""
import json, time, random, hashlib, platform
from pathlib import Path
import core, llm, conditions as C, embed, pipeline

ROOT = Path(__file__).parent / "experiments"
NEUTRAL = ("You rewrite research-gap candidates as neutral statements for blind review. Use ONLY the title, explanation and passages given. "
           'Never mention how the candidate was found. Reply JSON: {"title": "<max 15 words>", "description": "<two sentences>"}.')

def neutral(c, titles, use_llm):
    """Same rewrite for every condition, so templated G titles and free-form V/H titles are not distinguishable by style."""
    if not use_llm: return c["title"], ""
    ev = "\n".join(f"- {titles.get(e['paper'], e['paper'])}: {e['passage']}" for e in c["evidence"][:4])
    try: r = llm.chat_json(NEUTRAL, f"Title: {c['title']}\nExplanation: {c.get('explanation', '')}\nPassages:\n{ev}", max_tokens=300); return str(r.get("title") or c["title"]), str(r.get("description") or "")
    except Exception: return c["title"], ""

def _groups(sim, thr):
    parent = list(range(len(sim)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(sim)):
        for j in range(i + 1, len(sim)):
            if sim[i][j] >= thr: parent[find(j)] = find(i)
    out = {}
    for i in range(len(sim)): out.setdefault(find(i), []).append(i)
    return list(out.values())

def pool_topic(cands, proj, titles, use_llm):
    """Merge near-duplicate candidates (across and within conditions) into items."""
    if not cands: return []
    texts = [f"{c['title']}. {c.get('explanation', '')}" for c in cands]; ix = embed.Index(texts); groups = _groups(ix.sim_matrix(), ix.dedupe_threshold); items = []
    for gi in groups:
        members = [cands[i] for i in gi]; rep = max(members, key=lambda c: len(c["evidence"])); seen, ev = set(), []
        for m in members:
            for e in m["evidence"]:
                if e["passage"] not in seen and len(ev) < 4: seen.add(e["passage"]); ev.append(e)
        t, d = neutral(dict(rep, evidence=ev), titles, use_llm)
        items.append(dict(item_id=hashlib.sha1((proj + rep["key"]).encode()).hexdigest()[:8], project=proj, title=t, description=d, evidence=ev,
                          origins=[dict(condition=m["condition"], rank=m["rank"], key=m["key"]) for m in members]))
    return items

def run(projects, n=8, use_llm=True, store=None, root=ROOT, conds=("V", "G", "H"), seed=0, log=print):
    d = Path(root) / time.strftime("%Y%m%d-%H%M%S"); d.mkdir(parents=True, exist_ok=True); allc, stats, items = [], [], []
    for proj in projects:
        ex, corpus = pipeline.load(proj), pipeline.load_corpus(proj)
        if not ex or not corpus: raise FileNotFoundError(f"project '{proj}' has no extraction/corpus: run extraction first")
        topic = pipeline.project_meta(proj)["topic"]; facts = pipeline.apply_review(ex["facts"], pipeline.load_review(proj)); titles = {p["id"]: p["title"] for p in ex["papers"]}
        chunks = C.build_chunks(corpus); ix = embed.Index([c["text"] for c in chunks]); cands = []; log(f"{proj}: topic '{topic}', {len(corpus)} papers, {len(chunks)} chunks, index {ix.kind}")
        for cond in conds:
            u0 = dict(llm.USAGE); t0 = time.time()
            r = C.run_cond(cond, corpus=corpus, chunks=chunks, index=ix, facts=facts, topic=topic, n=n, use_llm=use_llm, titles=titles, store=store if cond == "G" else None, project=proj)
            for rank, g in enumerate(r["cands"], 1): g["condition"], g["rank"], g["project"] = cond, rank, proj; cands.append(g)
            stats.append(dict(project=proj, condition=cond, n_raw=r["n_raw"], n_verified=r["n_verified"], n_final=len(r["cands"]), seconds=round(time.time() - t0, 2), **{k: llm.USAGE[k] - u0[k] for k in u0}))
            log(f"  {cond}: {len(r['cands'])} candidates ({r['n_raw']} raw, {r['n_verified']} passed the provenance gate)")
        allc += cands; items += pool_topic(cands, proj, titles, use_llm)
    random.Random(seed).shuffle(items); tmap = {p: {x["id"]: x["title"] for x in pipeline.load(p)["papers"]} for p in projects}
    blind = [dict(item_id=i["item_id"], topic=pipeline.project_meta(i["project"])["topic"], title=i["title"], description=i["description"],
                  evidence=[dict(paper=tmap[i["project"]].get(e["paper"], e["paper"]), page=e["page"], passage=e["passage"]) for e in i["evidence"]]) for i in items]
    key = {i["item_id"]: dict(project=i["project"], origins=i["origins"]) for i in items}
    (d / "candidates.json").write_text(json.dumps(allc, indent=1, default=str), encoding="utf-8"); (d / "pool_blind.json").write_text(json.dumps(blind, indent=1), encoding="utf-8")
    (d / "pool_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    (d / "run.json").write_text(json.dumps(dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), projects=projects, n=n, use_llm=use_llm, model=llm.model() if use_llm else None, embedding=ix.kind,
                 dedupe_threshold=ix.dedupe_threshold, seed=seed, python=platform.python_version(), stats=stats, n_items=len(items)), indent=1), encoding="utf-8")
    log(f"pool: {len(allc)} candidates -> {len(items)} items to rate. Folder: {d}"); return d
