"""Neo4j layer: project-isolated ingestion with provenance on every fact, and the Cypher gap detectors.
Every node carries a `project` property, so one database holds many isolated project graphs (reset = delete that project only)."""
import core
from core import alnum

# ───────── INGESTION (each statement is idempotent: MERGE + SET) ─────────
ING = {
 "paper": "UNWIND $rows AS row MERGE (p:Paper {project:$p, id:row.id}) SET p.title = row.title",
 "problem": "UNWIND $rows AS row MATCH (p:Paper {project:$p, id:row.paper}) MERGE (x:Problem {project:$p, name:row.name}) MERGE (p)-[r:ADDRESSES]->(x) SET r.page = row.page, r.passage = row.passage",
 "method": "UNWIND $rows AS row MATCH (p:Paper {project:$p, id:row.paper}) MERGE (x:Method {project:$p, name:row.name}) MERGE (p)-[r:USES]->(x) SET r.page = row.page, r.passage = row.passage, r.role = row.role",
 "dataset": "UNWIND $rows AS row MATCH (p:Paper {project:$p, id:row.paper}) MERGE (x:Dataset {project:$p, name:row.name}) MERGE (p)-[r:USES_DATASET]->(x) SET r.page = row.page, r.passage = row.passage",
 "context": "UNWIND $rows AS row MATCH (p:Paper {project:$p, id:row.paper}) MERGE (x:Context {project:$p, name:row.name}) MERGE (p)-[r:STUDIES]->(x) SET r.page = row.page, r.passage = row.passage",
 "applied": "UNWIND $rows AS row MERGE (m:Method {project:$p, name:row.method}) MERGE (x:Problem {project:$p, name:row.problem}) MERGE (m)-[r:APPLIED_TO {paper:row.paper}]->(x) SET r.page = row.page, r.passage = row.passage",
 "evaluated": "UNWIND $rows AS row MERGE (m:Method {project:$p, name:row.method}) MERGE (d:Dataset {project:$p, name:row.dataset}) MERGE (m)-[r:EVALUATED_ON {paper:row.paper}]->(d) SET r.page = row.page, r.passage = row.passage",
 "result": ("UNWIND $rows AS row MATCH (p:Paper {project:$p, id:row.paper}) MERGE (m:Method {project:$p, name:row.method}) MERGE (d:Dataset {project:$p, name:row.dataset}) "
           "MERGE (mt:Metric {project:$p, name:row.metric}) MERGE (r:Result {project:$p, rid:row.rid}) SET r.paper = row.paper, r.value = row.value, r.context = row.context, r.page = row.page, r.passage = row.passage "
           "MERGE (p)-[rp:REPORTS]->(r) SET rp.page = row.page, rp.passage = row.passage MERGE (r)-[:OF_METHOD]->(m) MERGE (r)-[:ON_DATASET]->(d) MERGE (r)-[:MEASURED_BY]->(mt)"),
 "limitation": "UNWIND $rows AS row MATCH (p:Paper {project:$p, id:row.paper}) MERGE (l:Limitation {project:$p, lid:row.lid}) SET l.text = row.text MERGE (p)-[r:STATES]->(l) SET r.page = row.page, r.passage = row.passage",
 "future": ("UNWIND $rows AS row MATCH (p:Paper {project:$p, id:row.paper}) MERGE (f:FutureWork {project:$p, fid:row.fid}) SET f.text = row.text, f.paper = row.paper, f.page = row.page, f.passage = row.passage, f.mentions = row.mentions "
            "MERGE (p)-[r:SUGGESTS]->(f) SET r.page = row.page, r.passage = row.passage"),
}
ING_MENTIONS = ("UNWIND $rows AS row MATCH (f:FutureWork {project:$p, fid:row.fid}) UNWIND row.mentions AS mname "
                "MATCH (e {project:$p, name:mname}) WHERE e:Method OR e:Problem OR e:Context OR e:Dataset MERGE (f)-[:MENTIONS]->(e)")

def rows_for(facts):
    """Group facts by kind and add the ids the graph needs."""
    out = {}
    for f in facts:
        r = dict(f)
        if f["kind"] == "result": r["rid"] = f"{f['paper']}|{f['method']}|{f['dataset']}|{f['metric']}|{f['context']}"
        if f["kind"] == "limitation": r["lid"] = f"{f['paper']}|{alnum(f['text'])[:40]}"
        if f["kind"] == "future": r["fid"] = f"{f['paper']}|{alnum(f['text'])[:40]}"
        out.setdefault(f["kind"], []).append(r)
    return out

# ───────── CYPHER GAP DETECTORS ─────────
A1 = """MATCH (m:Method {project:$p})-[:APPLIED_TO]->(a:Problem {project:$p})
WITH m, collect(DISTINCT a) AS done
WHERE size(done) >= 2
MATCH (c:Problem {project:$p})
WHERE NOT c IN done
MATCH (o:Method {project:$p})-[:APPLIED_TO]->(c)
WITH m, c, done, collect(DISTINCT o.name) AS others
RETURN m.name AS method, c.name AS problem, [x IN done | x.name] AS applied_to, others
ORDER BY method, problem"""
A1_EV = """MATCH (m:Method {project:$p, name:$m})-[r:APPLIED_TO]->(a:Problem {project:$p})
WHERE a.name IN $applied
RETURN r.paper AS paper, r.page AS page, r.passage AS passage, m.name + ' applied to ' + a.name AS note
UNION
MATCH (o:Method {project:$p})-[r:APPLIED_TO]->(c:Problem {project:$p, name:$c})
WHERE o.name IN $others
RETURN r.paper AS paper, r.page AS page, r.passage AS passage, o.name + ' applied to ' + c.name AS note"""
A1_FEAS = "MATCH (o:Method {project:$p})-[:EVALUATED_ON]->(d:Dataset {project:$p}) WHERE o.name IN $others RETURN count(DISTINCT d) AS n"

A2 = """MATCH (x:Context {project:$p})<-[:STUDIES]-(px:Paper {project:$p})
WITH x, collect(DISTINCT px.id) AS cps
WHERE size(cps) >= 2
MATCH (y:Problem {project:$p})<-[:ADDRESSES]-(py:Paper {project:$p})
WITH x, cps, y, collect(DISTINCT py.id) AS pps
WHERE size(pps) >= 2 AND none(q IN pps WHERE q IN cps)
RETURN x.name AS context, y.name AS problem, cps, pps
ORDER BY context, problem"""
A2_EV = """MATCH (p:Paper {project:$p})-[r:STUDIES]->(x:Context {project:$p, name:$x})
RETURN p.id AS paper, r.page AS page, r.passage AS passage, 'studies ' + x.name AS note
UNION
MATCH (p:Paper {project:$p})-[r:ADDRESSES]->(y:Problem {project:$p, name:$y})
RETURN p.id AS paper, r.page AS page, r.passage AS passage, 'addresses ' + y.name AS note"""
A2_FEAS = ("MATCH (p:Paper {project:$p})-[:STUDIES]->(:Context {project:$p, name:$x}) MATCH (p)-[:USES_DATASET]->(d:Dataset {project:$p}) RETURN count(DISTINCT d) AS n")

B1 = """MATCH (r1:Result {project:$p})-[:OF_METHOD]->(m:Method), (r1)-[:ON_DATASET]->(d:Dataset), (r1)-[:MEASURED_BY]->(mt:Metric)
MATCH (r2:Result {project:$p})-[:OF_METHOD]->(m), (r2)-[:ON_DATASET]->(d), (r2)-[:MEASURED_BY]->(mt)
WHERE r1.rid < r2.rid AND r1.paper <> r2.paper AND r1.context = r2.context AND abs(r1.value - r2.value) >= $thr
RETURN m.name AS method, d.name AS dataset, mt.name AS metric, r1.context AS context, r1.paper AS paper1, r2.paper AS paper2,
       r1.value AS v1, r2.value AS v2, r1.page AS page1, r1.passage AS passage1, r2.page AS page2, r2.passage AS passage2
ORDER BY method, dataset, metric, paper1"""

B2_ITEMS = """MATCH (p:Paper {project:$p})-[:SUGGESTS]->(f:FutureWork {project:$p})
OPTIONAL MATCH (f)-[:MENTIONS]->(e)
WITH p, f, collect(DISTINCT e.name) AS matched
RETURN f.fid AS fid, p.id AS paper, f.text AS text, f.page AS page, f.passage AS passage, f.mentions AS mentions, matched
ORDER BY paper, fid"""
B2_LINKS = """MATCH (f:FutureWork {project:$p, fid:$fid})-[:MENTIONS]->(e)
MATCH (q:Paper {project:$p})-[:ADDRESSES|USES|STUDIES|USES_DATASET]->(e)
WHERE q.id <> f.paper
RETURN q.id AS paper, collect(DISTINCT e.name) AS hit"""
B2_EV = """MATCH (f:FutureWork {project:$p, fid:$fid})-[:MENTIONS]->(e)
MATCH (q:Paper {project:$p})-[r:ADDRESSES|USES|STUDIES|USES_DATASET]->(e)
WHERE q.id <> f.paper
RETURN q.id AS paper, r.page AS page, r.passage AS passage, 'related: ' + e.name AS note
ORDER BY paper LIMIT 4"""
PASSAGES = """MATCH (q:Paper {project:$p, id:$id})-[r]->()
WHERE r.passage IS NOT NULL
RETURN DISTINCT r.page AS page, r.passage AS passage LIMIT 8"""

class Store:
    def __init__(self, uri, user, password, database=None):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password)); self.db = database
    def q(self, cy, **params):
        with self.driver.session(database=self.db) as s: return s.run(cy, **params).data()
    def check(self): self.driver.verify_connectivity(); return True
    def projects(self): return [r["project"] for r in self.q("MATCH (p:Paper) RETURN DISTINCT p.project AS project ORDER BY project")]
    def titles(self, p): return {r["id"]: r["title"] for r in self.q("MATCH (x:Paper {project:$p}) RETURN x.id AS id, x.title AS title", p=p)}
    def stats(self, p):
        return dict(nodes={r["label"]: r["n"] for r in self.q("MATCH (n {project:$p}) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label", p=p)},
                    relationships={r["rel"]: r["n"] for r in self.q("MATCH (a {project:$p})-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel", p=p)})
    def reset(self, p): self.q("MATCH (n {project:$p}) DETACH DELETE n", p=p)          # only this project's nodes
    def ingest(self, p, papers, facts):
        self.q(ING["paper"], p=p, rows=[dict(id=x["id"], title=x["title"]) for x in papers])
        g = rows_for(facts)
        for kind in ("problem", "method", "dataset", "context", "applied", "evaluated", "result", "limitation", "future"):
            if g.get(kind): self.q(ING[kind], p=p, rows=g[kind])
        if g.get("future"): self.q(ING_MENTIONS, p=p, rows=g["future"])            # needs entities to exist first

    # --- detectors: same outputs as core.detect_*_py (selftest.py cross-checks them) ---
    def detect_all(self, p, use_llm=False, thr=core.THRESH):
        gaps = []
        for r in self.q(A1, p=p):
            m, c = r["method"], r["problem"]; ev = self.q(A1_EV, p=p, m=m, applied=r["applied_to"], c=c, others=r["others"]); n = self.q(A1_FEAS, p=p, others=r["others"])[0]["n"]
            gaps.append(core._gap("method_problem", f"mp|{m}|{c}", f"{m} has not been applied to {c}", [m, c], ev, [[m], [c]], 1.0 if n else .3, applied_to=sorted(r["applied_to"]), others=sorted(r["others"])))
        for r in self.q(A2, p=p):
            x, y = r["context"], r["problem"]; ev = self.q(A2_EV, p=p, x=x, y=y); n = self.q(A2_FEAS, p=p, x=x)[0]["n"]
            gaps.append(core._gap("context_problem", f"cp|{x}|{y}", f"{x} is never studied together with {y}", [x, y], ev, [[x], [y]], 1.0 if n else .3, context_papers=sorted(r["cps"]), problem_papers=sorted(r["pps"])))
        for r in self.q(B1, p=p, thr=thr):
            m, d, mt, c = r["method"], r["dataset"], r["metric"], r["context"]; where = f" ({c})" if c else ""
            ev = [dict(paper=r["paper1"], page=r["page1"], passage=r["passage1"], note=f"{r['paper1']} reports {r['v1']:g}"), dict(paper=r["paper2"], page=r["page2"], passage=r["passage2"], note=f"{r['paper2']} reports {r['v2']:g}")]
            gaps.append(core._gap("conflict", f"cf|{m}|{d}|{mt}|{c}|{r['paper1']}|{r['paper2']}", f"Conflicting results for {m} on {d}, {mt}{where}: {r['v1']:g} vs {r['v2']:g}", [m, d, mt] + ([c] if c else []), ev, [], 1.0, values=[r["v1"], r["v2"]]))
        items = []
        for r in self.q(B2_ITEMS, p=p):
            matched = list(r["matched"]); unmatched = [x for x in (r["mentions"] or []) if x not in matched]
            links = {l["paper"]: set(l["hit"]) for l in self.q(B2_LINKS, p=p, fid=r["fid"])}
            covering = sorted(q for q, hit in links.items() if matched and not unmatched and hit == set(matched))
            items.append(dict(fid=r["fid"], paper=r["paper"], text=r["text"], page=r["page"], passage=r["passage"], matched=matched, unmatched=unmatched, covering=covering, candidates=sorted(links)))
        gaps += core.build_b2(items, lambda pid: self.q(PASSAGES, p=p, id=pid), use_llm, lambda it: self.q(B2_EV, p=p, fid=it["fid"]))
        return gaps
