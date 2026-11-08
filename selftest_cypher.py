"""Runs the REAL Cypher of graph.py on an embedded openCypher engine (FalkorDB-lite, Linux/macOS only: pip install falkordblite)
and compares the detectors with the Python mirror. This is a dialect-neutral check, NOT a Neo4j run: still run  python selftest.py --neo4j  on your Neo4j."""
import sys, tempfile
from pathlib import Path
import core, graph, pipeline
try: from redislite.falkordb_client import FalkorDB
except Exception as e: print("SKIP: falkordblite not installed or unsupported on this OS:", type(e).__name__); sys.exit(0)

class FalkorStore(graph.Store):
    def __init__(self):
        self.db = FalkorDB(str(Path(tempfile.mkdtemp()) / "g.db")); self.g = self.db.select_graph("gf")
    def q(self, cy, **params):
        r = self.g.query(cy, params); names = [h[1] for h in r.header]; return [dict(zip(names, row)) for row in r.result_set]

ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or c
pdfs = sorted(Path("samples").glob("*.pdf")); res = pipeline.extract_all(pdfs, "fixture", False, log=lambda *_: None); keys = lambda gs: sorted(g["key"] for g in gs); R = []
s = FalkorStore(); s.ingest("p1", res["papers"], res["facts"]); s.ingest("p2", res["papers"][:2], [f for f in res["facts"] if f["paper"] in {p["id"] for p in res["papers"][:2]}])
st = s.stats("p1"); R.append(ok(st["nodes"].get("Paper") == 6 and "APPLIED_TO" in st["relationships"], f"ingestion on the Cypher engine: {sum(st['nodes'].values())} nodes, {sum(st['relationships'].values())} relationships"))
real = s.detect_all("p1"); mirror = core.detect_all_py(res["facts"]); R.append(ok(keys(real) == keys(mirror), f"Cypher detectors return the same {len(mirror)} gaps as the Python mirror"))
if keys(real) != keys(mirror): print("  only Cypher:", sorted(set(keys(real)) - set(keys(mirror))), "\n  only mirror:", sorted(set(keys(mirror)) - set(keys(real))))
by = {g["key"]: g for g in real}; k = "mp|mBART|domain adaptation"; R.append(ok(k in by and len(by[k]["evidence"]) >= 3 and all(e["passage"] and e["page"] for e in by[k]["evidence"]), "Cypher evidence queries return passages with pages (provenance survives the round trip)"))
R.append(ok(any(g["type"] == "conflict" and g["meta"]["values"] == [22.4, 25.6] for g in real), "Cypher conflict detector finds 22.4 vs 25.6"))
R.append(ok(len([g for g in real if g["type"] == "future_work"]) == 3, "Cypher future-work detector flags exactly the 3 unaddressed statements"))
p2 = s.detect_all("p2"); R.append(ok(keys(p2) == keys(core.detect_all_py([f for f in res["facts"] if f["paper"] in {p["id"] for p in res["papers"][:2]}])), "a second project in the same database is isolated and gives its own result"))
s.reset("p2"); R.append(ok(s.stats("p2")["nodes"] == {} and s.stats("p1")["nodes"].get("Paper") == 6, "reset deletes only the target project"))
print(f"\n{sum(R)}/{len(R)} Cypher checks passed"); sys.exit(0 if all(R) else 1)
