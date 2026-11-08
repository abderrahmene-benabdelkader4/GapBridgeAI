"""Offline self-test (no Neo4j / Groq / internet needed):   python selftest.py
Cross-check on YOUR Neo4j (the Cypher is not run anywhere else):   python selftest.py --neo4j"""
import sys, json, tempfile, types
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
import core, llm, graph, pipeline
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or c
results = []; chk = lambda c, m: results.append(ok(c, m))
pdfs = sorted(Path("samples").glob("*.pdf")); res = pipeline.extract_all(pdfs, "fixture", False, log=lambda *_: None)
keys = lambda gs: sorted(g["key"] for g in gs)
gaps = core.detect_all_py(res["facts"]); by = {t: [g for g in gaps if g["type"] == t] for t in ("method_problem", "context_problem", "conflict", "future_work")}
chk(len(res["dropped"]) == 0 and len(res["facts"]) == 73, "6 papers parsed, 73 facts kept, 0 rejected")
chk(sorted(g["entities"][0] + ">" + g["entities"][1] for g in by["method_problem"]) == ["LoRA>dialect translation", "mBART>domain adaptation"], "A1 finds LoRA×dialect translation and mBART×domain adaptation only")
chk([g["entities"] for g in by["context_problem"]] == [["Yoruba", "domain adaptation"]], "A2 finds Yoruba×domain adaptation only")
chk(len(by["conflict"]) == 1 and by["conflict"][0]["meta"]["values"] == [22.4, 25.6], "B1 finds the single mBART/FLORES-200 conflict (22.4 vs 25.6)")
chk(sorted(g["key"].split("|")[1] for g in by["future_work"]) == ["synth_P4_algerian_arabic_dialect", "synth_P5_backtranslation_medical", "synth_P6_lora_yoruba"], "B2 flags 3 unaddressed future-work statements, drops the 3 already addressed")
chk(all(e["passage"] and e["page"] for g in gaps for e in g["evidence"]), "every evidence item has a passage and page")

# provenance guard: a paraphrased passage is repaired to the real sentence; an invented one is rejected
p = core.parse_pdf(pdfs[0]); raw = json.loads((Path("fixtures") / f"{p['id']}.json").read_text(encoding="utf-8"))
raw["limitations"].append(dict(text="x", passage="The authors admit that the BLEU metric might miss morphology errors.")); raw["problems"].append(dict(name="made-up", passage="We also solve world hunger with transformers."))
f, d = core.validate(raw, p); chk(any("verbatim" in x[1] for x in d) and not any(x["name"] == "made-up" for x in f if x["kind"] == "problem"), "invented passage rejected by the provenance check")

# Groq path with a mocked LLM: extraction, cache, retry-free parsing
calls = []; llm.chat_json = lambda s, u, max_tokens=1800, schema=None: calls.append(1) or json.loads((Path("fixtures") / f"{p['id']}.json").read_text(encoding="utf-8"))
with tempfile.TemporaryDirectory() as t:
    r1, s1 = core.extract_paper(p, "llm", Path(t), log=lambda *_: None); r2, s2 = core.extract_paper(p, "llm", Path(t), log=lambda *_: None)
chk(s1["calls"] == 1 and s2["cached"] == 1 and len(calls) == 1, "LLM extraction is cached (2nd run makes 0 calls)")

# B2 LLM judge with a mocked reply: only accepted when the quote really exists in the provided passages
item = dict(paper="A", text="t", covering=[], candidates=["B"]); get = lambda pid: [dict(page=1, passage="Back-translation reaches 26.5 BLEU on OPUS-100 for Swahili.")]
llm.chat_json = lambda s, u, max_tokens=1800, schema=None: dict(addressed=True, paper="B", quote="Back-translation reaches 26.5 BLEU on OPUS-100 for Swahili.")
chk(core.judge_future(item, get, True)[0] is True, "judge accepts an addressed claim with a verifiable quote")
llm.chat_json = lambda s, u, max_tokens=1800, schema=None: dict(addressed=True, paper="B", quote="This paper solves it completely.")
chk(core.judge_future(item, get, True)[0] is False, "judge rejects a claim whose quote is not in the passages")

# external verification with a mocked OpenAlex
inv = lambda t: {w: [i] for i, w in enumerate(t.split())}
fake = lambda q: [dict(title="LoRA for dialect machine translation", publication_year=2024, doi="https://doi.org/x", abstract_inverted_index=inv("We apply low-rank adaptation to Arabic dialect translation"))]
g = next(x for x in gaps if x["key"] == "mp|LoRA|dialect translation"); v = core.verify_gap(g, core.load_aliases(), fake)
chk(v["status"] == "exists externally", "alias-aware external check recognises 'low-rank adaptation' ≈ LoRA")
chk(core.verify_gap(g, core.load_aliases(), lambda q: [])["status"] == "not found externally", "no results -> 'not found externally'")
chk(core.verify_gap(g, core.load_aliases(), lambda q: 1 / 0)["status"] == "unavailable", "search failure -> 'unavailable' (never crashes)")

# ranking: weights change the order
titles = {x["id"]: x["title"] for x in res["papers"]}
for x in gaps: x["verification"] = dict(status="not found externally") if x["type"] != "conflict" else dict(status="not applicable")
for x in gaps:
    if x["key"] == "mp|LoRA|dialect translation": x["verification"] = dict(status="exists externally")   # pretend OpenAlex found it
pos = lambda w, k="mp|LoRA|dialect translation": [x["key"] for x in core.rank([dict(x) for x in gaps], "low-resource machine translation", titles, w)].index(k)
chk(pos(core.DEFAULT_W) > pos(dict(evidence=.35, novelty=0, relevance=.2, feasibility=.15)), "a gap that already exists externally drops in rank when novelty is weighted (and not when it is ignored)")

# plumbing of the Neo4j layer with a fake driver that answers from the in-memory mirror (does NOT test the Cypher itself)
class Fake(graph.Store):
    def __init__(self): self.facts = []; self.stmts = []
    def q(self, cy, **kw):
        self.stmts.append(cy)
        for k, v in graph.ING.items():
            if cy is v: self.facts += [dict(r) for r in kw["rows"]] if k != "paper" else []; return []
        if cy is graph.ING_MENTIONS or cy is graph.B2_EV and False: return []
        F = self.facts; m = {x["key"]: x for x in core.detect_all_py(F)}
        if cy is graph.A1: return [dict(method=x["entities"][0], problem=x["entities"][1], applied_to=x["meta"]["applied_to"], others=x["meta"]["others"]) for x in m.values() if x["type"] == "method_problem"]
        if cy is graph.A1_EV: return m[f"mp|{kw['m']}|{kw['c']}"]["evidence"]
        if cy is graph.A1_FEAS: return [dict(n=1)]
        if cy is graph.A2: return [dict(context=x["entities"][0], problem=x["entities"][1], cps=x["meta"]["context_papers"], pps=x["meta"]["problem_papers"]) for x in m.values() if x["type"] == "context_problem"]
        if cy is graph.A2_EV: return m[f"cp|{kw['x']}|{kw['y']}"]["evidence"]
        if cy is graph.A2_FEAS: return [dict(n=1)]
        if cy is graph.B1:
            out = []
            for x in m.values():
                if x["type"] != "conflict": continue
                e, k = x["evidence"], x["key"].split("|"); out.append(dict(method=k[1], dataset=k[2], metric=k[3], context=k[4], paper1=k[5], paper2=k[6], v1=x["meta"]["values"][0], v2=x["meta"]["values"][1],
                     page1=e[0]["page"], passage1=e[0]["passage"], page2=e[1]["page"], passage2=e[1]["passage"]))
            return out
        if cy is graph.B2_ITEMS:
            ents = {x["name"] for x in F if x["kind"] in ("problem", "method", "dataset", "context")}
            return [dict(fid=f"{f['paper']}|{core.alnum(f['text'])[:40]}", paper=f["paper"], text=f["text"], page=f["page"], passage=f["passage"], mentions=f["mentions"], matched=[x for x in f["mentions"] if x in ents]) for f in F if f["kind"] == "future"]
        if cy is graph.B2_LINKS:
            fw = next(f for f in F if f["kind"] == "future" and f"{f['paper']}|{core.alnum(f['text'])[:40]}" == kw["fid"]); hit = {}
            for f in F:
                if f["kind"] in ("problem", "method", "dataset", "context") and f["name"] in fw["mentions"] and f["paper"] != fw["paper"]: hit.setdefault(f["paper"], set()).add(f["name"])
            return [dict(paper=p_, hit=sorted(h)) for p_, h in hit.items()]
        if cy is graph.B2_EV: return []
        if cy is graph.PASSAGES: return [dict(page=x["page"], passage=x["passage"]) for x in core.passages_py(F, kw["id"])]
        raise AssertionError("unexpected query")
fs = Fake(); fs.ingest("t", res["papers"], res["facts"]); got = fs.detect_all("t")
chk(keys(got) == keys(core.detect_all_py(res["facts"])), "Store.ingest/detect_all plumbing returns the same 7 gaps as the mirror (fake driver)")
chk(all(k in graph.ING for k in ("problem", "method", "dataset", "context", "applied", "evaluated", "result", "limitation", "future")), "an ingestion statement exists for every fact kind")

if "--neo4j" in sys.argv:       # the real test of the Cypher, on your machine
    s = pipeline.get_store(); assert s, "set NEO4J_PASSWORD"; s.reset("selftest"); s.ingest("selftest", res["papers"], res["facts"])
    real = s.detect_all("selftest"); mirror = core.detect_all_py(res["facts"]); print("Neo4j stats:", s.stats("selftest"))
    chk(keys(real) == keys(mirror), f"REAL Neo4j Cypher returns the same {len(mirror)} gaps as the Python mirror")
    if keys(real) != keys(mirror): print("  only in Neo4j:", set(keys(real)) - set(keys(mirror)), "\n  only in mirror:", set(keys(mirror)) - set(keys(real)))
    s.reset("selftest"); chk(s.stats("selftest")["nodes"] == {}, "reset removes only the test project")
# ---- regression: StopIteration in name cleanup (3+ similar spellings of one name crashed the build at "Building the knowledge graph") ----
import random, time
mk = lambda name, p: dict(kind="applied", paper=p, page=1, passage="x.", method=name, problem="translation")
fx = [mk("Transformer", f"p{i}") for i in range(3)] + [mk("Transformers", f"q{i}") for i in range(2)] + [mk("Transformerss", "r0")]
try: o_, rv_, lg_ = core.normalize(fx, {}, None); crashed = False
except StopIteration: crashed = True
chk(not crashed and {f["method"] for f in o_} == {"Transformer"} and len(lg_) == 2, "name cleanup: three spellings of one name collapse into one entity (used to raise StopIteration)")
rnd = random.Random(1); base = [f"Method{i}Net" for i in range(120)] + ["BERT", "BiLSTM", "LSTM", "CNN", "SVM"]; nm = []
for _ in range(800):
    n_ = rnd.choice(base); r_ = rnd.random(); nm.append(n_ if r_ < .4 else n_ + rnd.choice(["s", "ss", "-base", " v2", ""]) if r_ < .8 else n_[:-1] + rnd.choice("xyz"))
calls_ = []; t0_ = time.time(); o2_, rv2_, _ = core.normalize([mk(n_, f"p{i}") for i, n_ in enumerate(nm)], {}, lambda t, a, b: calls_.append(1) or False)
chk(time.time() - t0_ < 10 and len(calls_) <= 15 and len(rv2_) <= 200, f"name cleanup on 800 noisy names: {time.time() - t0_:.1f}s, LLM confirmations capped ({len(calls_)} <= 15), review list capped ({len(rv2_)})")
chk(core.normalize([], {}, None)[0] == [] and core.normalize([mk("A", "p")], {}, None)[0][0]["method"] == "A", "name cleanup handles empty and single-name input")
print(f"\n{sum(results)}/{len(results)} checks passed"); sys.exit(0 if all(results) else 1)
