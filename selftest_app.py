"""Offline tests for the thesis features (conditions, harness, blinded rating, analysis, abstention, export, collector, both UIs).  python selftest_app.py
The LLM is MOCKED and the abstention data is SYNTHETIC: these tests check the code, they produce no research results."""
import os, sys, json, shutil, sqlite3, random, tempfile
from pathlib import Path
os.environ["GAPFINDER_HOME"] = tempfile.mkdtemp(); os.environ["GROQ_API_KEY"] = "test-key"; os.environ.pop("NEO4J_PASSWORD", None)
import core, llm, pipeline, conditions as C, experiment, analysis, abstain, export, synth, collect_arxiv, embed
HERE = Path(__file__).parent; results = []
def chk(c, m): results.append(bool(c)); print(("PASS " if c else "FAIL ") + m)

# ---- mocked LLM (dispatch on the system prompt) ----
def fake_json(system, user, max_tokens=1800, schema=None):
    if system.startswith("You propose research gaps"):
        lines = [l for l in user.split("EXCERPTS:\n", 1)[1].split("\n") if l.startswith("[")]; out = []
        for l in lines[:2]: pid = l[1:l.index(" |")]; out.append(dict(title=f"Open question from {pid[:9]}", explanation="Mock explanation.", evidence=[dict(paper=pid, passage=l.split("] ", 1)[1].split(". ")[0] + ".")]))
        return dict(gaps=out)
    if system.startswith("You rewrite research-gap"): return dict(title="Neutral: " + user.split("\n")[0].replace("Title: ", "")[:80], description="Neutral description.")
    if system.startswith("You are a strict reviewer"): return dict(support="yes", valid="yes", novel="no")
    return {}
llm.chat_json = fake_json; llm.chat_text = lambda s, u, max_tokens=400: "Mock explanation."
samples = sorted((HERE / "samples").glob("*.pdf")); PROJ = {"tst_a": samples[:4], "tst_b": samples[2:]}
for p, pdfs in PROJ.items(): r = pipeline.extract_all(pdfs, "fixture", False, log=lambda *_: None); pipeline.save(p, r); pipeline.project_meta(p, topic="low-resource machine translation")

# ---- conditions ----
res = pipeline.load("tst_a"); corpus = pipeline.load_corpus("tst_a"); chunks = C.build_chunks(corpus); ix = embed.Index([c["text"] for c in chunks]); real = corpus[0]["sentences"][2]
llm.chat_json = lambda s, u, max_tokens=1800, schema=None: dict(gaps=[dict(title="ok", evidence=[dict(paper=corpus[0]["id"], passage=real)]), dict(title="invented", evidence=[dict(paper=corpus[0]["id"], passage="Nothing like this appears in the paper at all, honestly.")]),
                                                         dict(title="wrong paper", evidence=[dict(paper="no_such_paper", passage=real)])])
r = C.run_cond("V", corpus=corpus, chunks=chunks, index=ix, facts=res["facts"], topic="mt", n=8)
chk(r["n_raw"] == 3 and [c["title"] for c in r["cands"]] == ["ok"], "provenance gate keeps the real citation, drops the invented passage and the unknown paper")
llm.chat_json = fake_json
chk(len(C.graph_context(res["facts"], {corpus[0]["id"]}).splitlines()) > 3, "hybrid context expands over shared entities")
try: C.run_cond("H", corpus=corpus, chunks=chunks, index=ix, facts=res["facts"], topic="mt", use_llm=False); chk(False, "V/H without LLM must raise")
except RuntimeError: chk(True, "conditions V/H refuse to run without an LLM instead of faking an answer")

# ---- experiment harness + blinding ----
root = HERE / "experiments"; d = experiment.run(["tst_a", "tst_b"], n=5, use_llm=True, root=root, log=lambda *_: None)
blind = json.loads((d / "pool_blind.json").read_text(encoding="utf-8")); key = json.loads((d / "pool_key.json").read_text(encoding="utf-8")); run = json.loads((d / "run.json").read_text(encoding="utf-8"))
chk(len(blind) == len(key) == run["n_items"] > 0, f"pool has {len(blind)} items; every item has a key entry")
txt = (d / "pool_blind.json").read_text(encoding="utf-8")
chk(not any(w in txt for w in ('"condition"', '"origins"', '"rank"', '"key"')) and all(b["title"].startswith("Neutral:") for b in blind), "blind pool shows no condition/rank/key and every title went through the same neutral rewrite")
chk(any({"V", "H"} <= {o["condition"] for o in k["origins"]} for k in key.values()), "identical V and H candidates are merged into one item (de-duplication)")
chk(all(s["n_verified"] <= s["n_raw"] and "prompt_tokens" in s for s in run["stats"]), "run.json logs raw vs verified counts and token usage per condition and topic")

# ---- simulated raters -> analysis ----
rnd = random.Random(1); con = sqlite3.connect(d / "ratings.sqlite"); con.execute(analysis.SCHEMA)
for b in blind:
    base = rnd.random()
    for rater in ("r1", "r2"):
        flip = rnd.random() < .2; good = (base > .4) != flip
        con.execute("INSERT INTO ratings VALUES (?,?,?,?,?,?,?,?,?)", (b["item_id"], rater, "yes" if good else "no", "yes" if good else "no", rnd.choice(["yes", "no", "unsure"]), rnd.randint(1, 5), "other", "", "t"))
con.commit(); con.close(); a = analysis.analyze(d, n_boot=300)
chk(set(a["conditions"]) == {"G", "H", "V"} and a["items_rated_by_all"] == len(blind), "analysis covers V, G, H and all items rated by both raters")
chk(all(0 <= (m["strict"]["est"] or 0) <= 1 for m in a["conditions"].values()) and "r1 vs r2" in a["kappa"] and len(a["differences"]) == 3, "strict precision in [0,1], kappa for the rater pair, three paired differences")
chk(isinstance(a["conditions"]["G"]["strict"]["ci95"], list), "bootstrap confidence interval is produced")
j = analysis.judge(d, log=lambda *_: None); chk(j["n"] == len(blind) and j["agreement_strict"] is not None, "LLM-judge comparison runs and reports agreement with experts")
chk(analysis.kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == 1.0 and analysis.kappa(["a", "b"], ["b", "a"]) == -1.0, "Cohen's kappa is correct on perfect agreement and perfect disagreement")

# ---- abstention (SYNTHETIC facts) ----
fb = {f"t{i}": synth.make_facts(60, seed=i) for i in range(4)}; rows = abstain.build_rows(fb); ev = abstain.evaluate(rows); fr = ev["false_rate_by_withheld_fraction"]
chk(fr["0.4"] > fr["0.2"], f"false-gap rate rises with the withheld fraction on synthetic data ({fr})")
chk(ev["auroc_model"] is not None and len(ev["risk_coverage_model"]) == 5, "leave-one-topic-out evaluation returns AUROC and risk-coverage curves")
m = abstain.train(rows); tmp = Path(tempfile.mkdtemp()) / "m.json"; abstain.save_model(tmp, m); g0 = abstain.structural(fb["t0"])[0]
chk(abs(abstain.confidence(g0, fb["t0"], abstain.load_model(tmp))["conf"] - abstain.confidence(g0, fb["t0"], m)["conf"]) < 1e-9 and abstain.confidence(g0, fb["t0"])["basis"].startswith("heuristic"), "model round-trips to disk; heuristic fallback is labelled")
chk(abstain.auroc([3, 2, 1], [1, 1, 0]) == 1.0 and abstain.auroc([1, 1], [1, 0]) == .5, "AUROC correct on a perfect ranking and on ties")

# ---- export ----
ranked = pipeline.finish(core.detect_all_py(res["facts"]), {p["id"]: p["title"] for p in res["papers"]}, "mt", core.DEFAULT_W, False)
csv_ = export.to_csv(ranked, {}); md = export.to_markdown(ranked, {}, "p", "mt")
chk(csv_.count("\n") == len(ranked) + 1 and "CANDIDATE gaps" in md and ranked[0]["evidence"][0]["passage"][:30] in md, "CSV has one row per gap; Markdown carries the caveat and the evidence passages")

# ---- arXiv collector (XML parsing only; no network) ----
xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2301.00001v2</id><published>2023-01-02T10:00:00Z</published><title>A  Title\n with newline</title><link title="pdf" href="http://arxiv.org/pdf/2301.00001v2"/></entry>'
       '<entry><id>http://arxiv.org/abs/cs/0601001v1</id><published>2006-01-01T00:00:00Z</published><title>Old style</title></entry></feed>'); ps = collect_arxiv.parse_feed(xml)
chk([p["stem"] for p in ps] == ["2301.00001", "cs_0601001"] and ps[0]["title"] == "A Title with newline" and ps[1]["pdf_url"].endswith("cs/0601001"), "arXiv Atom parsing: ids, old-style ids, titles, pdf links")
chk(collect_arxiv.build_query("low-resource machine translation", "cs.CL", "2019-01-01", "2023-12-31").endswith("submittedDate:[201901010000 TO 202312312359]"), "arXiv date-window query is built correctly")

# ---- both UIs, headless ----
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(str(HERE / "research_app.py"), default_timeout=180).run(); chk(not at.exception, "main app starts")
at.radio[0].set_value("Offline fixtures (sample papers only)").run(); at.button(key="extract").click().run(); chk(not at.exception and any(m.label == "Facts kept" and m.value == "73" for m in at.metric), "extraction in the UI keeps 73 facts")
at.button(key="detect").click().run(); cards = [e.label for e in at.expander if e.label.startswith("#")]; chk(not at.exception and len(cards) == 7, f"condition G in the UI shows 7 cards ({len(cards)})")
at.radio[1].set_value("V: vector RAG").run(); at.button(key="detect").click().run(); chk(not at.exception and any("Open question" in e.label for e in at.expander), "condition V in the UI shows LLM-proposed, provenance-checked cards")
at.button(key="run_withhold").click().run(); chk(not at.exception, "withholding experiment runs from the UI")
at2 = AppTest.from_file(str(HERE / "annotate_app.py"), default_timeout=60).run(); chk(not at2.exception and any("name or id" in i.value for i in at2.info), "rating app asks for a rater name first")
at2.sidebar.text_input[0].set_value("tester").run(); rd = at2.radio; [rd[0].set_value("yes"), rd[1].set_value("yes"), rd[2].set_value("no"), rd[3].set_value(4)]
[b for b in at2.button if b.label == "Save and next"][0].click().run(); n = sqlite3.connect(sorted(root.glob("*/ratings.sqlite"))[-1].parent / "ratings.sqlite").execute("SELECT COUNT(*) FROM ratings WHERE rater='tester'").fetchone()[0]
chk(not at2.exception and n == 1, "rating app saves a rating to SQLite and moves on")
shutil.rmtree(root, ignore_errors=True); (HERE / "abstention_model.json").unlink(missing_ok=True)
print(f"\n{sum(results)}/{len(results)} checks passed"); sys.exit(0 if all(results) else 1)
