"""Drives the researcher app headlessly like a user, against fake Groq/OpenAlex servers.   python selftest_ui.py"""
import os, sys, json, shutil, tempfile, time
from pathlib import Path
HERE = Path(__file__).parent; TMP = Path(tempfile.mkdtemp()); os.environ["GAPFINDER_HOME"] = str(TMP / "home")
for k in ("GROQ_API_KEY", "NEO4J_PASSWORD"): os.environ.pop(k, None)
import testkit
fg = testkit.FakeGroq(default=lambda body: testkit.smart_content(body, HERE / "fixtures")); os.environ.update(GROQ_BASE_URL=fg.url)
import core, llm, sources, builder, pipeline, triage, settings, export
llm._sleep = lambda s: None; sources._sleep = lambda s: None; R = []
def chk(c, m): R.append(bool(c)); print(("PASS " if c else "FAIL ") + m)
stems = [f.stem for f in sorted((HERE / "samples").glob("*.pdf"))]; oa = testkit.FakeOpenAlex([], pdf_dir=HERE / "samples"); os.environ["OPENALEX_URL"] = oa.url
def abs_of(s): return core.parse_pdf(str(HERE / "samples" / f"{s}.pdf"))["chunks"][0]["text"].split("[Abstract]\n", 1)[1].split("\n[", 1)[0]
oa.works = [testkit.work(i + 1, core.parse_pdf(str(HERE / "samples" / f"{s}.pdf"))["title"], abs_of(s) + " Extra sentence for length. " * 3, pdf=f"{oa.url}/pdf/{s}.pdf") for i, s in enumerate(stems)]
from streamlit.testing.v1 import AppTest
APP = str(HERE / "app.py"); lab = lambda at, l: [w for w in at.text_input if w.label == l][0]; nav = lambda at, p: at.sidebar.radio(key="nav").set_value(p).run()

# ── first run: no project, no key ──
at = AppTest.from_file(APP, default_timeout=120).run(); chk(not at.exception and at.sidebar.button(key="create_first"), "first run asks for a project name")
at.sidebar.button(key="create_first").click().run(); chk(not at.exception and "my-research" in at.sidebar.selectbox[0].options, "project created")
chk(any("language model" in w.value for w in at.warning), "Build page tells the researcher a language model is needed (and offers the demo)")
# ── settings: enter key, test, save ──
nav(at, "Settings"); at.selectbox(key="prov").set_value("groq").run(); at.text_input(key="k_groq").set_value("gsk_test"); at.button(key="test_llm").click().run(); chk(not at.exception and any("Groq works" in s.value for s in at.success), "Settings: 'Test Groq' succeeds against the server")
at.text_input(key="m_groq").set_value("gone"); at.button(key="test_llm").click().run(); chk(any("not available" in e.value or "Suggested" in e.value for e in list(at.error) + list(at.info)), "Settings: a retired model is explained and a replacement suggested")
at.text_input(key="m_groq").set_value("llama-3.3-70b-versatile"); lab(at, "Your email (optional, gives faster OpenAlex access)").set_value("me@uni.edu"); at.button(key="save_settings").click().run()
chk(json.loads((TMP / "home" / "settings.json").read_text())["GROQ_API_KEY"] == "gsk_test" and os.getenv("GROQ_API_KEY") == "gsk_test", "Settings are saved to the user's data folder and applied")
# ── build: search, add, build ──
nav(at, "1 · Build corpus"); [w for w in at.text_input if w.label == "What are you researching?"][0].set_value("low-resource machine translation"); at.button(key="search").click().run()
chk(not at.exception and any("Add 6 selected papers" in b.label for b in at.button), "Search lists the papers found, all selected by default")
at.button(key="add_found").click().run(); chk(len(builder.list_papers("my-research")) == 6, "papers are added to the project")
at.button(key="build").click().run(); js = builder.wait("my-research", 90); at.run(); chk(js["state"] == "done" and not at.exception, f"build runs in the background and finishes ({js['summary']['facts']} facts)")
chk(any("Done:" in s.value for s in at.success) and len(at.dataframe) >= 1, "Progress panel shows the result and the per-paper table")
chk(any("Extraction quality" in e.label for e in at.expander), "Build page shows the extraction-quality panel")
at2_ = AppTest.from_file(APP, default_timeout=60); 
# ── explore ──
nav(at, "2 · Explore gaps"); cards = [e for e in at.expander if e.label.lstrip("⭐🕓✖ ").startswith("#")]; chk(not at.exception and len(cards) == 7, f"Explore shows the 7 gap cards ({len(cards)})")
res = pipeline.load("my-research"); key0 = core.detect_all_py(res["facts"])[0]["key"]; first_key = None
for b in at.button:
    if b.key and b.key.startswith("keep_"): first_key = b.key[5:]; b.click().run(); break
chk(triage.get_all("my-research").get(first_key, {}).get("status") == "keep", "Keep button persists the decision")
at.button(key="explain").click().run(); ex = json.loads((TMP / "home" / "projects" / "my-research" / "explanations.json").read_text()); chk(not at.exception and ex and all(v == "Mock explanation of the gap." for v in ex.values()), f"plain-language summaries written and cached ({len(ex)})")
at.button(key="verify_all").click().run(); vc = json.loads((TMP / "home" / "projects" / "my-research" / "verify_cache.json").read_text()); chk(not at.exception and len(vc) >= 5 and all("status" in v for v in vc.values()), f"external check ran and was cached ({len(vc)} gaps)")
# ── ask ──
nav(at, "3 · Ask the literature"); [w for w in at.text_input if w.key == "ask_q"][0].set_value("What BLEU does mBART reach on Swahili?"); at.button(key="ask").click().run()
h = json.loads((TMP / "home" / "projects" / "my-research" / "ask_history.json").read_text()); chk(not at.exception and h[0]["answered"] and h[0]["evidence"], "Ask: answer saved with a cited sentence")
# ── export ──
nav(at, "4 · Export"); chk(not at.exception and any("1 gaps selected" in m.value for m in at.markdown), "Export: only the kept gap is selected by default")
inc = [g for g in pipeline.finish(core.detect_all_py(res["facts"]), {p["id"]: p["title"] for p in res["papers"]}, "t", core.DEFAULT_W, False)][:2]; md = export.to_markdown(inc, {}, "p", "t", res["papers"]); bib = export.to_bibtex(res["papers"][:2])
chk("## References" in md and "CANDIDATE gaps" in md and bib.startswith("@article{") and "title = {" in bib, "Markdown report has references and the caveat; BibTeX entries are well-formed")
# ── delete a project safely ──
n_before = len(settings.projects_dir().iterdir.__self__.glob("*")) if False else len([p for p in settings.projects_dir().iterdir()])
at.sidebar.text_input(key="newname").set_value("scratch").run(); at.sidebar.button(key="create_proj").click().run(); chk("scratch" in at.sidebar.selectbox[0].options, "a second project can be created")
at.sidebar.text_input(key="delname").set_value("scratch").run(); at.sidebar.button(key="delete_proj").click().run(); chk(not (settings.projects_dir() / "scratch").exists() and "scratch" not in at.sidebar.selectbox[0].options, "deleting needs the typed name and removes only that project")
chk((settings.projects_dir() / "my-research" / "extraction.json").exists(), "other projects are untouched")
# ── instant demo with no key ──
for k in ("GROQ_API_KEY",): os.environ.pop(k, None)
shutil.rmtree(TMP / "home" / "settings.json", ignore_errors=True); (TMP / "home" / "settings.json").unlink(missing_ok=True)
os.environ["GAPFINDER_HOME"] = str(TMP / "home2"); at2 = AppTest.from_file(APP, default_timeout=120).run(); at2.sidebar.button(key="create_first").click().run(); at2.button(key="demo").click().run(); chk(not at2.exception and at2.sidebar.selectbox[0].value == "demo-synthetic", "demo loads without a key or internet")
nav(at2, "2 · Explore gaps"); chk(not at2.exception and len([e for e in at2.expander if e.label.lstrip("⭐🕓✖ ").startswith("#")]) == 7, "demo shows the 7 planted gaps")
print(f"\n{sum(R)}/{len(R)} UI checks passed"); oa.close(); fg.close(); shutil.rmtree(TMP, ignore_errors=True); sys.exit(0 if all(R) else 1)
