"""Realistic end-to-end tests: the real Groq SDK and real HTTP against fake servers that misbehave like the real services,
hard synthetic PDFs, and the failure modes of real use. No internet, Neo4j or API key needed.   python selftest_real.py"""
import os, sys, json, shutil, tempfile, time, threading
from pathlib import Path
HERE = Path(__file__).parent; TMP = Path(tempfile.mkdtemp()); os.environ["GAPFINDER_HOME"] = str(TMP / "home")
import testkit
fg = testkit.FakeGroq(default=lambda body: testkit.smart_content(body, HERE / "fixtures")); os.environ.update(GROQ_API_KEY="gsk_test", GROQ_BASE_URL=fg.url, GROQ_MODEL="llama-3.3-70b-versatile")
import core, llm, sources, builder, pipeline, qa, triage, conditions as C, embed, abstain
llm._sleep = lambda s: None; sources._sleep = lambda s: None; R = []
def chk(c, m): R.append(bool(c)); print(("PASS " if c else "FAIL ") + m)

# ───── A. Groq wrapper vs misbehaving server (real SDK) ─────
plain = testkit.FakeGroq(); os.environ["GROQ_BASE_URL"] = plain.url; A = lambda: llm.chat_json("s", "u"); sl = []; llm._sleep = lambda s: sl.append(round(s, 2))
plain.queue = [(200, {}, plain.ok('```json\n{"a": 1}\n```'))]; chk(A() == {"a": 1}, "A1 fenced JSON parsed")
plain.queue = [(200, {}, plain.ok('{"gaps": [{"title": "x", "evidence": [{"paper": "p", "passage": "hello wor', "length"))]; chk("gaps" in A(), "A2 truncated JSON repaired")
plain.queue = [(429, {"retry-after": "2"}, {"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}}), (200, {}, plain.ok('{"b": 1}'))]; sl.clear(); chk(A() == {"b": 1} and sl == [2.5], "A3 429 honours retry-after")
plain.queue = [plain.err(429, "Rate limit reached for tokens per day (TPD). Please try again in 7m12.5s.", "rate_limit_exceeded")]
try: A(); chk(False, "A4")
except llm.LLMRateLimit as e: chk("Resume" in str(e), "A4 daily limit -> LLMRateLimit (no hour-long sleep)")
plain.queue = [plain.err(400, "Failed to generate JSON.", "json_validate_failed", failed_generation='{"e": 5,}')]; chk(A() == {"e": 5}, "A5 json_validate_failed recovered from failed_generation")
plain.queue = [plain.err(400, "response_format json_object is not supported with this model")]; n0 = len(plain.requests); plain.queue.append((200, {}, plain.ok('{"f": 1}'))); r = A()
chk(r == {"f": 1} and "response_format" not in plain.requests[n0 + 1], "A6 unsupported JSON mode -> retried without it")
for st_, msg, code, exc in [(401, "Invalid API Key", "invalid_api_key", llm.LLMConfigError), (404, "The model `x` does not exist", "model_not_found", llm.LLMModelError), (413, "Request too large for model on tokens per minute", "x", llm.LLMTooLarge)]:
    plain.queue = [plain.err(st_, msg, code)]
    try: A(); chk(False, f"A7 {st_}")
    except exc: chk(True, f"A7 {st_} -> {exc.__name__}")
plain.queue = [(200, {}, plain.ok("nope")), (200, {}, plain.ok("still nope"))]
try: A(); chk(False, "A8")
except llm.LLMError as e: chk("valid JSON" in str(e), "A8 two non-JSON replies -> clear error")
chk(llm.pick_model(["whisper-large-v3", "llama-guard-4", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]) == "llama-3.3-70b-versatile", "A9 model picker avoids speech/guard models")
os.environ["GROQ_MODEL"] = "retired"; c = llm.check(); chk(not c["ok"] and c["model"], "A10 check() flags a retired model and suggests one"); os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"; plain.close()
os.environ["GROQ_BASE_URL"] = fg.url; llm._sleep = lambda s: None

# ───── B. PDF parser on hard layouts ─────
import pymupdf
d = pymupdf.open(); W, H = 595, 842
for pno in range(3):
    pg = d.new_page(width=W, height=H); pg.insert_text((150, 30), "Proceedings of the Synthetic Workshop 2024", fontsize=8); pg.insert_text((290, 820), str(pno + 1), fontsize=9); pg.insert_text((15, 600), "arXiv:2401.00001v1 [cs.CL] 2 Jan 2024", fontsize=9, rotate=90)
    if pno == 0:
        pg.insert_text((60, 90), "Comparing Back-translation and Adapters", fontsize=20); pg.insert_text((60, 118), "for Low-Resource Translation", fontsize=20)
        pg.insert_text((50, 170), "Abstract", fontsize=11); pg.insert_text((50, 184), "We study low-resource machine trans-\nlation for Swahili and show that back-\ntranslation improves accuracy.", fontsize=10)
        pg.insert_text((50, 240), "1 Introduction", fontsize=11); pg.insert_text((50, 254), "Low-resource settings remain hard.\nParallel data is scarce for many lan-\nguages, so transfer matters.", fontsize=10)
        pg.insert_text((310, 170), "2 Method", fontsize=11); pg.insert_text((310, 184), "We fine-tune mBART with low-resource\ndata and add synthetic pairs.", fontsize=10); pg.insert_text((310, 240), "3 Results", fontsize=11); pg.insert_text((310, 254), "mBART reaches 22.4 BLEU on Swahili.", fontsize=10)
        pg.insert_text((310, 300), "References", fontsize=11); pg.insert_text((310, 314), "[1] Some Author. A paper. 2020.", fontsize=10)
        for i in range(8): pg.insert_text((50, 360 + i * 14), f"Left filler line {i} about datasets and methods.", fontsize=10); pg.insert_text((310, 360 + i * 14), f"Right filler line {i} about results and metrics.", fontsize=10)
hard = TMP / "hard.pdf"; d.save(hard); d.close(); p = core.parse_pdf(str(hard), depth="full"); txt = " ".join(c["text"] for c in p["chunks"]); i = txt.find
chk(p["title"] == "Comparing Back-translation and Adapters for Low-Resource Translation", "B1 title found by font size")
chk("Proceedings" not in txt and "arXiv:2401" not in txt, "B2 running header and rotated arXiv stamp removed")
chk("trans-" not in txt and "lan-" not in txt and "low-resource machine translation" in txt.lower() and "back-translation improves" in txt, "B3 line-break hyphens joined, real compounds kept")
chk(0 < i("We study low-resource") < i("Low-resource settings") < i("Left filler line 0") < i("fine-tune mBART") < i("mBART reaches") and "Right filler" not in txt, "B4 two-column reading order, references excluded")
d2 = pymupdf.open(); pg = d2.new_page(); pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False); pix.set_rect(pix.irect, (200, 200, 200)); pg.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix); d2.save(TMP / "scan.pdf")
q = core.parse_pdf(str(TMP / "scan.pdf")); chk(any("scanned" in f for f in q["quality"]["flags"]) and not q["chunks"], "B5 image-only PDF flagged")

# ───── C. Paper discovery + resumable build through HTTP ─────
pdfdir = TMP / "pdfs"; pdfdir.mkdir(); stems = []
for f in sorted((HERE / "samples").glob("*.pdf")): shutil.copy(f, pdfdir / f.name); stems.append(f.stem)
(pdfdir / "corrupt.pdf").write_bytes(b"%PDF-1.4\nthis is not a real pdf at all")
def full_text(stem): return core.parse_pdf(str(HERE / "samples" / f"{stem}.pdf"))["chunks"][0]["text"]
def abstract_of(stem): t = core.parse_pdf(str(HERE / "samples" / f"{stem}.pdf"))["chunks"][0]["text"]; return t.split("[Abstract]\n", 1)[1].split("\n[", 1)[0]
oa = testkit.FakeOpenAlex([], pdf_dir=pdfdir); os.environ["OPENALEX_URL"] = oa.url
works = [testkit.work(i + 1, core.parse_pdf(str(HERE / "samples" / f"{s}.pdf"))["title"], abstract_of(s) + " " + "Extra abstract sentence for length. " * 3, pdf=f"{oa.url}/pdf/{s}.pdf") for i, s in enumerate(stems)]
works += [testkit.work(7, "Paywalled paper", "We study something with a paywalled full text and a long enough abstract to be useful here. " * 2, pdf=f"{oa.url}/html"),
          testkit.work(8, "Corrupt file paper", "We study something whose PDF is corrupt but with a usable abstract that is long enough. " * 2, pdf=f"{oa.url}/pdf/corrupt.pdf")]
oa.works = works; found = sources.search("low-resource machine translation", 2019, 2024, n=20)
chk(len(found) == 8 and all(f["abstract"] for f in found), f"C1 search finds {len(found)} papers with abstracts")
P = "proj"; builder.add_papers(P, found); n_req0 = len(fg.requests)
# daily limit hits during the 3rd extraction -> the job pauses, keeps its progress
calls = {"n": 0}; base = fg.default
def limited(body):
    if body["messages"][0]["content"].startswith("You extract facts"):
        calls["n"] += 1
        if calls["n"] == 3: raise RuntimeError("limit")
    return base(body)
orig_default = fg.default
class Gate:                                                                   # make the server answer 429 (long wait) on the 3rd extraction call
    pass
def install_limit():
    fg.default = lambda body: (_ for _ in ()).throw(RuntimeError) if False else base(body)
cnt = {"n": 0}; orig_post_queue = fg.queue
# simplest: pre-load the queue so the 3rd extraction request is a 429 with a 20-minute hint
fg.queue = [(200, {}, fg.ok(testkit.smart_content({"messages": [{"content": "You extract facts"}, {"content": "TEXT:\n"}]}, HERE / "fixtures"))) for _ in range(0)]
sc = fg.default; seq = {"n": 0}
def counting(body):
    if body["messages"][0]["content"].startswith("You extract facts"): seq["n"] += 1
    return sc(body)
fg.default = counting
# inject the limit as a queued error before the 3rd call by wrapping: queue entries are consumed first, so push two normal answers then the error
fg.queue = [(200, {}, fg.ok(sc({"messages": [{"content": "You extract facts"}, {"content": "TEXT:\n" + full_text(stems[0])}]}))), (200, {}, fg.ok(sc({"messages": [{"content": "You extract facts"}, {"content": "TEXT:\n" + full_text(stems[1])}]}))),
            fg.err(429, "Rate limit reached for tokens per day (TPD). Please try again in 20m0s.", "rate_limit_exceeded")]
builder.start_build(P, depth="standard", use_llm=True); js = builder.wait(P, 60); ps = builder.list_papers(P)
chk(js["state"] == "paused" and "Resume" in js["message"] and sum(x["state"] == "done" for x in ps) == 2, f"C2 daily limit -> job paused, 2 papers saved ({js['state']})")
fg.queue = []; n_before = len(fg.requests); builder.start_build(P, depth="standard", use_llm=True); js = builder.wait(P, 90); ps = builder.list_papers(P); extr = sum(1 for r in fg.requests[n_before:] if r["messages"][0]["content"].startswith("You extract facts"))
chk(js["state"] == "done" and extr == 6, f"C3 Resume continues: finished with {extr} new extraction calls, none repeated for the 2 saved papers")
by = {x["title"]: x for x in ps}; chk(by["Paywalled paper"]["source"] == "abstract" and any("web page" in f for f in by["Paywalled paper"]["flags"]), "C4 paywalled PDF -> falls back to the abstract and says why")
chk(by["Corrupt file paper"]["state"] == "done" and by["Corrupt file paper"]["source"] == "abstract" and any("could not be read" in f for f in by["Corrupt file paper"]["flags"]), "C5 corrupt PDF -> falls back to the abstract, flagged, job continues")
res = pipeline.load(P); chk(res is not None and len(res["papers"]) >= 6, f"C6 assembled corpus: {len(res['papers'])} papers, {len(res['facts'])} facts")
gaps = core.detect_all_py(pipeline.apply_review(res["facts"], pipeline.load_review(P))); chk(len([g for g in gaps if g["type"] == "method_problem"]) == 2 and len([g for g in gaps if g["type"] == "future_work"]) == 3 and len(gaps) == 7, f"C7 the 7 planted gaps found via search -> download -> parse -> Groq -> graph ({len(gaps)} gaps)")
chk(all(e["page"] and e["passage"] for g in gaps for e in g["evidence"]), "C8 every evidence item still has page and passage")
st_ = builder.job_status(P); chk(st_["state"] == "done" and st_["summary"]["abstract_only"] >= 1, f"C9 job summary persisted: {st_['summary']}")
json.dump(dict(state="running", total=3, done=1, updated=time.time()), open(builder._jf("ghost"), "w")); chk(builder.job_status("ghost")["state"] == "interrupted", "C10 a 'running' job with no worker is reported as interrupted")

# ───── D. isolation of bad model output, uploads, alias cleanup, cancel ─────
P2 = "proj2"; builder.add_papers(P2, found[:1]); fg.queue = [(200, {}, fg.ok("not json")), (200, {}, fg.ok("still not json"))]; builder.start_build(P2); builder.wait(P2, 60); x = builder.list_papers(P2)[0]
chk(x["state"] == "done" and any("valid JSON" in f for f in x["flags"]), "D1 a chunk the model cannot answer is skipped and flagged; the paper still completes")
P3 = "proj3"; pid = builder.add_upload(P3, "My Paper.pdf", (HERE / "samples" / f"{stems[3]}.pdf").read_bytes()); builder.start_build(P3); builder.wait(P3, 60)
chk(builder.list_papers(P3)[0]["state"] == "done" and pipeline.load(P3)["facts"], "D2 an uploaded PDF goes through the same pipeline")
rawf = pipeline.pdir(P) / "papers"; target = rawf / "W2"; raw = json.loads((target / "raw.json").read_text(encoding="utf-8"))
for k in ("methods", "applications", "evaluations", "results"):
    for it in raw[k]:
        for f_ in ("name", "method"):
            if it.get(f_) == "mBART": it[f_] = "multilingual-BART-50"
(target / "raw.json").write_text(json.dumps(raw), encoding="utf-8"); g1 = core.detect_all_py(builder.assemble(P)["facts"]); builder.add_alias(P, "Method", "mBART", "multilingual-BART-50"); g2 = core.detect_all_py(builder.assemble(P)["facts"])
k_ = lambda gs: sorted(x["key"] for x in gs); chk(len(g2) == 7 and k_(g1) != k_(g2), f"D3 a name variant changes the gaps ({len(g1)} vs {len(g2)}); one alias click merges it back")
fg.default = lambda body: (time.sleep(.4), testkit.smart_content(body, HERE / "fixtures"))[1]; P4 = "proj4"; builder.add_papers(P4, found[:6]); builder.start_build(P4); time.sleep(.6); builder.cancel(P4); js = builder.wait(P4, 30); n_done = sum(x["state"] == "done" for x in builder.list_papers(P4))
chk(js["state"] == "cancelled" and 0 < n_done < 6, f"D4 cancel stops the job and keeps progress ({n_done}/6 done)")
fg.default = lambda body: testkit.smart_content(body, HERE / "fixtures"); builder.start_build(P4); js = builder.wait(P4, 90); chk(js["state"] == "done" and all(x["state"] == "done" for x in builder.list_papers(P4)[:6]), "D5 resume after cancel completes the rest")

# ───── E. ask the corpus, triage ─────
corpus = pipeline.load_corpus(P); facts = pipeline.load(P)["facts"]; chunks = C.build_chunks(corpus); ix = embed.Index([c["text"] for c in chunks])
a = qa.ask("What BLEU does mBART reach on Swahili?", corpus, chunks, ix, facts); chk(a["answered"] and a["evidence"] and a["evidence"][0]["page"], "E1 answer comes with a verified, paged citation")
a = qa.ask("UNANSWERABLE what is the capital of France", corpus, chunks, ix, facts); chk(not a["answered"] and "Not enough evidence" in a["answer"], "E2 unanswerable question -> explicit abstention")
fg.default = lambda body: json.dumps(dict(answerable=True, answer="Confident but invented.", evidence=[dict(paper=corpus[0]["id"], passage="This sentence does not exist in the paper at all, honestly.")]))
a = qa.ask("What BLEU does mBART reach on Swahili?", corpus, chunks, ix, facts); chk(not a["answered"], "E3 a confident answer with an invented citation is rejected by the provenance gate")
fg.default = lambda body: testkit.smart_content(body, HERE / "fixtures")
triage.set_(P, "k1", "keep", "check P4"); triage.set_(P, "k1", note="updated"); triage.set_(P, "k2", "dismiss"); t = triage.get_all(P)
chk(t["k1"] == dict(status="keep", note="updated") and t["k2"]["status"] == "dismiss", "E4 triage keeps status and notes (update note keeps status)")
# ───── G. failures after the papers are read (the "Building the knowledge graph..." step) ─────
P5 = "proj5"; builder.add_papers(P5, found[:2]); real_norm = core.normalize
def boom(*a, **k): raise StopIteration
core.normalize = boom; builder.start_build(P5); js = builder.wait(P5, 60); res5 = pipeline.load(P5)
chk(js["state"] == "done" and res5["facts"] and "Name cleanup failed" in res5["norm_error"] and (pipeline.pdir(P5) / "error.log").exists(), "G1 a crash in name cleanup no longer loses the corpus: facts are kept, the problem is reported, error.log written")
core.normalize = real_norm; real_assemble = builder.assemble; builder.assemble = boom; (pipeline.pdir(P5) / "papers" / "W1" / "status.json").unlink(missing_ok=True)
builder.add_papers(P5, found[2:4]); builder.start_build(P5); js = builder.wait(P5, 60)
chk(js["state"] == "failed" and "StopIteration" in js["message"] and "(at " in js["message"] and "Resume" in js["message"], f"G2 any other failure in this step says WHAT and WHERE, and that the papers are saved ({js['message'][:80]}...)")
builder.assemble = real_assemble; n_before = len(fg.requests); builder.start_build(P5); js = builder.wait(P5, 60); n_new = sum(1 for q_ in fg.requests[n_before:] if q_["messages"][0]["content"].startswith("You extract facts"))
chk(js["state"] == "done" and n_new == 0 and len(pipeline.load(P5)["papers"]) >= 2, "G3 after fixing the code, Resume only rebuilds the graph: 0 papers are read again, 0 tokens spent")

# ───── F. concurrency: the UI polls while the worker writes ─────
import threading as _th
pj = pipeline.pdir("race"); (pj / "papers").mkdir(exist_ok=True); stop = _th.Event(); errs = []
def writer():
    i = 0
    while not stop.is_set(): builder._jset("race", state="running", done=i, message="x" * 500); i += 1
def reader():
    for _ in range(3000):
        try: s = builder.job_status("race"); assert s["state"] in ("running", "interrupted", "idle")
        except Exception as e: errs.append(repr(e))
w = _th.Thread(target=writer); w.start(); reader(); stop.set(); w.join()
chk(not errs, f"F1 3000 status reads while a worker writes continuously: {len(errs)} errors")
print(f"\n{sum(R)}/{len(R)} realistic checks passed"); oa.close(); fg.close(); shutil.rmtree(TMP, ignore_errors=True); sys.exit(0 if all(R) else 1)
