"""GapFinder: find credible research gaps in the literature you care about. Run:  streamlit run app.py"""
import os, json, time, datetime
from pathlib import Path
import pandas as pd, streamlit as st
from dotenv import load_dotenv; load_dotenv()
import settings; settings.apply()
import core, llm, pipeline, graph, builder, sources, triage, qa, conditions as C, embed, abstain, export

st.set_page_config(page_title="GapFinder", page_icon="🔭", layout="wide")
HERE = Path(__file__).parent; S = st.session_state
TYPES = {"method_problem": "Method never tried on a problem", "context_problem": "Setting never studied with a problem", "conflict": "Conflicting results", "future_work": "Future work nobody has done"}
WHY = {"method_problem": "The method may have been used under another name, or in papers outside this corpus. Check the external search before relying on it.",
       "context_problem": "These two topics may be studied together in papers that are not in your corpus. Add papers on both topics, or search below.",
       "conflict": "Different results can come from different settings that the papers do not state. Read both passages before concluding.",
       "future_work": "Authors often write future work that is never published, or published later. Check whether a newer paper did it."}
now = datetime.date.today().year

def projects(): return sorted(p.name for p in settings.projects_dir().iterdir() if p.is_dir())
@st.cache_resource(show_spinner=False)
def connect(uri, user, pw, db):
    try: s = graph.Store(uri, user, pw, db or None); s.check(); return s, None
    except Exception as e: return None, f"{type(e).__name__}: {str(e)[:160]}"
def get_store():
    if os.getenv("GAPFINDER_USE_NEO4J") == "1" and os.getenv("NEO4J_PASSWORD"): return connect(os.getenv("NEO4J_URI", "bolt://localhost:7687"), os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"], os.getenv("NEO4J_DATABASE"))[0]
def jread(project, name, default): f = pipeline.pdir(project) / name; return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default
def jwrite(project, name, obj): (pipeline.pdir(project) / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")

# ───────── sidebar ─────────
with st.sidebar:
    st.title("🔭 GapFinder"); st.caption("Find credible research gaps, with the evidence.")
    names = projects()
    if not names:
        n0 = st.text_input("Name your first project", "my-research", help="One project per research topic."); 
        if st.button("Create project", type="primary", key="create_first"): pipeline.pdir(n0.strip() or "my-research"); S.project = n0.strip() or "my-research"; st.rerun()
        st.stop()
    project = st.selectbox("Project", names, index=names.index(S.get("project", names[0])) if S.get("project", names[0]) in names else 0); S.project = project
    with st.expander("New project"):
        nn = st.text_input("Project name", key="newname")
        if st.button("Create", key="create_proj") and nn.strip(): pipeline.pdir(nn.strip()); S.project = nn.strip(); st.rerun()
    with st.expander("Delete this project"):
        sure = st.text_input("Type the project name to confirm", key="delname").strip() == project
        if st.button("Delete project and its data", disabled=not sure, key="delete_proj"):
            import shutil
            s_ = get_store()
            if s_: s_.reset(project)
            shutil.rmtree(pipeline.pdir(project), ignore_errors=True); S.pop("project", None); S.pop("gk", None); st.rerun()
    nav = st.radio("Step", ["1 · Build corpus", "2 · Explore gaps", "3 · Ask the literature", "4 · Export", "Settings"], key="nav")
    store = get_store(); st.divider()
    st.caption(("✅ " + llm.label() if llm.available() else "⚠ No language model yet (Settings)") + " · " + ("✅ Neo4j graph" if store else "local graph engine"))
    res0 = pipeline.load(project)
    if res0: st.caption(f"{len(res0['papers'])} papers · {len(res0['facts'])} facts in this project")

meta = pipeline.project_meta(project)

# ───────── helpers ─────────
def rebuild_graph():
    s = get_store()
    if s: res = pipeline.load(project); s.reset(project); s.ingest(project, res["papers"], pipeline.apply_review(res["facts"], pipeline.load_review(project)))
def get_ranked(force=False):
    res = pipeline.load(project)
    if not res: return None, None
    facts = pipeline.apply_review(res["facts"], pipeline.load_review(project)); key = (project, len(facts), hash(tuple(f.get("id") for f in facts)))
    if force or S.get("gk") != key:
        try: gaps = pipeline.detect(dict(facts=facts), project, store, llm.available())
        except Exception as e: st.warning(f"Neo4j query failed ({type(e).__name__}); using the local engine instead."); gaps = core.detect_all_py(facts, llm.available())
        vc = jread(project, "verify_cache.json", {})
        for g in gaps:
            if g["key"] in vc: g["verification"] = vc[g["key"]]
        S.gk, S.graw = key, gaps
    return S.graw, facts
def rank(gaps, facts, W):
    titles = {p["id"]: p["title"] for p in pipeline.load(project)["papers"]}; ranked = pipeline.finish(gaps, titles, meta["topic"], W, False); ex = jread(project, "explanations.json", {}); idx = abstain.Idx(facts); model = abstain.load_model(HERE / "abstention_model.json")
    for g in ranked:
        g["explanation"] = ex.get(g["key"], g["explanation"]); g["confidence"] = abstain.confidence(g, facts, model, idx)
    return ranked, titles

# ───────── page: build ─────────
def page_build():
    st.header("1 · Build your corpus")
    if not llm.available(): st.warning("To read papers you need a language model. Open **Settings**: a free Google Gemini key is the easiest. Or try the demo below, which needs no key.")
    papers = builder.list_papers(project)
    with st.expander("A. Find papers by topic", expanded=not papers):
        q = st.text_input("What are you researching?", meta.get("query", meta["topic"]), key="q"); c1, c2, c3 = st.columns([2, 1, 1])
        yrs = c1.slider("Publication years", 2000, now, (now - 6, now)); n = c2.number_input("How many papers", 10, 200, 40, step=10); srt = c3.radio("Order by", ["relevance", "most cited"], horizontal=True)
        if st.button("Search", type="primary", key="search"):
            pipeline.project_meta(project, topic=q, query=q)
            with st.spinner("Searching OpenAlex..."):
                try: S.found = sources.search(q, yrs[0], yrs[1], int(n), "cited" if srt == "most cited" else "relevance")
                except sources.SourceError as e: st.error(str(e))
        if S.get("found"):
            df = pd.DataFrame([dict(select=True, title=f["title"], year=f["year"], cited=f["cited_by"], text="full text link" if f["pdf_urls"] else "abstract only", id=f["id"]) for f in S.found])
            ed = st.data_editor(df, hide_index=True, key="found_editor", disabled=["title", "year", "cited", "text", "id"], column_order=["select", "title", "year", "cited", "text"], height=300)
            st.caption("Papers without a full-text link are read from their abstract only, which gives shallower gaps.")
            if st.button(f"Add {int(ed["select"].sum())} selected papers", key="add_found"):
                chosen = set(ed[ed["select"]]["id"]); builder.add_papers(project, [f for f in S.found if f["id"] in chosen]); S.pop("found"); st.rerun()
    with st.expander("B. Add your own PDFs"):
        up = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
        if up and st.button(f"Add {len(up)} PDFs", key="add_up"): [builder.add_upload(project, f.name, f.getbuffer().tobytes()) for f in up]; st.rerun()
    if not papers:
        if st.button("Try the demo (6 synthetic papers, no key needed)", key="demo"):
            with st.spinner("Loading demo..."): builder.load_demo("demo-synthetic", HERE / "samples")
            S.project = "demo-synthetic"; st.rerun()
        return
    st.subheader("Build")
    depth = st.radio("How deeply should each paper be read?", ["abstract", "fast", "standard", "full"], index=2, horizontal=True, key="depth",
                     format_func=lambda d: {"abstract": "Abstract only (fastest)", "fast": "Intro + conclusions", "standard": "Key sections (recommended)", "full": "Everything"}[d])
    st.caption("Rough size of the job (model requests per paper). Gemini reads big chunks: about 1 to 3 requests per paper. Groq and small local models use smaller chunks: abstract only about 1, key sections about 3 to 6. Free accounts have daily limits: if you hit one, the build pauses and keeps its progress.")
    js = builder.job_status(project); pending = [p for p in papers if p["state"] != "done"]
    if js["state"] == "running": pass
    elif st.button(("Resume build" if js["state"] in ("paused", "interrupted", "failed", "cancelled") or any(p["state"] == "done" for p in papers) else "Build corpus") + f" ({len(papers)} papers)", type="primary", key="build", disabled=not llm.available()):
        builder.start_build(project, depth, True, get_store()); S[f"wr_{project}"] = True; st.rerun()
    progress(project)
    quality_panel()
    res = pipeline.load(project)
    if res and res.get("norm_error"): st.warning(res["norm_error"])
    if res and res.get("review"):
        st.subheader("Clean up names")
        st.caption("The model may name the same thing differently in different papers, which creates false gaps. Confirm the pairs that are the same.")
        for i, (typ, a, b, r) in enumerate(res["review"][:8]):
            c1, c2, c3 = st.columns([3, 3, 1]); c1.write(f"**{typ}:** {a}"); c2.write(f"{b}  _(similarity {r})_")
            if c3.button("Same", key=f"alias_{i}"): builder.add_alias(project, typ, a, b); builder.assemble(project); rebuild_graph(); S.pop("gk", None); st.rerun()

def quality_panel():
    dg = jread(project, "diagnostics.json", {})
    if not dg: return
    kept = sum(d["facts_kept"] for d in dg.values()); rej = sum(sum(d["rejected"].values()) for d in dg.values()); rate = kept / (kept + rej) if kept + rej else 1.0
    with st.expander(f"Extraction quality: {rate:.0%} of the model's statements were backed by a real sentence from the paper", expanded=rate < .6):
        st.caption("Every statement the model makes must quote a sentence that really exists in the paper; otherwise it is thrown away. A low rate means the model paraphrases too much or the PDF text is messy.")
        if rate < .6: st.warning("Many statements were rejected. Try a larger model in Settings, or check the PDFs (scanned or two-column files can extract badly).")
        st.dataframe(pd.DataFrame([dict(paper=d["title"], read_from=d["source"], facts_kept=d["facts_kept"], rejected=sum(d["rejected"].values()), why="; ".join(f"{k} ({v})" for k, v in d["rejected"].items()), notes="; ".join(d["flags"])) for d in dg.values()]), hide_index=True)

def progress(project):
    @st.fragment(run_every=2 if builder.running(project) else None)
    def panel():
        js = builder.job_status(project); ps = builder.list_papers(project); total = max(js.get("total", len(ps)), 1)
        if js["state"] == "running": st.progress(min(js.get("done", 0) / total, 1.0), text=f"{js.get('done', 0)} of {total} papers · {js.get('message', '')}"); S[f"wr_{project}"] = True
        elif S.pop(f"wr_{project}", False): st.rerun()
        if js["state"] in ("paused", "failed", "interrupted"): st.warning(js.get("message", ""))
        if js["state"] == "cancelled": st.info("Stopped. Progress is saved.")
        if js["state"] == "running" and st.button("Stop", key="stop"): builder.cancel(project)
        if js["state"] == "done" and js.get("summary"):
            sm = js["summary"]; st.success(f"Done: {sm['done']} papers read, {sm['facts']} facts extracted" + (f", {sm['abstract_only']} from abstracts only" if sm["abstract_only"] else "") + (f", {sm['failed']} failed" if sm["failed"] else "") + ".")
            if sm["papers"] and sm["abstract_only"] / sm["papers"] > .5: st.warning("Most papers were read from abstracts only. Add the PDFs of the key papers (section B) for deeper, more reliable gaps.")
        if ps:
            st.dataframe(pd.DataFrame([dict(paper=p["title"], year=p["year"], status=p["state"], read_from=p["source"] or "", facts=p["facts_raw"], notes="; ".join(p["flags"] + ([p["error"]] if p["error"] else []))) for p in ps]), hide_index=True, height=260)
    panel()

# ───────── page: explore ─────────
def page_gaps():
    st.header("2 · Explore gaps")
    gaps, facts = get_ranked()
    if gaps is None: st.info("Build a corpus first (step 1)."); return
    with st.expander("Ranking preferences"):
        W = dict(evidence=st.slider("Evidence strength", 0., 1., core.DEFAULT_W["evidence"]), novelty=st.slider("Novelty (needs external check)", 0., 1., core.DEFAULT_W["novelty"]),
                 relevance=st.slider("Relevance to my topic", 0., 1., core.DEFAULT_W["relevance"]), feasibility=st.slider("Feasibility", 0., 1., core.DEFAULT_W["feasibility"]))
    ranked, titles = rank(gaps, facts, W); dec = triage.get_all(project); papers = {p["id"]: p for p in pipeline.load(project)["papers"]}
    k0, k1, k2, k3, k4 = st.columns(5); k0.metric("Candidate gaps", len(ranked)); k1.metric("Kept", sum(d["status"] == "keep" for d in dec.values())); k2.metric("Later", sum(d["status"] == "later" for d in dec.values())); k3.metric("Dismissed", sum(d["status"] == "dismiss" for d in dec.values())); k4.metric("Not reviewed", sum(g["key"] not in dec for g in ranked))
    with st.expander("Coverage map: which methods have been tried on which problems"): coverage(facts, ranked)
    f1, f2, f3 = st.columns([2, 2, 1]); sel = f1.multiselect("Gap types", list(TYPES), default=list(TYPES), format_func=TYPES.get); stat = f2.multiselect("Status", ["new", "keep", "later", "dismiss"], default=["new", "keep", "later"], format_func=lambda s: {"new": "Not reviewed", "keep": "Kept", "later": "Later", "dismiss": "Dismissed"}[s])
    top = f3.number_input("Show", 5, 100, 20, step=5)
    shown = [g for g in ranked if g["type"] in sel and dec.get(g["key"], {"status": "new"})["status"] in stat][:int(top)]
    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("Write plain-language summaries (LLM)", disabled=not llm.available(), key="explain"):
        ex = jread(project, "explanations.json", {}); bar = st.progress(0)
        for i, g in enumerate(shown[:10]):
            if g["key"] not in ex:
                try: ex[g["key"]] = core.explain(g, titles, True)
                except llm.LLMError as e: st.error(str(e)); break
            bar.progress((i + 1) / min(len(shown), 10))
        jwrite(project, "explanations.json", ex); st.rerun()
    if b2.button("Check shown gaps on OpenAlex", key="verify_all"):
        with st.spinner("Searching the literature..."): vc = jread(project, "verify_cache.json", {}); pipeline.verify_all(shown, project); S.pop("gk", None)
        st.rerun()
    st.caption(f"Showing {len(shown)} of {len(ranked)} candidates. A gap means 'not found in this corpus', never 'nobody has done it'.")
    for i, g in enumerate(shown, 1): card(i, g, titles, papers, dec.get(g["key"], {"status": "new", "note": ""}))

def card(i, g, titles, papers, d):
    cf, v, cm = g["confidence"], g["verification"], g["components"]; low = bool(cf and cf["conf"] < .5)
    badge = {"keep": "⭐ ", "later": "🕓 ", "dismiss": "✖ "}.get(d["status"], "")
    with st.expander(f"{badge}#{i}  {g['title']}", expanded=i <= 2 and d["status"] == "new"):
        st.caption(f"{TYPES[g['type']]} · score {g['score']:.2f}" + (f" · confidence {abstain.band(cf['conf'])}" if cf else "") + f" · already done elsewhere? **{v['status']}**")
        if low: st.warning("Low confidence: this gap is likely to disappear if more papers are added.")
        st.write(g["explanation"])
        for e in g["evidence"][:6]:
            p = papers.get(e["paper"], {}); t = p.get("title") or titles.get(e["paper"], e["paper"]); link = f"[{t}]({p['url']})" if p.get("url") else t
            st.markdown(f"- {link}, p.{e['page']}: “{e['passage']}”")
        st.caption("Why this might be wrong: " + WHY[g["type"]])
        for p in v.get("papers", [])[:3]: st.markdown(f"↳ possibly already done: [{p['title']}]({p['url']}) ({p['year']})")
        q = (v.get("query") or " ".join(g["entities"])).replace(" ", "+"); st.markdown(f"[Search Google Scholar](https://scholar.google.com/scholar?q={q}) · [Search OpenAlex](https://openalex.org/works?search={q})")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
        for col, (lab, s) in zip((c1, c2, c3), (("⭐ Keep", "keep"), ("🕓 Later", "later"), ("✖ Dismiss", "dismiss"))):
            if col.button(lab, key=f"{s}_{g['key']}"): triage.set_(project, g["key"], s); st.rerun()
        note = c4.text_input("Note", d["note"], key=f"note_{g['key']}", label_visibility="collapsed", placeholder="Your note")
        if note != d["note"]: triage.set_(project, g["key"], note=note)

def coverage(facts, ranked):
    import altair as alt
    pairs = {}
    for f in facts:
        if f["kind"] == "applied": pairs.setdefault((f["method"], f["problem"]), set()).add(f["paper"])
    if not pairs: st.info("No method-problem pairs extracted yet."); return
    from collections import Counter
    ms = [m for m, _ in Counter(m for m, _ in pairs).most_common(12)]; ps = [p for p, _ in Counter(p for _, p in pairs).most_common(10)]; gp = {tuple(g["entities"]) for g in ranked if g["type"] == "method_problem"}
    df = pd.DataFrame([dict(method=m, problem=p, papers=len(pairs.get((m, p), ())), gap=(m, p) in gp) for m in ms for p in ps]); df["label"] = df.apply(lambda r: "?" if r.gap else (str(r.papers) if r.papers else ""), axis=1)
    base = alt.Chart(df).encode(x=alt.X("problem:N", title=None, axis=alt.Axis(labelAngle=-30)), y=alt.Y("method:N", title=None))
    st.altair_chart(base.mark_rect().encode(color=alt.condition("datum.gap", alt.value("#ffe8a3"), alt.Color("papers:Q", scale=alt.Scale(scheme="blues"), legend=None))) + base.mark_text().encode(text="label:N"))
    st.caption("Numbers = papers applying that method to that problem. ? = a gap candidate.")

# ───────── page: ask ─────────
def page_ask():
    st.header("3 · Ask the literature"); res = pipeline.load(project); corpus = pipeline.load_corpus(project)
    if not res or not corpus: st.info("Build a corpus first (step 1)."); return
    if not llm.available(): st.warning("Asking questions needs a language model (Settings)."); return
    st.caption("Answers use only your papers and always cite a sentence from them. If the papers do not say, you get 'not enough evidence', not a guess.")
    q = st.text_input("Your question", key="ask_q", placeholder="e.g. Which methods were evaluated on FLORES-200?")
    if st.button("Ask", type="primary", key="ask") and q.strip():
        mt = (project, len(corpus))
        if S.get("ix_key") != mt: S.chunks = C.build_chunks(corpus); S.ix = embed.Index([c["text"] for c in S.chunks]); S.ix_key = mt
        with st.spinner("Reading the papers..."):
            try: a = qa.ask(q, corpus, S.chunks, S.ix, pipeline.apply_review(res["facts"], pipeline.load_review(project))); h = jread(project, "ask_history.json", []); h.insert(0, dict(q=q, **a)); jwrite(project, "ask_history.json", h[:30])
            except llm.LLMError as e: st.error(str(e))
    papers = {p["id"]: p for p in res["papers"]}
    for h in jread(project, "ask_history.json", [])[:10]:
        with st.container(border=True):
            st.markdown(f"**{h['q']}**"); (st.success if h["answered"] else st.info)(h["answer"])
            for e in h["evidence"]: p = papers.get(e["paper"], {}); st.markdown(f"- {p.get('title', e['paper'])}, p.{e['page']}: “{e['passage']}”")

# ───────── page: export ─────────
def page_export():
    st.header("4 · Export"); gaps, facts = get_ranked()
    if gaps is None: st.info("Build a corpus first (step 1)."); return
    ranked, titles = rank(gaps, facts, core.DEFAULT_W); dec = triage.get_all(project); papers = pipeline.load(project)["papers"]
    what = st.radio("Which gaps?", ["Kept gaps only", "All except dismissed"], horizontal=True); inc = [g for g in ranked if (dec.get(g["key"], {"status": "new"})["status"] == "keep" if what.startswith("Kept") else dec.get(g["key"], {"status": "new"})["status"] != "dismiss")]
    for g in inc: g["explanation"] = g["explanation"] + (f"\n\nMy note: {dec[g['key']]['note']}" if dec.get(g["key"], {}).get("note") else "")
    if not inc: st.info("Nothing to export yet. Mark gaps as Keep in step 2."); return
    st.write(f"{len(inc)} gaps selected."); c = st.columns(3)
    c[0].download_button("Report (Markdown)", export.to_markdown(inc, titles, project, meta["topic"], papers), f"{project}_gaps.md", key="dl_md"); c[1].download_button("Spreadsheet (CSV)", export.to_csv(inc, titles), f"{project}_gaps.csv", "text/csv", key="dl_csv")
    cited = {e["paper"] for g in inc for e in g["evidence"]}; c[2].download_button("References (BibTeX)", export.to_bibtex([p for p in papers if p["id"] in cited]), f"{project}_refs.bib", key="dl_bib")
    st.caption("Exports label every item as a candidate for your review and include the supporting sentences and page numbers.")

# ───────── page: settings ─────────
def page_settings():
    st.header("Settings"); cur = settings.load(); g = lambda k, d="": cur.get(k, os.getenv(k, d))
    st.subheader("Language model (reads the papers)")
    labels = {"gemini": "Google Gemini, free tier (recommended)", "openai": "Mistral / Cerebras / OpenRouter / other OpenAI-compatible", "groq": "Groq", "ollama": "Ollama (runs on your computer)"}
    prov = st.selectbox("Provider", list(labels), index=list(labels).index(llm.provider()), format_func=labels.get, key="prov"); v = {}
    if prov == "gemini":
        st.caption("Free key with a Google account only (no card): aistudio.google.com/apikey. The app sends a whole paper per request, so the free daily request limit goes a long way, and it picks the model with the biggest free allowance automatically.")
        v["GEMINI_API_KEY"] = st.text_input("Gemini API key", g("GEMINI_API_KEY"), type="password", key="k_gem"); v["GEMINI_MODEL"] = st.text_input("Model (leave empty for automatic)", g("GEMINI_MODEL"), key="m_gem")
    elif prov == "openai":
        st.caption("Any service with an OpenAI-compatible API. Mistral's free plan has by far the largest token allowance (console.mistral.ai, needs phone verification). Examples: https://api.mistral.ai/v1 , https://api.cerebras.ai/v1 , https://openrouter.ai/api/v1")
        v["OPENAI_BASE_URL"] = st.text_input("Base URL", g("OPENAI_BASE_URL", "https://api.mistral.ai/v1"), key="u_oai"); v["OPENAI_API_KEY"] = st.text_input("API key", g("OPENAI_API_KEY"), type="password", key="k_oai"); v["OPENAI_MODEL"] = st.text_input("Model", g("OPENAI_MODEL", "mistral-small-latest"), key="m_oai")
    elif prov == "groq":
        st.caption("Free key at console.groq.com. Free accounts have small daily token limits.")
        v["GROQ_API_KEY"] = st.text_input("Groq API key", g("GROQ_API_KEY"), type="password", key="k_groq"); v["GROQ_MODEL"] = st.text_input("Model", g("GROQ_MODEL", "llama-3.3-70b-versatile"), key="m_groq")
    else:
        st.caption("Needs the Ollama app running. Small models (1B to 3B) are too weak for reliable extraction: use 7B to 8B or larger if your computer allows.")
        v["OLLAMA_HOST"] = st.text_input("Ollama address", g("OLLAMA_HOST", "http://localhost:11434"), key="h_ol"); v["OLLAMA_MODEL"] = st.text_input("Model", g("OLLAMA_MODEL", "llama3.1:8b"), key="m_ol"); v["OLLAMA_NUM_CTX"] = str(st.number_input("Context window (tokens)", 2048, 131072, int(g("OLLAMA_NUM_CTX", "8192")), step=1024, key="ctx_ol"))
    temp = st.slider("Temperature", 0.0, 1.0, float(g("LLM_TEMPERATURE", "0") or 0), 0.1, key="temp", help="0 gives repeatable, faithful extraction. Higher values make the model less reliable at copying exact sentences.")
    if temp > .3: st.warning("Temperature above 0.3 makes extraction less reliable. 0 is recommended.")
    if st.button("Test this model", key="test_llm"):
        os.environ["LLM_PROVIDER"] = prov; os.environ["LLM_TEMPERATURE"] = str(temp); [os.environ.__setitem__(k, x) for k, x in v.items() if x != ""]; llm._AUTO.clear(); r = llm.check()
        (st.success if r["ok"] else st.error)(r["message"])
        if r.get("warning"): st.warning(r["warning"])
        if r.get("model") and not r["ok"]: st.info(f"Suggested model: {r['model']} (type it in the Model box and Save).")
        S.models = r.get("models", [])
    if S.get("models"): st.caption("Available: " + ", ".join(m for m in S.models if not any(b in m.lower() for b in llm.BAD))[:900])
    st.subheader("Papers"); mail = st.text_input("Your email (optional, gives faster OpenAlex access)", cur.get("OPENALEX_MAILTO", ""))
    st.subheader("Neo4j graph database (optional)"); st.caption("GapFinder works without it. Use it to explore the graph in Neo4j Browser.")
    use = st.checkbox("Use Neo4j", cur.get("GAPFINDER_USE_NEO4J") == "1"); c = st.columns(4); uri = c[0].text_input("URI", cur.get("NEO4J_URI", "bolt://localhost:7687")); usr = c[1].text_input("User", cur.get("NEO4J_USER", "neo4j")); pw = c[2].text_input("Password", cur.get("NEO4J_PASSWORD", ""), type="password"); db = c[3].text_input("Database", cur.get("NEO4J_DATABASE", "neo4j"))
    if st.button("Test Neo4j", key="test_neo"):
        s, err = connect(uri, usr, pw, db); (st.success("Neo4j connected") if s else st.error(err))
    st.subheader("Retrieval"); emb = st.text_input("Embedding model (optional)", cur.get("EMBED_MODEL", ""), help="e.g. sentence-transformers/all-MiniLM-L6-v2 (pip install sentence-transformers). Empty = built-in keyword retrieval.")
    if st.button("Save settings", type="primary", key="save_settings"):
        settings.save(dict(LLM_PROVIDER=prov, LLM_TEMPERATURE=str(temp), OPENALEX_MAILTO=mail, GAPFINDER_USE_NEO4J="1" if use else "0", NEO4J_URI=uri, NEO4J_USER=usr, NEO4J_PASSWORD=pw, NEO4J_DATABASE=db, EMBED_MODEL=emb, **v)); settings.apply(); llm._AUTO.clear(); connect.clear(); st.success("Saved.")
    st.caption(f"Your projects are stored in: {settings.home()}")
    with st.expander("For the thesis: research workbench"): st.write("Conditions V/G/H, blinded rating and withholding experiments live in a separate app:  `streamlit run research_app.py`")

{"1 · Build corpus": page_build, "2 · Explore gaps": page_gaps, "3 · Ask the literature": page_ask, "4 · Export": page_export, "Settings": page_settings}[nav]()
