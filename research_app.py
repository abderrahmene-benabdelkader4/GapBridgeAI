import os, glob, tempfile
from pathlib import Path
import pandas as pd, streamlit as st
from dotenv import load_dotenv; load_dotenv()
import settings; settings.apply()
import core, llm, pipeline, graph, conditions as C, embed, abstain, export, experiment, analysis

st.set_page_config(page_title="GapFinder", layout="wide")
HERE = Path(__file__).parent; MODEL_PATH = HERE / "abstention_model.json"; S = st.session_state
SAMPLES = sorted(glob.glob(str(HERE / "samples" / "*.pdf")))
LABEL = dict(method_problem="Method x problem", context_problem="Context x problem", conflict="Conflicting results", future_work="Unaddressed future work", llm_gap="LLM-proposed gap")

# ───── sidebar ─────
with st.sidebar:
    st.header("Connections")
    uri = st.text_input("Neo4j URI", os.getenv("NEO4J_URI", "bolt://localhost:7687")); user = st.text_input("Neo4j user", os.getenv("NEO4J_USER", "neo4j"))
    pw = st.text_input("Neo4j password", os.getenv("NEO4J_PASSWORD", ""), type="password"); dbn = st.text_input("Database", os.getenv("NEO4J_DATABASE", "neo4j"))
    st.caption("Language model: " + llm.label() + "  (change it in the main app, Settings)")

    @st.cache_resource(show_spinner=False)
    def connect(uri, user, pw, dbn):
        if not pw: return None, "no password entered"
        try: s = graph.Store(uri, user, pw, dbn or None); s.check(); return s, None
        except Exception as e: return None, f"{type(e).__name__}: {str(e)[:150]}"
    store, err = connect(uri, user, pw, dbn)
    st.success("Neo4j connected") if store else st.warning(f"Neo4j not connected ({err}). In-memory mirror is used for condition G.")
    st.success("Language model ready") if llm.available() else st.info("No language model: V and H are unavailable; fixtures and templates are used")

    st.header("Project")
    local = [p.name for p in settings.projects_dir().iterdir() if p.is_dir()]; known = sorted(set(local + (store.projects() if store else [])))
    project = st.text_input("Project name (one per research topic)", known[0] if known else "demo"); meta = pipeline.project_meta(project)
    topic = st.text_input("Topic", meta["topic"]); pipeline.project_meta(project, topic=topic)
    with st.expander("Reset this project"):
        ok = st.text_input("Type the project name to confirm") == project and bool(project)
        if st.button("Delete this project's graph", disabled=not (ok and store)): store.reset(project); S.pop("gaps", None); st.success("Graph deleted (other projects untouched).")
    st.header("Ranking"); W = dict(evidence=st.slider("Evidence strength", 0., 1., core.DEFAULT_W["evidence"]), novelty=st.slider("Novelty (external check)", 0., 1., core.DEFAULT_W["novelty"]),
                                   relevance=st.slider("Relevance", 0., 1., core.DEFAULT_W["relevance"]), feasibility=st.slider("Feasibility", 0., 1., core.DEFAULT_W["feasibility"]))
    thr = st.slider("Abstain below confidence", 0.0, 1.0, 0.5, help="Structural gaps with confidence under this value are marked 'insufficient evidence'.")

if S.get("proj") != project:
    for k in ("res", "gaps", "chunks", "index"): S.pop(k, None)
    S.proj = project
st.title("GapFinder"); st.caption("PDFs -> verified facts -> Neo4j graph -> gap candidates (graph / vector / hybrid) -> external check -> ranked, explainable, exportable. Sample papers are SYNTHETIC.")
t1, t2, t3, t4 = st.tabs(["1 Papers and extraction", "2 Graph", "3 Gaps", "4 Experiments"])

def subj(f):
    k = f["kind"]
    return {"applied": lambda: f"{f['method']} -> {f['problem']}", "evaluated": lambda: f"{f['method']} on {f['dataset']}", "result": lambda: f"{f['method']} | {f['dataset']} | {f['metric']} = {f['value']:g} ({f.get('context') or '-'})",
            "limitation": lambda: f["text"], "future": lambda: f["text"]}.get(k, lambda: f.get("name", ""))()
def facts_now(): return pipeline.apply_review(S.res["facts"], pipeline.load_review(project))

with t1:
    up = st.file_uploader("Upload PDFs for this project", type="pdf", accept_multiple_files=True)
    use_samples = st.checkbox(f"Use the {len(SAMPLES)} synthetic sample papers (works offline)", value=not up)
    mode = st.radio("Extraction", ["Offline fixtures (sample papers only)", "Language model"], index=1 if llm.available() else 0, horizontal=True)
    if st.button("Run extraction", type="primary", key="extract"):
        pdfs = []
        for f in up or []:
            p = pipeline.pdir(project) / "pdfs"; p.mkdir(exist_ok=True); (p / f.name).write_bytes(f.getbuffer()); pdfs.append(str(p / f.name))
        if use_samples: pdfs += SAMPLES
        if not pdfs: st.error("Add PDFs or tick the sample papers.")
        elif mode.startswith("Language") and not llm.available(): st.error("Configure a language model in the main app (Settings).")
        else:
            with st.status("Extracting...", expanded=True) as s:
                try:
                    S.res = pipeline.extract_all(pdfs, "llm" if mode.startswith("Language") else "fixture", mode.startswith("Language"), log=s.write); pipeline.save(project, S.res); pipeline.save_review(project, dict(rejected=[]))
                    if store: store.reset(project); store.ingest(project, S.res["papers"], S.res["facts"]); s.write("Ingested into Neo4j.")
                    s.update(label="Extraction finished", state="complete"); S.pop("gaps", None)
                except Exception as e: s.update(label="Extraction failed", state="error"); st.exception(e)
    if "res" not in S and (saved := pipeline.load(project)): S.res = dict(papers=saved["papers"], facts=saved["facts"], dropped=[], review=[], merges=[], stats={})
    if "res" in S:
        r = S.res; c = st.columns(4); c[0].metric("Papers", len(r["papers"])); c[1].metric("Facts kept", len(r["facts"])); c[2].metric("Rejected (no verbatim provenance)", len(r.get("dropped", []))); c[3].metric("Entity merges", len(r.get("merges", [])))
        st.subheader("Review extracted facts"); st.caption("Tick 'Reject' for wrong facts, then save. Rejected facts are excluded from every condition and from the graph.")
        rej = set(pipeline.load_review(project)["rejected"])
        df = pd.DataFrame([dict(reject=f.get("id") in rej, id=f.get("id"), paper=f["paper"], kind=f["kind"], fact=subj(f), page=f["page"], passage=f["passage"]) for f in r["facts"]])
        ed = st.data_editor(df, hide_index=True, height=300, key=f"review_{project}", disabled=[x for x in df.columns if x != "reject"], column_config={"reject": st.column_config.CheckboxColumn("Reject")})
        a, b = st.columns(2)
        if a.button("Save review", key="save_review"): pipeline.save_review(project, dict(rejected=ed[ed.reject].id.tolist())); S.pop("gaps", None); st.success(f"{int(ed.reject.sum())} facts rejected.")
        if b.button("Save review and rebuild graph", key="rebuild", disabled=store is None):
            pipeline.save_review(project, dict(rejected=ed[ed.reject].id.tolist())); store.reset(project); store.ingest(project, r["papers"], facts_now()); S.pop("gaps", None); st.success("Graph rebuilt from reviewed facts.")
        if r.get("dropped"):
            with st.expander(f"{len(r['dropped'])} facts rejected automatically"): st.dataframe(pd.DataFrame([dict(paper=d[0], kind=d[1], reason=d[2], item=str(d[3])[:160]) for d in r["dropped"]]), hide_index=True)
        if r.get("review") or r.get("merges"):
            with st.expander("Normalization: merges and near-duplicates to review (edit aliases.json)"): st.write("merged:", r["merges"]); st.write("review:", r["review"])

with t2:
    if store and "res" in S:
        try: stats = store.stats(project); a, b = st.columns(2); a.subheader("Nodes"); a.dataframe(pd.Series(stats["nodes"], name="count")); b.subheader("Relationships"); b.dataframe(pd.Series(stats["relationships"], name="count"))
        except Exception as e: st.error(str(e))
        st.info("Explore in Neo4j Browser:  MATCH (n {project:'%s'})-[r]->(m) RETURN n,r,m LIMIT 100" % project)
    else: st.info("Connect Neo4j and run an extraction to see the graph.")
    with st.expander("The Cypher behind each detector"):
        for n, q in [("A1 Method x problem", graph.A1), ("A2 Context x problem", graph.A2), ("B1 Conflicting results", graph.B1), ("B2 Future-work statements", graph.B2_ITEMS)]: st.markdown(f"**{n}**"); st.code(q, language="cypher")

with t3:
    if "res" not in S: st.info("Run an extraction first.")
    else:
        cond = st.radio("Condition", ["G: graph (structural detectors)", "V: vector RAG", "H: hybrid (chunks + graph facts)"], horizontal=True)[0]
        c1, c2, c3 = st.columns([1, 1, 3]); verify = c1.checkbox("External check (OpenAlex)"); n_c = c2.number_input("Max candidates", 3, 30, 10)
        if c3.button("Generate gaps", type="primary", key="detect"):
            with st.spinner("Working..."):
                try:
                    titles = {p["id"]: p["title"] for p in S.res["papers"]}
                    if cond == "G": gaps = pipeline.detect(dict(facts=facts_now()), project, store, llm.available())
                    else:
                        corpus = pipeline.load_corpus(project)
                        if not corpus: raise RuntimeError("No parsed corpus for this project: run extraction again.")
                        if "chunks" not in S: S.chunks = C.build_chunks(corpus); S.index = embed.Index([x["text"] for x in S.chunks])
                        gaps = C.run_cond(cond, corpus=corpus, chunks=S.chunks, index=S.index, facts=facts_now(), topic=topic, n=int(n_c), use_llm=llm.available(), titles=titles)["cands"]
                    if verify: gaps = pipeline.verify_all(gaps, project)
                    S.setdefault("gaps", {})[cond] = gaps; S.titles = titles
                except Exception as e: st.error(f"{type(e).__name__}: {e}")
        if cond in S.get("gaps", {}):
            ranked = pipeline.finish(S.gaps[cond], S.titles, topic, W, llm.available()); facts = facts_now(); model = abstain.load_model(MODEL_PATH); idx = abstain.Idx(facts)
            for g in ranked: g["confidence"] = abstain.confidence(g, facts, model, idx)
            hide = st.checkbox("Hide gaps below the confidence threshold"); shown = [g for g in ranked if not (hide and g["confidence"] and g["confidence"]["conf"] < thr)]
            st.caption(f"{len(shown)} of {len(ranked)} candidates shown | engine for G: {'Neo4j Cypher' if store else 'in-memory mirror'} | confidence: {'trained model' if model else 'heuristic (run the withholding experiment to train)'}")
            x1, x2 = st.columns(2); x1.download_button("Export CSV", export.to_csv(shown, S.titles), f"gaps_{project}_{cond}.csv", "text/csv"); x2.download_button("Export Markdown", export.to_markdown(shown, S.titles, project, topic), f"gaps_{project}_{cond}.md")
            for i, g in enumerate(shown, 1):
                v, cm, cf = g["verification"], g["components"], g["confidence"]; low = bool(cf and cf["conf"] < thr)
                with st.expander(f"#{i} {'[insufficient evidence] ' if low else ''}{g['title']}   |   score {g['score']:.2f}", expanded=i <= 2 and not low):
                    st.caption(f"{LABEL[g['type']]} | external check: **{v['status']}**" + (f" | confidence **{cf['conf']:.2f}** ({abstain.band(cf['conf'])}; {cf['basis']})" if cf else " | confidence: not scored for this gap type"))
                    if low: st.warning("Insufficient evidence: this gap is likely to disappear if more papers are added.")
                    st.write(g["explanation"]); cols = st.columns(4)
                    for col, k in zip(cols, ["evidence", "novelty", "relevance", "feasibility"]): col.metric(k.capitalize(), f"{cm[k]:.2f}")
                    st.dataframe(pd.DataFrame([dict(paper=S.titles.get(e["paper"], e["paper"]), page=e["page"], passage=e["passage"], why=e["note"]) for e in g["evidence"]]), hide_index=True)
                    if g["type"] == "future_work": st.caption("Judge: " + g["meta"].get("judge", ""))
                    for p in v.get("papers", []): st.markdown(f"external: [{p['title']}]({p['url']}) ({p['year']})")
                    st.caption(v.get("note", ""))

with t4:
    ready = [p for p in known if pipeline.load_corpus(p)]; st.subheader("A. Compare conditions V, G, H (RQ1)")
    sel = st.multiselect("Projects (one per topic)", ready, default=ready[:5]); n_e = st.slider("Candidates per condition and topic", 3, 12, 8); use_l = st.checkbox("Use the LLM (required for V and H)", value=llm.available())
    if st.button("Run experiment", key="run_exp", disabled=not sel):
        with st.status("Running conditions...", expanded=True) as s:
            try: d = experiment.run(sel, n_e, use_l, store, log=s.write); s.update(label=f"Done: {d.name}", state="complete"); st.success(f"Blinded pool written to {d}. Raters: run  streamlit run annotate_app.py")
            except Exception as e: s.update(label="Failed", state="error"); st.exception(e)
    st.subheader("B. Corpus withholding and abstention (RQ2)")
    wsel = st.multiselect("Projects for the withholding experiment (use 2 or more)", ready, default=ready, key="wsel")
    if st.button("Run withholding experiment", key="run_withhold", disabled=len(wsel) < 1):
        fb = {p: pipeline.apply_review(pipeline.load(p)["facts"], pipeline.load_review(p)) for p in wsel}; rows = abstain.build_rows(fb); ev = abstain.evaluate(rows)
        st.json({k: v for k, v in ev.items() if not k.startswith("risk")}); st.dataframe(pd.DataFrame(ev["risk_coverage_model"]).merge(pd.DataFrame(ev["risk_coverage_evidence_count"]), on="coverage", suffixes=("_model", "_evidence_count")).merge(pd.DataFrame(ev["risk_coverage_always_answer"]).rename(columns={"risk": "risk_always_answer"}), on="coverage"), hide_index=True)
        m = abstain.train(rows)
        if m: abstain.save_model(MODEL_PATH, m); st.success("Confidence model trained on all rows and saved: gap cards now use it.")
    st.subheader("C. Analyze expert ratings")
    runs = sorted([p.name for p in (HERE / "experiments").glob("*") if (p / "ratings.sqlite").exists()], reverse=True) if (HERE / "experiments").exists() else []
    if runs:
        rsel = st.selectbox("Run", runs)
        if st.button("Analyze ratings", key="analyze"):
            res = analysis.analyze(HERE / "experiments" / rsel); st.write("Raters:", res["raters"], "| items rated by all:", res["items_rated_by_all"])
            st.dataframe(pd.DataFrame({c: {k: (f"{v['est']} {v['ci95']}" if isinstance(v, dict) else v) for k, v in m.items()} for c, m in res["conditions"].items()}))
            st.write("Differences:", res["differences"]); st.write("Kappa:", res["kappa"]); st.caption(res["note"])
    else: st.info("No rated runs yet. Raters use  streamlit run annotate_app.py")
