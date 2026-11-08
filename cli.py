"""Command line for the whole workflow.
  python cli.py check                                   test the language model, Neo4j, OpenAlex
  python cli.py collect --project mt --phrase "..." --from 2019-01-01 --to 2023-12-31 --max 60
  python cli.py run --project mt [--llm] [--neo4j] [--verify]     extract + detect + rank (default: sample papers, offline)
  python cli.py experiment --projects a b c [--n 8] [--neo4j]     conditions V/G/H -> blinded pool for raters
  python cli.py withhold --projects a b c                         RQ2 withholding experiment + train confidence model
  python cli.py analyze --run <folder> [--judge]                  analyze expert ratings (+ LLM judge)
  python cli.py reset --project p --neo4j --yes                   delete ONE project's graph"""
import argparse, glob, json, sys, os
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
import core, llm, pipeline, abstain, experiment, analysis
HERE = Path(__file__).parent

def show(g, titles):
    c, v, cf = g["components"], g["verification"], g.get("confidence")
    print(f"\n#{g['rank']}  [{g['type']}]  score {g['score']:.2f}  {g['title']}\n    why: {g['explanation']}")
    print(f"    scores: evidence {c['evidence']:.2f} | novelty {c['novelty']:.2f} | relevance {c['relevance']:.2f} | feasibility {c['feasibility']:.2f} | external: {v['status']}" + (f" | confidence {cf['conf']:.2f} ({cf['basis']})" if cf else ""))
    for e in g["evidence"][:3]: print(f"    - {titles.get(e['paper'], e['paper'])[:50]} p.{e['page']}: \"{e['passage'][:110]}\"")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check"); sp.add_parser("models")
    c = sp.add_parser("collect"); c.add_argument("--project", required=True); c.add_argument("--phrase", required=True); c.add_argument("--cat", default="cs.CL"); c.add_argument("--from", dest="d1", required=True); c.add_argument("--to", dest="d2", required=True); c.add_argument("--max", type=int, default=60)
    r = sp.add_parser("run", aliases=["demo"]); r.add_argument("--project", default="demo"); r.add_argument("--pdfs", nargs="*"); r.add_argument("--topic"); r.add_argument("--llm", action="store_true"); r.add_argument("--neo4j", action="store_true"); r.add_argument("--verify", action="store_true")
    e = sp.add_parser("experiment"); e.add_argument("--projects", nargs="+", required=True); e.add_argument("--n", type=int, default=8); e.add_argument("--no-llm", action="store_true"); e.add_argument("--neo4j", action="store_true")
    w = sp.add_parser("withhold"); w.add_argument("--projects", nargs="+", required=True); w.add_argument("--fracs", nargs="+", type=float, default=[.2, .4])
    a = sp.add_parser("analyze"); a.add_argument("--run", required=True); a.add_argument("--judge", action="store_true")
    z = sp.add_parser("reset"); z.add_argument("--project", required=True); z.add_argument("--neo4j", action="store_true"); z.add_argument("--yes", action="store_true")
    o = ap.parse_args()

    if o.cmd == "models": print("\n".join(llm.list_models())); return
    if o.cmd == "collect":
        import collect_arxiv; collect_arxiv.collect(o.project, o.phrase, o.cat, o.d1, o.d2, o.max); return
    if o.cmd == "check":
        print("Language model:", llm.label(), "| configured:", llm.available())
        r_ = llm.check(); print("LLM test:", "OK" if r_["ok"] else "FAILED", "-", r_["message"], ("| " + r_["warning"]) if r_.get("warning") else "")
        try: s = pipeline.get_store(); print("Neo4j:", f"connected, projects = {s.projects()}" if s else "NEO4J_PASSWORD not set")
        except Exception as ex: print("Neo4j FAILED:", type(ex).__name__, str(ex)[:200])
        try: print("OpenAlex:", len(core.openalex_search("machine translation", 1)), "result OK")
        except Exception as ex: print("OpenAlex FAILED:", type(ex).__name__, str(ex)[:120])
        import embed; print("Embeddings:", embed.Index(["a b", "c d"]).kind, "(set EMBED_MODEL in .env for the thesis experiments)"); return
    store = pipeline.get_store() if getattr(o, "neo4j", False) else None
    if getattr(o, "neo4j", False) and not store: sys.exit("Set NEO4J_PASSWORD (see .env.example)")
    if o.cmd == "reset":
        if not store: sys.exit("use --neo4j")
        if not o.yes: sys.exit(f"this deletes every node of project '{o.project}'. Re-run with --yes.")
        store.reset(o.project); print("reset", o.project); return
    if o.cmd in ("run", "demo"):
        pdfs = o.pdfs or sorted(glob.glob(str(pipeline.pdir(o.project) / "pdfs" / "*.pdf"))) or sorted(glob.glob(str(HERE / "samples" / "*.pdf"))); mode = "llm" if o.llm else "fixture"
        if mode == "llm" and not llm.available(): sys.exit("GROQ_API_KEY not set")
        if o.topic: pipeline.project_meta(o.project, topic=o.topic)
        topic = pipeline.project_meta(o.project)["topic"]; res = pipeline.extract_all(pdfs, mode, o.llm); pipeline.save(o.project, res); pipeline.save_review(o.project, dict(rejected=[]))
        print(f"\nfacts: {len(res['facts'])} | rejected (no verbatim provenance / bad schema): {len(res['dropped'])} | merges: {res['merges']} | near-duplicates to review: {res['review']}")
        if store: store.reset(o.project); store.ingest(o.project, res["papers"], res["facts"]); print("Neo4j:", store.stats(o.project))
        gaps = pipeline.detect(res, o.project, store, o.llm)
        if o.verify: gaps = pipeline.verify_all(gaps, o.project)
        titles = {p["id"]: p["title"] for p in res["papers"]}; ranked = pipeline.finish(gaps, titles, topic, core.DEFAULT_W, o.llm); model = abstain.load_model(HERE / "abstention_model.json"); idx = abstain.Idx(res["facts"])
        for i, g in enumerate(ranked, 1): g["rank"] = i; g["confidence"] = abstain.confidence(g, res["facts"], model, idx); show(g, titles)
        print(f"\n{len(ranked)} potential gaps (G condition; engine: {'Neo4j Cypher' if store else 'in-memory mirror'}, extraction: {mode}, external check: {'on' if o.verify else 'off'})"); return
    if o.cmd == "experiment":
        use = not o.no_llm
        if use and not llm.available(): sys.exit("V and H need GROQ_API_KEY (or use --no-llm for condition G only)")
        d = experiment.run(o.projects, o.n, use, store, conds=("V", "G", "H") if use else ("G",)); print("\nNext: raters run  streamlit run annotate_app.py  and pick run", d.name); return
    if o.cmd == "withhold":
        fb = {p: pipeline.apply_review(pipeline.load(p)["facts"], pipeline.load_review(p)) for p in o.projects}
        if len(fb) < 2: print("WARNING: leave-one-topic-out needs 2+ projects; with one project the model cannot be evaluated.")
        rows = abstain.build_rows(fb, tuple(o.fracs)); ev = abstain.evaluate(rows); d = HERE / "experiments" / ("withholding_" + __import__("time").strftime("%Y%m%d-%H%M%S")); d.mkdir(parents=True, exist_ok=True)
        (d / "rows.json").write_text(json.dumps(rows), encoding="utf-8"); (d / "evaluation.json").write_text(json.dumps(ev, indent=1), encoding="utf-8"); print(json.dumps(ev, indent=1))
        m = abstain.train(rows)
        if m: abstain.save_model(HERE / "abstention_model.json", m); print("\nconfidence model trained on all rows -> abstention_model.json (the app now uses it)")
        print("results in", d); return
    if o.cmd == "analyze":
        d = HERE / "experiments" / o.run; res = analysis.analyze(d); print(json.dumps(res, indent=1))
        if o.judge: print("LLM judge vs experts:", json.dumps(analysis.judge(d), indent=1))
        return

if __name__ == "__main__": main()
