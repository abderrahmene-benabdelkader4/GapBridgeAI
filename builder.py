"""Resumable corpus builder. Every paper is processed independently and saved (meta, parsed text, raw LLM output, status), so a crash,
a closed browser or a daily rate limit never loses work: pressing Build/Resume again continues where it stopped."""
import os, json, re, time, hashlib, threading
from pathlib import Path
import core, llm, pipeline, sources

def _papers(project): d = pipeline.pdir(project) / "papers"; d.mkdir(exist_ok=True); return d
def slug(s): return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")[:80] or hashlib.sha1(str(s).encode()).hexdigest()[:10]
def _rj(f, default=None):
    for _ in range(5):                                             # the UI polls while the worker writes: tolerate a rare half-written read
        try: return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default
        except (ValueError, OSError): time.sleep(.02)
    return default
def _wj(f, obj):
    tmp = f.with_name(f.name + f".{threading.get_ident()}.tmp"); tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8"); os.replace(tmp, f)    # atomic: readers see the old or the new file, never half
def _set(project, pid, **kw): f = _papers(project) / pid / "status.json"; s = _rj(f, {}); s.update(kw, updated=time.time()); _wj(f, s)

def add_papers(project, items):
    """items: dicts from sources.search (id, title, year, doi, url, pdf_urls, abstract...). Existing papers keep their status."""
    out = []
    for it in items:
        pid = slug(it.get("id") or it["title"]); d = _papers(project) / pid; d.mkdir(exist_ok=True)
        if not (d / "meta.json").exists(): _wj(d / "meta.json", dict(it, id=pid, origin="openalex")); _wj(d / "status.json", dict(state="queued", updated=time.time()))
        out.append(pid)
    return out

def add_upload(project, filename, data):
    pid = slug(Path(filename).stem); d = _papers(project) / pid; d.mkdir(exist_ok=True); (d / "source.pdf").write_bytes(data)
    if not (d / "meta.json").exists(): _wj(d / "meta.json", dict(id=pid, title=Path(filename).stem.replace("_", " "), origin="upload", pdf_urls=[], abstract="")); _wj(d / "status.json", dict(state="queued", updated=time.time()))
    return pid

def list_papers(project):
    rows = []
    for d in sorted(_papers(project).iterdir()):
        if not (d / "meta.json").exists(): continue
        m, s = _rj(d / "meta.json"), _rj(d / "status.json", {}); rows.append(dict(id=d.name, title=m.get("title"), year=m.get("year"), origin=m.get("origin"), state=s.get("state", "queued"), error=s.get("error", ""), flags=s.get("flags", []), facts_raw=s.get("facts_raw"), source=s.get("source", "")))
    return rows

def _cfg(depth, use_llm): return dict(depth=depth, llm=use_llm, model=llm.label() if use_llm else "")

def process_one(project, pid, depth="standard", use_llm=True, log=lambda *_: None):
    """depth: 'abstract' | 'fast' | 'standard' | 'full'. Raises llm.FATAL-type errors (stop the job); other failures mark just this paper."""
    d = _papers(project) / pid; meta = _rj(d / "meta.json"); st = _rj(d / "status.json", {}); cfg = _cfg(depth, use_llm)
    if st.get("state") == "done" and st.get("config") == cfg and (d / "raw.json").exists() and (d / "parsed.json").exists(): return
    try:
        parsed, flags = None, []; pdf = d / "source.pdf"
        if depth != "abstract":
            if not pdf.exists() and meta.get("pdf_urls"):
                _set(project, pid, state="downloading", config=cfg); ok, why = sources.download_pdf(meta["pdf_urls"], pdf)
                if not ok: flags.append(f"full text unavailable ({why})")
            if pdf.exists():
                _set(project, pid, state="parsing", config=cfg)
                try:
                    parsed = core.parse_pdf(str(pdf), depth=depth, chunk_chars=llm.chunk_chars() if use_llm else 3500)
                    if not parsed["chunks"]: flags += parsed["quality"]["flags"]; parsed = None
                except Exception as e: flags.append(f"PDF could not be read ({type(e).__name__}): using the abstract instead"); parsed = None
        if parsed is None:
            if len(meta.get("abstract", "")) > 80: parsed = core.parse_text(pid, meta["title"], meta["abstract"])
            else: raise ValueError("no usable full text and no abstract")
        parsed["id"] = pid; parsed["title"] = meta.get("title") or parsed["title"]; flags += parsed["quality"]["flags"]
        _set(project, pid, state="extracting", config=cfg)
        raw, stats = core.extract_paper(parsed, "llm" if use_llm else "fixture", cache_dir=pipeline.pdir(project) / "cache", log=log)
        _wj(d / "parsed.json", parsed); _wj(d / "raw.json", raw)
        _set(project, pid, state="done", config=cfg, error="", flags=list(dict.fromkeys(flags + stats["errors"])), source=parsed["source"], facts_raw=sum(len(v) for v in raw.values()), calls=stats["calls"])
    except core.FATAL as e: _set(project, pid, state="queued", error=str(e)); raise
    except Exception as e: _set(project, pid, state="failed", error=f"{type(e).__name__}: {str(e)[:200]}")

def _aliases(project):
    base = core.load_aliases(); extra = _rj(pipeline.pdir(project) / "aliases.json", {})
    for t, ents in extra.items():
        for canon, als in ents.items(): base.setdefault(t, {}).setdefault(canon, []); base[t][canon] = list(dict.fromkeys(base[t][canon] + als))
    return base

def add_alias(project, typ, canon, alias):
    f = pipeline.pdir(project) / "aliases.json"; a = _rj(f, {}); a.setdefault(typ, {}).setdefault(canon, [])
    if alias not in a[typ][canon]: a[typ][canon].append(alias)
    _wj(f, a)

def assemble(project, use_llm=False):
    """All finished papers -> validated, normalized facts + corpus + diagnostics (cheap; run again after changing aliases or review)."""
    papers, facts, dropped, parsed_all, diag = [], [], [], [], {}
    for d in sorted(_papers(project).iterdir()):
        st, meta = _rj(d / "status.json", {}), _rj(d / "meta.json")
        if st.get("state") != "done" or not (d / "parsed.json").exists(): continue
        parsed, raw = _rj(d / "parsed.json"), _rj(d / "raw.json"); f, dr = core.validate(raw, parsed); reasons = {}
        for _, why, _ in dr: reasons[why] = reasons.get(why, 0) + 1
        facts += f; dropped += [(d.name,) + x for x in dr]; parsed_all.append(parsed)
        papers.append(dict(id=d.name, title=parsed["title"], year=meta.get("year"), doi=meta.get("doi"), url=meta.get("url") or meta.get("doi"), source=parsed["source"], quality=parsed["quality"]))
        diag[d.name] = dict(title=parsed["title"], source=parsed["source"], facts_kept=len(f), rejected=reasons, flags=st.get("flags", []))
    norm_error = ""
    try: facts, review, merges = core.normalize(facts, _aliases(project), core.llm_confirm if use_llm and llm.available() else None)
    except Exception as e:                                          # keep the corpus, leave names unmerged, and say so
        import traceback; (pipeline.pdir(project) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        review, merges, norm_error = [], [], f"Name cleanup failed ({type(e).__name__}: {e or 'no message'}); names were left unmerged. Details: error.log in the project folder."
    for f in facts: f["id"] = pipeline.fact_id(f)
    res = dict(papers=papers, facts=facts, dropped=dropped, review=review, merges=merges, stats={}, parsed=parsed_all, norm_error=norm_error); pipeline.save(project, res); (pipeline.pdir(project) / "diagnostics.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
    return res

# ───────── background job (state in job.json, so a page refresh or a restarted browser does not lose track) ─────────
_THREADS, _CANCEL = {}, {}
def _jf(project): return pipeline.pdir(project) / "job.json"
def _jset(project, **kw): s = _rj(_jf(project), {}); s.update(kw, updated=time.time()); _wj(_jf(project), s)
def job_status(project):
    s = _rj(_jf(project), dict(state="idle", total=0, done=0))
    if s.get("state") == "running" and not (project in _THREADS and _THREADS[project].is_alive()): s["state"] = "interrupted"; s["message"] = "The build was interrupted (app closed). Press Resume to continue."
    return s
def running(project): return job_status(project)["state"] == "running"

def _run(project, depth, use_llm, store):
    try:
        todo = [r for r in list_papers(project) if r["state"] != "done" or _rj(_papers(project) / r["id"] / "status.json", {}).get("config") != _cfg(depth, use_llm)]
        _jset(project, state="running", total=len(list_papers(project)), done=len(list_papers(project)) - len(todo), message="Starting...", started=time.time(), summary=None, cancel=False)
        for r in todo:
            if _CANCEL.get(project) and _CANCEL[project].is_set(): _jset(project, state="cancelled", message="Stopped. Progress is saved."); return
            _jset(project, current=r["title"], message=f"Processing: {r['title'][:70]}"); process_one(project, r["id"], depth, use_llm)
            _jset(project, done=_rj(_jf(project), {}).get("done", 0) + 1)
        _jset(project, message="Building the knowledge graph..."); res = assemble(project, use_llm)
        if store: store.reset(project); store.ingest(project, res["papers"], pipeline.apply_review(res["facts"], pipeline.load_review(project)))
        ps = list_papers(project); _jset(project, state="done", message="Finished.", summary=dict(papers=len(ps), done=sum(p["state"] == "done" for p in ps), failed=sum(p["state"] == "failed" for p in ps), facts=len(res["facts"]), abstract_only=sum(p["source"] == "abstract" for p in ps)))
    except core.FATAL as e: _jset(project, state="paused", message=str(e))
    except Exception as e:
        import traceback; tb = traceback.extract_tb(e.__traceback__)[-1]; (pipeline.pdir(project) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        _jset(project, state="failed", message=f"{type(e).__name__}: {str(e)[:250] or '(no message)'} (at {Path(tb.filename).name}:{tb.lineno} in {tb.name}). Your papers are saved: fix the problem and press Resume. Details: error.log in the project folder.")

def start_build(project, depth="standard", use_llm=True, store=None):
    if running(project): return False
    _CANCEL[project] = threading.Event(); t = threading.Thread(target=_run, args=(project, depth, use_llm, store), daemon=True); _THREADS[project] = t; _jset(project, state="running", message="Starting...", done=0, total=len(list_papers(project))); t.start(); return True
def cancel(project):
    if project in _CANCEL: _CANCEL[project].set()
def wait(project, timeout=120):
    t = _THREADS.get(project)
    if t: t.join(timeout)
    return job_status(project)

def load_demo(project, samples_dir):
    """Instant demo with the bundled SYNTHETIC papers and ideal extractions: works with no Groq key and no internet."""
    for f in sorted(Path(samples_dir).glob("*.pdf")): add_upload(project, f.name, f.read_bytes())
    for r in list_papers(project): process_one(project, r["id"], "standard", False)
    return assemble(project, False)
