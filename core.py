"""GapFinder mini-prototype core: PDF parsing -> LLM extraction (+provenance check) -> normalization -> gap detectors (in-memory mirror)
-> external verification (OpenAlex) -> ranking -> explanation."""
import os, re, json, hashlib, difflib, itertools
from pathlib import Path
from collections import Counter, defaultdict
import pymupdf, requests
import llm

HERE = Path(__file__).parent
norm_ws = lambda s: re.sub(r"-\s+", "-", re.sub(r"\s+", " ", str(s))).strip()     # also repairs line-end hyphenation
alnum = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
words = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

# ───────── STEP 3a · PDF PARSING (PyMuPDF with column ordering, header/footer removal, hyphen repair, title by font size, quality flags) ─────────
NAMES = (r"abstract|introduction|related works?|background|preliminaries|problem (?:statement|formulation|definition)|methods?|methodology|approach|proposed (?:method|approach)|datasets?|"
         r"experiments?|experimental (?:setup|setting|results|evaluation)|implementation details|evaluation|results?|results and discussion|discussion|analysis|case stud(?:y|ies)|limitations?|"
         r"conclusions?|conclusions? and future work|limitations? and future work|future work|future directions|ethic(?:s|al)(?: (?:statement|considerations))?|broader impacts?|references|acknowledg\w*|appendix\w*|appendices")
HEAD = re.compile(rf"^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+|[A-Z])[.)]?\s+)?({NAMES})\s*[.:]?\s*$", re.I)
SKIP = re.compile(r"^(?:(?:\d+|[IVX]+|[A-Z])[.)]?\s+)?(references|acknowledg\w*|appendix\w*|appendices)\b", re.I)
DROP_STANDARD = re.compile(r"related|background|ethic|broader", re.I); KEEP_FAST = re.compile(r"abstract|intro|conclu|future|limitation|discussion|front", re.I)

def _blocks(pg):
    """Line SEGMENTS (a line is split where a big horizontal gap separates columns/cells). More robust than PyMuPDF blocks, which can merge across columns."""
    W, H = pg.rect.width, pg.rect.height; out = []
    for b in pg.get_text("dict")["blocks"]:
        if b.get("type") != 0: continue
        for l in b["lines"]:
            if abs(l["dir"][0]) <= .5: continue                                                   # rotated margin stamps (arXiv watermark)
            spans = [sp for sp in l["spans"] if sp["text"].strip()]
            if not spans: continue
            groups = [[spans[0]]]
            for sp in spans[1:]:
                if sp["bbox"][0] - groups[-1][-1]["bbox"][2] > 22: groups.append([sp])
                else: groups[-1].append(sp)
            for g in groups:
                txt = "".join(sp["text"] for sp in g).strip()
                if txt: out.append(dict(x0=g[0]["bbox"][0], x1=g[-1]["bbox"][2], y0=min(sp["bbox"][1] for sp in g), y1=max(sp["bbox"][3] for sp in g), text=txt, size=max(sp["size"] for sp in g)))
    return out, W, H

def _order(blocks, W):
    mid = W / 2; full = lambda b: b["x0"] < mid * .85 and b["x1"] > mid * 1.15
    left = [b for b in blocks if not full(b) and (b["x0"] + b["x1"]) / 2 < mid]; right = [b for b in blocks if not full(b) and (b["x0"] + b["x1"]) / 2 >= mid]
    if len(left) < 6 or len(right) < 6: return sorted(blocks, key=lambda b: (round(b["y0"] / 3), b["x0"]))
    out, prev = [], -1e9
    for f in sorted([b for b in blocks if full(b)], key=lambda b: b["y0"]) + [None]:
        hi = f["y0"] if f else 1e9; out += sorted([b for b in left if prev <= b["y0"] < hi], key=lambda b: b["y0"]) + sorted([b for b in right if prev <= b["y0"] < hi], key=lambda b: b["y0"])
        if f: out.append(f); prev = f["y1"]
    return out

def _vocab(text): return set(re.findall(r"[a-z]{3,}", re.sub(r"[a-z]+-\n+[a-z]+", " ", text.lower())))
def _repair_hyphens(text, vocab=None):
    """Join line-break hyphens (trans-/lation, lan-/guages) but keep real compounds (low-/resource): a hyphen is kept only if the
    joined word is unknown AND both halves are independent words elsewhere in the document."""
    vocab = vocab if vocab is not None else _vocab(text)
    def fix(m):
        a, b = m.group(1), m.group(2)
        if (a + b).lower() in vocab: return a + b
        return a + "-" + b if a.lower() in vocab and b.lower() in vocab else a + b
    return re.sub(r"([A-Za-z]+)-\n+([a-z]+)", fix, text)

def parse_pdf(path, chunk_chars=3500, max_chars=60000, depth="standard"):
    """depth: 'fast' (abstract, intro, conclusions, limitations), 'standard' (everything except related work/background/ethics/references), 'full' (everything except references/appendix)."""
    doc = pymupdf.open(path); per_page, flags = [], []
    for pg in doc: bl, W, H = _blocks(pg); per_page.append((bl, W, H))
    n = len(per_page); norm = lambda t: re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.lower()))[:80]; freq = {}
    for bl, W, H in per_page:
        for b in bl:
            if b["y1"] < H * .08 or b["y0"] > H * .92: freq[norm(b["text"])] = freq.get(norm(b["text"]), 0) + 1
    texts = []
    for bl, W, H in per_page:
        keep = [b for b in bl if not ((b["y1"] < H * .08 or b["y0"] > H * .92) and (n >= 3 and freq.get(norm(b["text"]), 0) >= max(2, .4 * n) or re.fullmatch(r"[\divxlc\s\-–]{1,6}", b["text"].lower())))]
        texts.append("\n".join(b["text"] for b in _order(keep, W)))
    raw = "\n".join(texts); voc = _vocab(raw); texts = [_repair_hyphens(t, voc) for t in texts]; fixed = "\n".join(texts)
    chars = len(fixed.replace("\n", "").replace(" ", ""))
    if chars < 300 * max(n, 1): flags.append("little or no text: the PDF may be scanned (run OCR, e.g. ocrmypdf, then re-add it)")
    # title: largest-font lines in the top of page 1, else metadata, else first line
    title = ""
    if per_page:
        bl, W, H = per_page[0]; top = [b for b in bl if b["y0"] < H * .45 and b["size"] > 0]
        if top:
            mx = max(b["size"] for b in top); title = " ".join(" ".join(b["text"].split()) for b in sorted(top, key=lambda b: (round(b["y0"] / 3), b["x0"])) if b["size"] >= mx * .92)[:250]
    meta = (doc.metadata or {}).get("title", "").strip()
    if len(title) < 8 or len(title) > 250: title = meta if len(meta) > 5 else next((l.strip() for l in texts[0].splitlines() if len(l.strip()) > 8 and not l.upper().startswith("SYNTHETIC")), Path(path).stem) if texts else Path(path).stem
    elif len(meta) > 5 and abs(len(meta) - len(title)) < 40 and not title.upper().startswith("SYNTHETIC"): title = meta
    secs, cur = [], dict(name="Front matter", page=1, lines=[])
    for i, t in enumerate(texts, 1):
        for line in t.splitlines():
            if HEAD.match(line): secs.append(cur); cur = dict(name=line.strip(), page=i, lines=[])
            elif line.strip(): cur["lines"].append(line.strip())
    secs.append(cur); found = [x["name"] for x in secs if x["name"] != "Front matter" and x["lines"] or x["name"] != "Front matter"]
    chunks, buf, size, page0, total = [], [], 0, 1, 0
    def flush():
        nonlocal buf, size
        if buf: chunks.append(dict(page=page0, sections=[nm for nm, _ in buf], text="\n".join(f"[{nm}]\n{t}" for nm, t in buf))); buf, size = [], 0
    for sc in secs:
        t = " ".join(sc["lines"]); nm = sc["name"]
        if sc["name"] == "Front matter": t = t[:1800]
        if SKIP.match(nm) or not t or total + len(t) > max_chars: continue
        if depth == "fast" and not KEEP_FAST.search(nm): continue
        if depth == "standard" and DROP_STANDARD.search(nm): continue
        while len(t) > chunk_chars * 1.5:                                  # very long section: split on a sentence boundary
            cut = t.rfind(". ", 0, chunk_chars); cut = cut + 1 if cut > chunk_chars // 2 else chunk_chars
            if buf: flush()
            page0 = sc["page"]; buf.append((nm, t[:cut])); size = cut; flush(); t = t[cut:].strip()
        if buf and size + len(t) > chunk_chars: flush()
        if not buf: page0 = sc["page"]
        buf.append((nm, t)); size += len(t); total += len(t)
    flush()
    if not any(re.search("abstract|intro", x["name"], re.I) for x in secs) and not any("front" in c["sections"][0].lower() for c in chunks): flags.append("no Abstract/Introduction heading found: structure may be unusual")
    full = norm_ws(" ".join(texts))
    return dict(id=Path(path).stem, title=norm_ws(title), source="pdf", pages=[norm_ws(t).lower() for t in texts], chunks=chunks, full_lower=full.lower(),
                sentences=[x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])", full) if len(x) > 20], quality=dict(pages=n, chars=chars, chunks=len(chunks), sections=[x["name"] for x in secs if x["name"] != "Front matter"][:20], flags=flags))

def parse_text(pid, title, text):
    """Abstract-only paper (no PDF available): same structure so the rest of the pipeline is unchanged."""
    full = norm_ws(text); body = f"[Abstract]\n{full}"
    return dict(id=pid, title=norm_ws(title), source="abstract", pages=[full.lower()], chunks=[dict(page=1, sections=["Abstract"], text=body)] if len(full) > 80 else [], full_lower=full.lower(),
                sentences=[x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])", full) if len(x) > 20], quality=dict(pages=1, chars=len(full), chunks=1, sections=["Abstract"], flags=["abstract only (no full text)"] + ([] if len(full) > 80 else ["abstract too short"])))

# ───────── STEP 3b · LLM EXTRACTION with a strict schema + cache; offline mode uses fixtures ─────────
SCHEMA = """{
 "problems":     [{"name": str, "passage": str}],
 "methods":      [{"name": str, "role": "proposed"|"baseline"|"used", "passage": str}],
 "datasets":     [{"name": str, "passage": str}],
 "contexts":     [{"name": str, "passage": str}],
 "applications": [{"method": str, "problem": str, "passage": str}],
 "evaluations":  [{"method": str, "dataset": str, "passage": str}],
 "results":      [{"method": str, "dataset": str, "metric": str, "value": number, "context": str|null, "passage": str}],
 "limitations":  [{"text": str, "passage": str}],
 "future_work":  [{"text": str, "mentions": [str], "passage": str}]
}"""
SYSTEM = "You extract facts from scientific papers into a fixed JSON schema. Reply with a single JSON object and nothing else."
EXAMPLE = ('Text: "We apply LoRA to Swahili translation. LoRA reaches 19.8 BLEU on FLORES-200. Future work: test on Hausa."\n'
           'Output: {"problems":[{"name":"low-resource machine translation","passage":"We apply LoRA to Swahili translation."}],"methods":[{"name":"LoRA","role":"proposed","passage":"We apply LoRA to Swahili translation."}],'
           '"datasets":[{"name":"FLORES-200","passage":"LoRA reaches 19.8 BLEU on FLORES-200."}],"contexts":[{"name":"Swahili","passage":"We apply LoRA to Swahili translation."}],'
           '"applications":[{"method":"LoRA","problem":"low-resource machine translation","passage":"We apply LoRA to Swahili translation."}],"evaluations":[{"method":"LoRA","dataset":"FLORES-200","passage":"LoRA reaches 19.8 BLEU on FLORES-200."}],'
           '"results":[{"method":"LoRA","dataset":"FLORES-200","metric":"BLEU","value":19.8,"context":"Swahili","passage":"LoRA reaches 19.8 BLEU on FLORES-200."}],"limitations":[],'
           '"future_work":[{"text":"Test on Hausa.","mentions":["Hausa"],"passage":"Future work: test on Hausa."}]}')
USER = ("Paper title: {title}\nExtract facts from the paper section below into JSON with exactly this schema (use [] when nothing applies):\n{schema}\n\n"
        "Rules:\n- problems = research problems or tasks; contexts = languages, populations or domains studied; applications = a method applied to a problem.\n"
        "- \"passage\" must be ONE sentence copied verbatim from the text. Never paraphrase or merge sentences.\n"
        "- Use short canonical names (for example \"mBART\", \"BLEU\", \"Swahili\").\n- Extract only what the text explicitly states. Do not guess or add outside knowledge.\n"
        "- \"value\" must be a number. \"mentions\" lists the methods, problems, contexts or datasets a future-work statement names.\n\nEXAMPLE\n{example}\n\nTEXT:\n{text}")
KEYS = ["problems", "methods", "datasets", "contexts", "applications", "evaluations", "results", "limitations", "future_work"]
_S = {"type": "string"}
def _arr(props, req): return {"type": "array", "items": {"type": "object", "properties": props, "required": req}}
EXTRACT_SCHEMA = {"type": "object", "required": KEYS, "properties": {   # constrains output on providers that support it (Gemini, Ollama); others rely on the prompt
    "problems": _arr({"name": _S, "passage": _S}, ["name", "passage"]), "datasets": _arr({"name": _S, "passage": _S}, ["name", "passage"]), "contexts": _arr({"name": _S, "passage": _S}, ["name", "passage"]),
    "methods": _arr({"name": _S, "role": {"type": "string", "enum": ["proposed", "baseline", "used"]}, "passage": _S}, ["name", "passage"]),
    "applications": _arr({"method": _S, "problem": _S, "passage": _S}, ["method", "problem", "passage"]), "evaluations": _arr({"method": _S, "dataset": _S, "passage": _S}, ["method", "dataset", "passage"]),
    "results": _arr({"method": _S, "dataset": _S, "metric": _S, "value": {"type": "number"}, "context": _S, "passage": _S}, ["method", "dataset", "metric", "value", "passage"]),
    "limitations": _arr({"text": _S, "passage": _S}, ["text", "passage"]), "future_work": _arr({"text": _S, "mentions": {"type": "array", "items": _S}, "passage": _S}, ["text", "passage"])}}
FATAL = (llm.LLMConfigError, llm.LLMModelError, llm.LLMRateLimit, llm.LLMTooLarge)       # stop the whole job; progress is saved

def extract_paper(paper, mode="llm", cache_dir=HERE / "cache", log=print):
    """mode 'fixture' (offline demo) or 'llm' (Groq). A chunk that fails (bad JSON twice) is recorded and skipped; fatal errors (key, model, daily limit) propagate."""
    if mode == "fixture":
        f = HERE / "fixtures" / f"{paper['id']}.json"
        if not f.exists(): raise FileNotFoundError(f"no offline fixture for {paper['id']} - set GROQ_API_KEY and use LLM mode")
        return json.loads(f.read_text(encoding="utf-8")), dict(calls=0, cached=0, errors=[])
    Path(cache_dir).mkdir(parents=True, exist_ok=True); merged = {k: [] for k in KEYS}; st = dict(calls=0, cached=0, errors=[])
    for i, ch in enumerate(paper["chunks"]):
        prompt = USER.format(title=paper.get("title", ""), schema=SCHEMA, example=EXAMPLE, text=ch["text"])
        key = hashlib.sha1((llm.label() + "v3" + prompt).encode()).hexdigest(); cf = Path(cache_dir) / f"{key}.json"
        if cf.exists(): raw = json.loads(cf.read_text(encoding="utf-8")); st["cached"] += 1
        else:
            try: raw = llm.chat_json(SYSTEM, prompt, max_tokens=llm.output_tokens(), schema=EXTRACT_SCHEMA); cf.write_text(json.dumps(raw), encoding="utf-8"); st["calls"] += 1
            except FATAL: raise
            except llm.LLMError as e: st["errors"].append(f"chunk {i + 1}: {str(e)[:120]}"); log(f"  chunk {i + 1} skipped: {e}"); continue
        for k in KEYS: merged[k] += [x for x in (raw.get(k) or []) if isinstance(x, dict)]
        log(f"  chunk {i + 1}/{len(paper['chunks'])} done")
    return merged, st

# ───────── STEP 3c · VALIDATION + PROVENANCE: reject anything that does not match the schema or is not quoted from the paper ─────────
def _passage(p, paper):
    n = norm_ws(p)
    if n and n.lower() in paper["full_lower"]: return n
    m = difflib.get_close_matches(n, paper["sentences"], n=1, cutoff=0.85); return m[0] if m else None
def _page(p, paper): return next((i for i, pg in enumerate(paper["pages"], 1) if p.lower() in pg), 1)

SPEC = {"problems": ("problem", ["name"]), "methods": ("method", ["name"]), "datasets": ("dataset", ["name"]), "contexts": ("context", ["name"]),
        "applications": ("applied", ["method", "problem"]), "evaluations": ("evaluated", ["method", "dataset"]), "results": ("result", ["method", "dataset", "metric", "value"]),
        "limitations": ("limitation", ["text"]), "future_work": ("future", ["text"])}

def validate(raw, paper):
    facts, dropped = [], []
    for key, (kind, req) in SPEC.items():
        for it in raw.get(key) or []:
            if not isinstance(it, dict) or any(it.get(r) in (None, "") for r in req): dropped.append((kind, "missing required field", it)); continue
            ps = _passage(it.get("passage", ""), paper)
            if ps is None: dropped.append((kind, "passage is not a verbatim sentence of the paper", it)); continue
            f = dict(kind=kind, paper=paper["id"], page=_page(ps, paper), passage=ps)
            for r in req:
                if r == "value":
                    try: f["value"] = float(it["value"])
                    except (TypeError, ValueError): f = None; dropped.append((kind, "value is not a number", it)); break
                else: f[r if r != "text" else "text"] = norm_ws(it[r]).strip(" .,:;") if r != "text" else norm_ws(it[r])
            if f is None: continue
            if kind == "method": f["role"] = it.get("role") if it.get("role") in ("proposed", "baseline", "used") else "used"
            if kind == "result": f["context"] = norm_ws(it.get("context") or "").strip(" .,:;")
            if kind == "future": f["mentions"] = [norm_ws(m).strip(" .,:;") for m in (it.get("mentions") or []) if isinstance(m, str) and m.strip()]
            facts.append(f)
    return facts, dropped

# ───────── STEP 3d · ENTITY NORMALIZATION (alias list -> exact -> fuzzy; ambiguous pairs go to an LLM yes/no or to a review list) ─────────
FIELDS = {"Method": [("method", "name"), ("applied", "method"), ("evaluated", "method"), ("result", "method")], "Problem": [("problem", "name"), ("applied", "problem")],
          "Dataset": [("dataset", "name"), ("evaluated", "dataset"), ("result", "dataset")], "Metric": [("result", "metric")], "Context": [("context", "name"), ("result", "context")]}

def load_aliases(path=HERE / "aliases.json"): return json.loads(Path(path).read_text(encoding="utf-8"))

def normalize(facts, aliases, confirm=None, max_confirm=15):
    """Returns (facts rewritten with canonical names, review list of unresolved near-duplicates, merge log).
    Names are merged by alias list, then exact match (case/punctuation), then fuzzy similarity: >= .92 merges automatically (chains allowed:
    'Transformers' and 'Transformerss' both join 'Transformer'); .80 to .92 asks `confirm` (an LLM yes/no, at most `max_confirm` calls) or goes to the review list."""
    maps, review, log, asked = {}, [], [], 0
    for typ, fl in FIELDS.items():
        m = {}; seeds = aliases.get(typ, {})
        for canon, als in seeds.items():
            for s in [canon] + als: m[alnum(s)] = canon
        cnt = Counter(f[fld] for f in facts for kind, fld in fl if f["kind"] == kind and f.get(fld))
        groups = defaultdict(Counter); freq = Counter()
        for s, c in cnt.items():
            freq[alnum(s)] += c
            if alnum(s) not in m: groups[alnum(s)][s] += c
        for k, ctr in groups.items(): m[k] = sorted(ctr, key=lambda s: (-ctr[s], len(s)))[0]
        shown = {alnum(v): v for v in set(m.values())}                       # one display name per distinct entity
        order = sorted(shown, key=lambda k: (-freq.get(k, 0), k)); alive = set(order); members = {k: [k] for k in order}; rev = []
        def ratio(x, y):
            if 2 * min(len(x), len(y)) / (len(x) + len(y) or 1) < .8: return 0.0       # cheap length bound before the expensive comparison
            sm = difflib.SequenceMatcher(None, x, y)
            return sm.ratio() if sm.real_quick_ratio() >= .8 and sm.quick_ratio() >= .8 else 0.0
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                if a not in alive or b not in alive: continue                  # b may have been merged away earlier: skip instead of looking it up
                r = max(ratio(x, b) for x in members[a])
                if r < .80: continue
                ca, cb = shown[a], shown[b]; same = r >= .92
                if not same and confirm and asked < max_confirm: asked += 1; same = bool(confirm(typ, ca, cb))
                if same:
                    for k2, v in list(m.items()):
                        if v == cb: m[k2] = ca
                    alive.discard(b); members[a] += members.pop(b); log.append((typ, cb, ca, round(r, 2)))
                else: rev.append((typ, ca, cb, round(r, 2)))
        final = {shown[k] for k in alive}; review += [x for x in rev if x[1] in final and x[2] in final]      # only pairs that are still separate entities
        maps[typ] = m
    out = []
    for f in facts:
        g = dict(f)
        for typ, fl in FIELDS.items():
            for kind, fld in fl:
                if g["kind"] == kind and g.get(fld): g[fld] = maps[typ].get(alnum(g[fld]), g[fld])
        if g["kind"] == "future":
            g["mentions"] = list(dict.fromkeys(next((maps[t][alnum(x)] for t in maps if alnum(x) in maps[t]), x) for x in g["mentions"]))
        out.append(g)
    review = sorted(review, key=lambda x: -x[3])[:200]                        # keep the file small: the most similar pairs are the ones worth a human look
    return out, review, log

def llm_confirm(typ, a, b):
    if not llm.available(): return False
    try: return bool(llm.chat_json("Answer with JSON only.", f'Are "{a}" and "{b}" two names for the same {typ.lower()} in machine-learning papers? Reply {{"same": true|false}}.').get("same"))
    except Exception: return False

# ───────── STEP 4 · GAP DETECTORS (in-memory mirror of the Cypher queries in graph.py; used offline and for cross-checking) ─────────
THRESH = 2.0   # results of the same method/dataset/metric/context differing by >= this many points count as conflicting
def _gap(typ, key, title, entities, evidence, terms, feas, **meta):
    return dict(type=typ, key=key, title=title, entities=entities, evidence=evidence, verify_terms=terms, support_papers=len({e["paper"] for e in evidence}), meta=dict(feasibility=feas, **meta))
def _ev(f, note): return dict(paper=f["paper"], page=f["page"], passage=f["passage"], note=note)

def detect_a1(facts):
    app = defaultdict(list)
    for f in facts:
        if f["kind"] == "applied": app[(f["method"], f["problem"])].append(f)
    by_m, by_p = defaultdict(set), defaultdict(set)
    for m, p in app: by_m[m].add(p); by_p[p].add(m)
    ev_ds = defaultdict(set)
    for f in facts:
        if f["kind"] == "evaluated": ev_ds[f["method"]].add(f["dataset"])
    out = []
    for m in sorted(by_m):
        done = by_m[m]
        if len(done) < 2: continue
        for c in sorted(by_p):
            others = sorted(by_p[c] - {m}) if c not in done else []
            if c in done or not others: continue
            ev = [_ev(f, f"{m} applied to {a}") for a in sorted(done) for f in app[(m, a)][:1]] + [_ev(f, f"{o} applied to {c}") for o in others for f in app[(o, c)][:1]]
            feas = 1.0 if any(ev_ds[o] for o in others) else .3
            out.append(_gap("method_problem", f"mp|{m}|{c}", f"{m} has not been applied to {c}", [m, c], ev, [[m], [c]], feas, applied_to=sorted(done), others=others))
    return out

def detect_a2(facts):
    cp, pp, cds = defaultdict(set), defaultdict(set), defaultdict(set)
    for f in facts:
        if f["kind"] == "context": cp[f["name"]].add(f["paper"])
        if f["kind"] == "problem": pp[f["name"]].add(f["paper"])
    ds_by_paper = defaultdict(set)
    for f in facts:
        if f["kind"] == "dataset": ds_by_paper[f["paper"]].add(f["name"])
    out = []
    for x in sorted(cp):
        for y in sorted(pp):
            if len(cp[x]) < 2 or len(pp[y]) < 2 or cp[x] & pp[y]: continue
            ev = [_ev(f, f"studies {x}") for f in facts if f["kind"] == "context" and f["name"] == x] + [_ev(f, f"addresses {y}") for f in facts if f["kind"] == "problem" and f["name"] == y]
            feas = 1.0 if any(ds_by_paper[p] for p in cp[x]) else .3
            out.append(_gap("context_problem", f"cp|{x}|{y}", f"{x} is never studied together with {y}", [x, y], ev, [[x], [y]], feas, context_papers=sorted(cp[x]), problem_papers=sorted(pp[y])))
    return out

def detect_b1(facts, thr=THRESH):
    groups = defaultdict(list)
    for f in facts:
        if f["kind"] == "result": groups[(f["method"], f["dataset"], f["metric"], f["context"])].append(f)
    out = []
    for (m, d, mt, c), rs in sorted(groups.items()):
        for a, b in itertools.combinations(sorted(rs, key=lambda r: r["paper"]), 2):
            if a["paper"] == b["paper"] or abs(a["value"] - b["value"]) < thr: continue
            where = f" ({c})" if c else ""
            out.append(_gap("conflict", f"cf|{m}|{d}|{mt}|{c}|{a['paper']}|{b['paper']}", f"Conflicting results for {m} on {d}, {mt}{where}: {a['value']:g} vs {b['value']:g}", [m, d, mt] + ([c] if c else []),
                            [_ev(a, f"{a['paper']} reports {a['value']:g}"), _ev(b, f"{b['paper']} reports {b['value']:g}")], [], 1.0, values=[a["value"], b["value"]]))
    return out

def fw_items_py(facts):
    ent = {}  # canonical entity name -> set of papers connected to it
    for f in facts:
        if f["kind"] in ("problem", "method", "dataset", "context"): ent.setdefault(f["name"], set()).add(f["paper"])
    items = []
    for f in facts:
        if f["kind"] != "future": continue
        matched = [m for m in f["mentions"] if m in ent]; unmatched = [m for m in f["mentions"] if m not in ent]
        others = {p for m in matched for p in ent[m]} - {f["paper"]}
        covering = sorted(p for p in others if all(p in ent[m] for m in matched)) if matched and not unmatched else []
        items.append(dict(fid=f"{f['paper']}|{alnum(f['text'])[:40]}", paper=f["paper"], text=f["text"], page=f["page"], passage=f["passage"], matched=matched, unmatched=unmatched,
                          covering=covering, candidates=sorted(others)))
    return items

def passages_py(facts, paper, limit=8):
    seen, out = set(), []
    for f in facts:
        if f["paper"] == paper and f["passage"] not in seen and f["kind"] != "future": seen.add(f["passage"]); out.append(dict(page=f["page"], passage=f["passage"]))
    return out[:limit]

JUDGE = ("You judge whether a future-work statement from one paper is already addressed by other papers in a corpus. Use ONLY the passages provided. "
         'Reply JSON: {"addressed": true|false, "paper": "<paper id or null>", "quote": "<one passage copied verbatim from the provided passages, or null>"}. '
         "Say addressed=true only if a passage clearly shows the proposed work was done.")
def judge_future(item, get_passages, use_llm):
    """Returns (addressed: bool, by: str|None, note: str)."""
    if item["covering"]: return True, item["covering"][0], "another paper covers all entities named in the statement"
    if not use_llm or not item["candidates"]: return False, None, "no paper in the corpus covers the entities named" + ("" if use_llm else " (LLM judge not used)")
    cands = {p: get_passages(p) for p in item["candidates"][:4]}
    txt = "\n".join(f"[{p}] {x['passage']}" for p, ps in cands.items() for x in ps)
    try:
        r = llm.chat_json(JUDGE, f"Future-work statement (from {item['paper']}): {item['text']}\n\nPassages:\n{txt}")
        q = norm_ws(r.get("quote") or "").lower()
        if r.get("addressed") and q and q in norm_ws(txt).lower(): return True, r.get("paper"), f"LLM judge, quote verified: \"{r.get('quote')}\""
        return False, None, "LLM judge found no passage that addresses it"
    except Exception as e: return False, None, f"LLM judge failed ({type(e).__name__}); treated as unaddressed"

def build_b2(items, get_passages, use_llm, extra_evidence):
    out = []
    for it in items:
        addressed, by, note = judge_future(it, get_passages, use_llm)
        if addressed: continue
        ev = [dict(paper=it["paper"], page=it["page"], passage=it["passage"], note="future-work statement")] + extra_evidence(it)
        out.append(_gap("future_work", f"fw|{it['fid']}", f"Unaddressed future work ({it['paper']}): {it['text']}", it["matched"] + it["unmatched"], ev, [[m] for m in it["matched"] + it["unmatched"]],
                        .5 if it["unmatched"] else .8, judge=note, unmatched=it["unmatched"]))
    return out

def detect_all_py(facts, use_llm=False):
    ps = lambda p: passages_py(facts, p)
    def related(it):
        rel = [_ev(f, f"related: {f['name']}") for f in facts if f["kind"] in ("problem", "method", "dataset", "context") and f["name"] in it["matched"] and f["paper"] != it["paper"]]
        return rel[:4]
    return detect_a1(facts) + detect_a2(facts) + detect_b1(facts) + build_b2(fw_items_py(facts), ps, use_llm, related)

# ───────── STEP 5 · EXTERNAL VERIFICATION (OpenAlex) ─────────
def openalex_search(query, n=5, mailto=None):
    r = requests.get(os.getenv("OPENALEX_URL", "https://api.openalex.org").rstrip("/") + "/works", params={"search": query, "per-page": n, **({"mailto": mailto} if mailto else {})}, timeout=20); r.raise_for_status()
    return r.json().get("results", [])
def _abstract(w): inv = w.get("abstract_inverted_index") or {}; return " ".join(wd for _, wd in sorted((pos, wd) for wd, ps in inv.items() for pos in ps))

def verify_gap(gap, aliases, search=openalex_search):
    if gap["type"] == "llm_gap": return _verify_title(gap, search)
    if gap["type"] == "conflict": return dict(status="not applicable", papers=[], query="", note="Conflicts are checked by reading the two papers, not by search.")
    syn = {}
    for typ in aliases.values():
        for canon, als in typ.items(): syn[words(canon)] = [words(canon)] + [words(a) for a in als]
    groups = [syn.get(words(t[0]), [words(t[0])]) for t in gap["verify_terms"] if t]
    query = " ".join(t[0] for t in gap["verify_terms"] if t)
    try: results = search(query)
    except Exception as e: return dict(status="unavailable", papers=[], query=query, note=f"external search failed: {type(e).__name__}")
    hits = []
    for w in results:
        text = words((w.get("title") or "") + " " + _abstract(w)); got = sum(any(s in text for s in g) for g in groups)
        hits.append(dict(title=w.get("title"), year=w.get("publication_year"), url=w.get("doi") or w.get("id"), matched=got, of=len(groups)))
    full = [h for h in hits if h["of"] and h["matched"] == h["of"]]
    part = [h for h in hits if len(groups) >= 3 and h["matched"] >= 2]
    status = "exists externally" if full else ("partially addressed" if part else "not found externally")
    return dict(status=status, papers=(full or part)[:3], query=query, note="Based on the top-5 OpenAlex results only: a 'not found' is a lead to check, not proof of absence.")

def _verify_title(gap, search):
    """LLM-proposed gaps have no entity list: search by title and measure how much of it a result's title+abstract covers."""
    q = gap["title"]
    try: results = search(q)
    except Exception as e: return dict(status="unavailable", papers=[], query=q, note=f"external search failed: {type(e).__name__}")
    tw = {w for w in words(q).split() if w not in STOP and len(w) > 3}; hits = []
    for w in results:
        txt = set(words((w.get("title") or "") + " " + _abstract(w)).split())
        hits.append(dict(title=w.get("title"), year=w.get("publication_year"), url=w.get("doi") or w.get("id"), matched=round(len(tw & txt) / len(tw), 2) if tw else 0.0, of=1))
    best = max((h["matched"] for h in hits), default=0)
    status = "exists externally" if best >= .8 else ("partially addressed" if best >= .6 else "not found externally")
    return dict(status=status, papers=sorted(hits, key=lambda h: -h["matched"])[:3] if best >= .6 else [], query=q, note="Title-overlap check on the top-5 OpenAlex results: a lead, not proof.")

# ───────── STEP 6 · RANKING with visible, adjustable criteria ─────────
NOVELTY = {"not found externally": 1.0, "partially addressed": .6, "exists externally": .15, "unavailable": .5, "not applicable": .5, "not checked": .5}
DEFAULT_W = dict(evidence=.35, novelty=.30, relevance=.20, feasibility=.15)
STOP = set("the a an of in on for and to with from by using is are".split())
def components(gap, topic, titles):
    gt = set(words(" ".join(gap["entities"] + [titles.get(e["paper"], "") for e in gap["evidence"]] + [e["passage"] for e in gap["evidence"]])).split()); tt = {t for t in words(topic).split() if t not in STOP and len(t) > 2}
    return dict(evidence=min(gap["support_papers"] / 4, 1.0), novelty=NOVELTY.get(gap.get("verification", {}).get("status", "not checked"), .5),
                relevance=(len(tt & gt) / len(tt)) if tt else .5, feasibility=gap["meta"].get("feasibility", .5))
def rank(gaps, topic, titles, weights=DEFAULT_W):
    for g in gaps:
        g["components"] = components(g, topic, titles); s = sum(weights.values()) or 1
        g["score"] = sum(weights[k] * g["components"][k] for k in weights) / s
    return sorted(gaps, key=lambda g: -g["score"])

# ───────── EXPLANATION (LLM explains; it never discovers) ─────────
def explain(gap, titles, use_llm):
    facts = "\n".join(f"- {titles.get(e['paper'], e['paper'])}, p.{e['page']}: \"{e['passage']}\" ({e['note']})" for e in gap["evidence"][:8])
    base = {"method_problem": "Within this corpus, {0} has been used on several problems but never on {1}, although other methods have been applied to {1}.",
            "context_problem": "Within this corpus, {0} and {1} are each studied by at least two papers, but no paper studies them together.",
            "conflict": "Two papers report different values for the same method, dataset, metric and context, which suggests unreported differences in setup.",
            "future_work": "An author proposes this as future work, and no other paper in the corpus appears to address it."}[gap["type"]]
    tpl = base.format(*(gap["entities"] + ["", ""])[:2]) + " This is a potential research direction supported by the evidence below, to be validated against the wider literature."
    if not use_llm: return tpl
    try: return llm.chat_text("You explain research gaps in 2-3 plain sentences using ONLY the facts given. Say it is a potential gap in this corpus, never claim novelty.", f"Gap: {gap['title']}\nFacts:\n{facts}")
    except Exception: return tpl
