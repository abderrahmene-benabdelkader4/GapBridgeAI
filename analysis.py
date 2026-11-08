"""Rating analysis: agreement (Cohen's kappa), per-condition metrics with cluster-bootstrap CIs (resampling topics, then items), LLM-judge comparison."""
import json, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
import llm

SCHEMA = "CREATE TABLE IF NOT EXISTS ratings (item_id TEXT, rater TEXT, support TEXT, valid TEXT, novel TEXT, useful INTEGER, tag TEXT, note TEXT, ts TEXT, PRIMARY KEY (item_id, rater))"
def load_ratings(db):
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row; con.execute(SCHEMA); rows = [dict(r) for r in con.execute("SELECT * FROM ratings")]; con.close(); return rows

def kappa(a, b):
    n = len(a)
    if n == 0: return None
    po = sum(x == y for x, y in zip(a, b)) / n; pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b)); return None if pe == 1 else round((po - pe) / (1 - pe), 3)

def verdicts(ratings):
    by = defaultdict(dict)
    for r in ratings: by[r["item_id"]][r["rater"]] = r
    raters = sorted({r["rater"] for r in ratings}); out = {}
    for i, rs in by.items():
        if len(rs) < len(raters): continue                       # only items rated by every rater
        v = list(rs.values()); ok = lambda x, s: x["support"] in s and x["valid"] == "yes"
        out[i] = dict(strict=float(all(ok(x, ("yes",)) for x in v)), lenient=float(all(ok(x, ("yes", "partly")) for x in v)), any=float(any(ok(x, ("yes",)) for x in v)),
                      supported=float(all(x["support"] == "yes" for x in v)), novel=float(all(x["novel"] == "yes" for x in v)), useful=float(np.mean([x["useful"] for x in v])))
    return out, raters

def _pooled(cti, verd, field, topics, rng=None):
    vals = []
    for t in topics:
        ids = cti.get(t, [])
        if rng is not None and ids: ids = [ids[j] for j in rng.integers(0, len(ids), len(ids))]
        vals += [verd[i][field] for i in ids]
    return float(np.mean(vals)) if vals else None

def _ci(fn, topics, n_boot, rng):
    reps = []
    for _ in range(n_boot):
        v = fn([topics[j] for j in rng.integers(0, len(topics), len(topics))], rng)
        if v is not None: reps.append(v)
    return [round(float(np.percentile(reps, 2.5)), 3), round(float(np.percentile(reps, 97.5)), 3)] if len(reps) > 20 else None

def analyze(run_dir, n_boot=2000, seed=0):
    d = Path(run_dir); key = json.loads((d / "pool_key.json").read_text(encoding="utf-8")); cands = json.loads((d / "candidates.json").read_text(encoding="utf-8"))
    verd, raters = verdicts(load_ratings(d / "ratings.sqlite")); rng = np.random.default_rng(seed); conds = sorted({o["condition"] for k in key.values() for o in k["origins"]})
    cti = {c: defaultdict(list) for c in conds}; dup = {c: 0 for c in conds}; total = {c: sum(1 for x in cands if x["condition"] == c) for c in conds}
    for i, k in key.items():
        for c in conds:
            n_orig = sum(o["condition"] == c for o in k["origins"])
            if n_orig: dup[c] += n_orig - 1
            if n_orig and i in verd: cti[c][k["project"]].append(i)
    topics = sorted({k["project"] for k in key.values()}); res = dict(raters=raters, items_rated_by_all=len(verd), conditions={}, differences={}, kappa={}, note="Exploratory: few topics, so intervals are wide.")
    for c in conds:
        m = dict(n_items=sum(len(v) for v in cti[c].values()), duplicate_rate=round(dup[c] / total[c], 3) if total[c] else None)
        for f in ("strict", "lenient", "supported", "novel", "useful"):
            est = _pooled(cti[c], verd, f, topics); m[f] = dict(est=None if est is None else round(est, 3), ci95=_ci(lambda tp, r, f=f, c=c: _pooled(cti[c], verd, f, tp, r), topics, n_boot, rng))
        res["conditions"][c] = m
    for a, b in (("G", "V"), ("H", "V"), ("H", "G")):
        if a in conds and b in conds:
            fn = lambda tp, r, a=a, b=b: (lambda x, y: None if x is None or y is None else x - y)(_pooled(cti[a], verd, "strict", tp, r), _pooled(cti[b], verd, "strict", tp, r))
            est = fn(topics, None); res["differences"][f"{a}-{b} (strict precision)"] = dict(est=None if est is None else round(est, 3), ci95=_ci(fn, topics, n_boot, rng))
    rs = load_ratings(d / "ratings.sqlite"); by = defaultdict(dict)
    for r in rs: by[r["rater"]][r["item_id"]] = r
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            common = sorted(set(by[a]) & set(by[b])); res["kappa"][f"{a} vs {b}"] = dict(n=len(common), **{q: kappa([by[a][x][q] for x in common], [by[b][x][q] for x in common]) for q in ("support", "valid", "novel")})
    (d / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8"); return res

JUDGE = ('You are a strict reviewer of research-gap candidates. Use ONLY the title, description and passages. Reply JSON: {"support":"yes|partly|no","valid":"yes|no|unsure","novel":"yes|no|unsure"} where '
         'support = do the passages support the claim, valid = is it a real gap in these papers, novel = would it be new to the wider literature.')
def judge(run_dir, log=print):
    """RQ3: an LLM judge rates the same items; compare with expert consensus."""
    d = Path(run_dir); blind = {b["item_id"]: b for b in json.loads((d / "pool_blind.json").read_text(encoding="utf-8"))}; verd, _ = verdicts(load_ratings(d / "ratings.sqlite")); out = {}
    for i in verd:
        b = blind[i]; ev = "\n".join(f"- {e['paper']} p.{e['page']}: {e['passage']}" for e in b["evidence"])
        try: out[i] = llm.chat_json(JUDGE, f"Title: {b['title']}\nDescription: {b['description']}\nPassages:\n{ev}", max_tokens=200)
        except Exception as e: log(f"judge failed on {i}: {type(e).__name__}")
    ids = [i for i in out if i in verd]; strict = [float(out[i].get("support") == "yes" and out[i].get("valid") == "yes") for i in ids]
    res = dict(n=len(ids), agreement_strict=round(float(np.mean([s == verd[i]["strict"] for s, i in zip(strict, ids)])), 3) if ids else None,
               llm_novel_rate=round(float(np.mean([out[i].get("novel") == "yes" for i in ids])), 3) if ids else None, expert_novel_rate=round(float(np.mean([verd[i]["novel"] for i in ids])), 3) if ids else None,
               llm_strict_rate=round(float(np.mean(strict)), 3) if ids else None, expert_strict_rate=round(float(np.mean([verd[i]["strict"] for i in ids])), 3) if ids else None)
    (d / "llm_judge.json").write_text(json.dumps(dict(summary=res, items=out), indent=1), encoding="utf-8"); return res
