"""RQ2: corpus-withholding experiment + coverage-aware confidence (abstention) for STRUCTURAL gaps.
Label (no human needed): a gap found in a reduced corpus is 'false by incompleteness' if the FULL corpus fills it."""
import math, json, random
from pathlib import Path
import numpy as np
import core

def structural(facts): return core.detect_a1(facts) + core.detect_a2(facts)
def papers_of(facts): return sorted({f["paper"] for f in facts})

def withhold(facts, frac, seed=0, mode="random"):
    """mode 'recent' drops the last papers in sorted-id order (arXiv ids sort chronologically)."""
    ps = papers_of(facts); k = int(round(len(ps) * frac)); drop = set(ps[len(ps) - k:]) if mode == "recent" else set(random.Random(seed).sample(ps, k))
    return [f for f in facts if f["paper"] not in drop], drop

def is_filled(g, full):
    if g["type"] == "method_problem": m, c = g["entities"]; return any(f["kind"] == "applied" and f["method"] == m and f["problem"] == c for f in full)
    x, y = g["entities"]; return bool({f["paper"] for f in full if f["kind"] == "context" and f["name"] == x} & {f["paper"] for f in full if f["kind"] == "problem" and f["name"] == y})

class Idx:
    """Per-corpus counts used by the confidence features (built once per corpus, not per gap)."""
    def __init__(self, facts):
        self.n = len(papers_of(facts)); self.m, self.c = {}, {}
        for f in facts:
            if f["kind"] == "applied": self.m.setdefault(f["method"], set()).add(f["paper"]); self.c.setdefault(f["problem"], set()).add(f["paper"])

FEATS = ["log_support", "log_n_a", "log_n_b", "log_breadth", "log_others", "log_papers"]
def features(g, idx):
    if g["type"] == "method_problem": m, c = g["entities"]; a, b = len(idx.m.get(m, ())), len(idx.c.get(c, ())); br, ot = len(g["meta"]["applied_to"]), len(g["meta"]["others"])
    else: a, b, br, ot = len(g["meta"]["context_papers"]), len(g["meta"]["problem_papers"]), 0, 0
    return [math.log1p(g["support_papers"]), math.log1p(a), math.log1p(b), math.log1p(br), math.log1p(ot), math.log(max(idx.n, 1))]

def build_rows(facts_by_topic, fracs=(.2, .4), seeds=(0, 1, 2), modes=("random",)):
    rows = []
    for topic, full in facts_by_topic.items():
        for frac in fracs:
            for mode in modes:
                for seed in (seeds if mode == "random" else (0,)):
                    red, _ = withhold(full, frac, seed, mode); idx = Idx(red)
                    for g in structural(red): rows.append(dict(topic=topic, frac=frac, mode=mode, seed=seed, key=g["key"], type=g["type"], x=features(g, idx), support=g["support_papers"], false=int(is_filled(g, full))))
    return rows

class Logit:
    def fit(self, X, y, l2=1.0, iters=600, lr=0.2):
        X = np.asarray(X, float); y = np.asarray(y, float); self.mu, self.sd = X.mean(0), X.std(0) + 1e-9; Z = (X - self.mu) / self.sd; self.w = np.zeros(Z.shape[1]); self.b = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b))); self.w -= lr * (Z.T @ (p - y) / len(y) + l2 * self.w / len(y)); self.b -= lr * float((p - y).mean())
        return self
    def proba(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd; return 1 / (1 + np.exp(-(Z @ self.w + self.b)))
    def to_dict(self): return dict(w=self.w.tolist(), b=self.b, mu=self.mu.tolist(), sd=self.sd.tolist(), feats=FEATS)
    @classmethod
    def from_dict(cls, d): m = cls(); m.w, m.b, m.mu, m.sd = np.array(d["w"]), d["b"], np.array(d["mu"]), np.array(d["sd"]); return m

def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]; neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return None
    return sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

def risk_coverage(conf, false, levels=(1.0, .8, .6, .4, .2)):
    f = np.asarray(false)[np.argsort(-np.asarray(conf), kind="stable")]; return [dict(coverage=c, risk=round(float(f[:max(1, int(round(c * len(f))))].mean()), 3)) for c in levels]

def evaluate(rows):
    """Leave-one-topic-out. Compares the learned confidence with an evidence-count baseline and with always answering."""
    topics = sorted({r["topic"] for r in rows}); P, S, F = [], [], []
    for t in topics:
        tr = [r for r in rows if r["topic"] != t]; te = [r for r in rows if r["topic"] == t]
        if not te: continue
        y = [r["false"] for r in tr]
        if len(set(y)) < 2: p = [float(np.mean(y)) if y else .5] * len(te)
        else: p = Logit().fit([r["x"] for r in tr], y).proba([r["x"] for r in te]).tolist()
        P += p; S += [r["support"] for r in te]; F += [r["false"] for r in te]
    conf_model, conf_base = [1 - x for x in P], [float(s) for s in S]
    by_frac = {str(fr): round(float(np.mean([r["false"] for r in rows if r["frac"] == fr])), 3) for fr in sorted({r["frac"] for r in rows})}
    return dict(n=len(F), false_rate=round(float(np.mean(F)), 3) if F else None, false_rate_by_withheld_fraction=by_frac, auroc_model=auroc(P, F), auroc_evidence_count=auroc([-s for s in S], F),
                risk_coverage_model=risk_coverage(conf_model, F), risk_coverage_evidence_count=risk_coverage(conf_base, F), risk_coverage_always_answer=risk_coverage([0] * len(F), F))

def train(rows): return Logit().fit([r["x"] for r in rows], [r["false"] for r in rows]) if len({r["false"] for r in rows}) == 2 else None
def save_model(path, m): Path(path).write_text(json.dumps(m.to_dict()), encoding="utf-8")
def load_model(path): p = Path(path); return Logit.from_dict(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None

def confidence(g, facts, model=None, idx=None):
    """P(gap survives more literature). Only structural gaps are scored; others return None."""
    if g["type"] not in ("method_problem", "context_problem"): return None
    if model is None: c = g["support_papers"] / (g["support_papers"] + 2); return dict(conf=round(c, 2), basis="heuristic (no trained model)")
    return dict(conf=round(float(1 - model.proba([features(g, idx or Idx(facts))])[0]), 2), basis="trained on withholding experiments")
def band(c): return "high" if c >= .75 else "medium" if c >= .5 else "low"
