"""SYNTHETIC facts for testing the abstention code (not for results): papers with skewed method/problem/context popularity."""
import random

def make_facts(n_papers=60, seed=0, n_methods=8, n_problems=6, n_contexts=6, prefix="S"):
    rnd = random.Random(seed); M = [f"M{i}" for i in range(n_methods)]; P = [f"P{i}" for i in range(n_problems)]; C = [f"C{i}" for i in range(n_contexts)]
    wm = [1 / (i + 1) for i in range(n_methods)]; wp = [1 / (i + 1) for i in range(n_problems)]; wc = [1 / (i + 1) for i in range(n_contexts)]; facts = []
    for i in range(n_papers):
        pid = f"{prefix}{seed}_{i:03d}"; ps = set(rnd.choices(P, wp, k=rnd.randint(1, 2))); ms = set(rnd.choices(M, wm, k=rnd.randint(1, 3))); cs = set(rnd.choices(C, wc, k=rnd.randint(1, 2)))
        base = dict(paper=pid, page=1)
        for p in ps: facts.append(dict(base, kind="problem", name=p, passage=f"We study {p}."))
        for c in cs: facts.append(dict(base, kind="context", name=c, passage=f"We work on {c}."))
        for m in ms:
            facts.append(dict(base, kind="method", name=m, role="used", passage=f"We use {m}."))
            for p in ps: facts.append(dict(base, kind="applied", method=m, problem=p, passage=f"{m} is applied to {p}."))
        facts.append(dict(base, kind="dataset", name=f"D{rnd.randint(0, 3)}", passage="We use a dataset."))
    return facts
