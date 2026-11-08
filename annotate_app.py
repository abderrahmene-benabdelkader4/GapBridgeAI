"""Blinded rating page for the expert raters:  streamlit run annotate_app.py
Shows title, description and evidence only. The condition (V/G/H) is never displayed; it lives in pool_key.json."""
import json, sqlite3, hashlib, time
from pathlib import Path
import pandas as pd, streamlit as st
from analysis import SCHEMA

ROOT = Path(__file__).parent / "experiments"; S = st.session_state
st.set_page_config(page_title="Rate research-gap candidates", layout="centered")
runs = sorted([p.name for p in ROOT.glob("*") if (p / "pool_blind.json").exists()], reverse=True) if ROOT.exists() else []
with st.sidebar:
    st.header("Rater"); run = st.selectbox("Rating run", runs) if runs else None; rater = st.text_input("Your name or id").strip()
    st.caption("Your ratings are saved after every item. You can close the page and resume later.")
st.title("Rate research-gap candidates")
if not runs: st.info("No experiment run found. Create one with:  python cli.py experiment --projects ..."); st.stop()
if not rater: st.info("Enter your name or id in the sidebar to start."); st.stop()
d = ROOT / run; items = json.loads((d / "pool_blind.json").read_text(encoding="utf-8")); items.sort(key=lambda it: hashlib.sha1((rater + it["item_id"]).encode()).hexdigest())
con = sqlite3.connect(d / "ratings.sqlite"); con.execute(SCHEMA); con.commit()
def mine(): return {r[0]: r for r in con.execute("SELECT item_id, support, valid, novel, useful, tag, note FROM ratings WHERE rater=?", (rater,))}
rated = mine(); first_open = next((i for i, it in enumerate(items) if it["item_id"] not in rated), len(items))
if S.get("ctx") != (run, rater): S.ctx = (run, rater); S.pos = first_open
S.pos = max(0, min(S.pos, len(items)))
st.progress(len(rated) / len(items), text=f"{len(rated)} of {len(items)} rated by {rater}")
if S.pos >= len(items):
    st.success("All items rated. Thank you!"); out = pd.read_sql_query("SELECT * FROM ratings WHERE rater=?", con, params=(rater,))
    st.download_button("Download my ratings (CSV)", out.to_csv(index=False), f"ratings_{rater}.csv", "text/csv")
    if st.button("Review from the start"): S.pos = 0; st.rerun()
    st.stop()
it = items[S.pos]; prev = rated.get(it["item_id"])
st.caption(f"Item {S.pos + 1} of {len(items)} | topic: {it['topic']}"); st.subheader(it["title"])
if it["description"]: st.write(it["description"])
st.markdown("**Evidence cited**"); st.dataframe(pd.DataFrame(it["evidence"]), hide_index=True)
st.caption("Judge only what is shown. A gap means 'not addressed by these papers', not 'never done anywhere'.")
OPT = dict(support=["yes", "partly", "no"], valid=["yes", "no", "unsure"], novel=["yes", "no", "unsure"], tag=["method x problem", "context x problem", "conflicting results", "future work", "other"])
pick = lambda k, i: OPT[k].index(prev[i]) if prev else None
with st.form("rate"):
    sup = st.radio("1. Do the cited passages support the claim?", OPT["support"], index=pick("support", 1), horizontal=True, key=f"s{it['item_id']}")
    val = st.radio("2. Is it a real gap within these papers (not already addressed by them)?", OPT["valid"], index=pick("valid", 2), horizontal=True, key=f"v{it['item_id']}")
    nov = st.radio("3. After a quick search, is it new to the wider literature?", OPT["novel"], index=pick("novel", 3), horizontal=True, key=f"n{it['item_id']}")
    use = st.radio("4. Would you consider it as a research direction? (1 = no, 5 = definitely)", [1, 2, 3, 4, 5], index=(prev[4] - 1) if prev else None, horizontal=True, key=f"u{it['item_id']}")
    tag = st.selectbox("5. Type", OPT["tag"], index=pick("tag", 5) if prev else 4, key=f"t{it['item_id']}"); note = st.text_input("Comment (optional)", prev[6] if prev else "", key=f"c{it['item_id']}")
    go = st.form_submit_button("Save and next", type="primary")
if go:
    if None in (sup, val, nov, use): st.warning("Please answer questions 1 to 4.")
    else:
        con.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?,?,?,?,?)", (it["item_id"], rater, sup, val, nov, int(use), tag, note, time.strftime("%Y-%m-%d %H:%M:%S"))); con.commit()
        rated = mine(); S.pos = next((i for i in range(S.pos + 1, len(items)) if items[i]["item_id"] not in rated), min(S.pos + 1, len(items))); st.rerun()
c1, c2 = st.columns(2)
if c1.button("Previous", disabled=S.pos == 0): S.pos -= 1; st.rerun()
if c2.button("Skip"): S.pos += 1; st.rerun()
