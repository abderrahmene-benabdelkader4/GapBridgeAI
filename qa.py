"""Ask the corpus: hybrid retrieval (excerpts + graph facts), answers only with verified verbatim evidence, otherwise an explicit 'not enough evidence'."""
import llm, conditions as C
ASK_SYSTEM = "You answer questions about a set of scientific papers using ONLY the excerpts provided. Reply with a single JSON object and nothing else."
ASK_USER = ('Question: {q}\nReturn JSON: {{"answerable": true|false, "answer": str, "evidence": [{{"paper": str, "passage": str}}]}}\n'
            'Rules: answer in at most 120 words and cite papers by id in square brackets. Each "passage" must be ONE sentence copied verbatim from the excerpts; "paper" is the id before the "|". '
            'If the excerpts do not contain the answer set answerable=false and answer="". Never use outside knowledge.\n\n{graph_context}EXCERPTS:\n{excerpts}')
ASK_SCHEMA = {"type": "object", "required": ["answerable", "answer", "evidence"], "properties": {"answerable": {"type": "boolean"}, "answer": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "object", "required": ["paper", "passage"], "properties": {"paper": {"type": "string"}, "passage": {"type": "string"}}}}}}
NO = "Not enough evidence in this corpus to answer this. Try adding papers on the topic, or rephrase the question."

def ask(question, corpus, chunks, index, facts, k=12):
    hits = [i for i, _ in index.search(question, k)]
    if not hits: return dict(answered=False, answer=NO, evidence=[], retrieved=0)
    ids = {chunks[i]["paper"] for i in hits}; by_id = {p["id"]: p for p in corpus}
    raw = llm.chat_json(ASK_SYSTEM, ASK_USER.format(q=question, graph_context=C.graph_context(facts, ids, 40) if facts else "", excerpts=C.excerpts(hits, chunks)), max_tokens=max(900, llm.output_tokens() // 4), schema=ASK_SCHEMA)
    ev = C.gate([dict(evidence=raw.get("evidence") or [])], by_id); ev = ev[0]["evidence"] if ev else []
    if not raw.get("answerable") or not ev or not str(raw.get("answer", "")).strip(): return dict(answered=False, answer=NO, evidence=[], retrieved=len(hits))
    return dict(answered=True, answer=str(raw["answer"]).strip(), evidence=ev, retrieved=len(hits))
