"""Generates gapfinder.owl (open it in Protégé). Class and relationship names mirror graph.py; canonical entities and synonyms come from aliases.json.
Run:  python build_ontology.py      (also checks that every relationship type used in graph.py exists in the ontology)"""
import json, re
from pathlib import Path
from owlready2 import (get_ontology, Thing, ObjectProperty, DataProperty, AnnotationProperty, FunctionalProperty, AllDisjoint, Or)

HERE = Path(__file__).parent
onto = get_ontology("http://gapfinder.example.org/onto#")
with onto:
    class neo4jRelationshipType(AnnotationProperty): pass     # links an OWL property to the Cypher relationship type
    class altLabel(AnnotationProperty): pass                  # synonyms used by entity normalization
    DOC = {"Paper": "A scientific paper in the project corpus (Neo4j label Paper).",
           "Problem": "A research problem or task a paper addresses, e.g. 'low-resource machine translation'.",
           "Method": "A technique, model or approach, e.g. mBART, LoRA, back-translation.",
           "Dataset": "A dataset or benchmark, e.g. FLORES-200.",
           "Metric": "An evaluation measure, e.g. BLEU, chrF.",
           "Result": "One reported number: a method evaluated on a dataset with a metric, in a context. Kept as its own node so the score never loses its setting.",
           "Limitation": "A limitation explicitly stated by the authors.",
           "FutureWork": "A future-work statement made by the authors.",
           "Context": "The setting studied: a language, population or domain (merges the plan's 'Population / Context')."}
    C = {}
    for n, d in DOC.items():
        C[n] = type(n, (Thing,), {}); C[n].comment = [d]
    AllDisjoint(list(C.values()))
    ENTITY = Or([C["Problem"], C["Method"], C["Dataset"], C["Context"]])

    REL = [("addresses", "ADDRESSES", "Paper", "Problem", "The paper tackles this problem."), ("uses", "USES", "Paper", "Method", "The paper uses/proposes this method (edge property role: proposed|baseline|used)."),
           ("usesDataset", "USES_DATASET", "Paper", "Dataset", "The paper uses this dataset."), ("studies", "STUDIES", "Paper", "Context", "The paper studies this language/population/domain."),
           ("appliedTo", "APPLIED_TO", "Method", "Problem", "A method applied to a problem. In Neo4j each edge carries the source paper, page and passage."),
           ("evaluatedOn", "EVALUATED_ON", "Method", "Dataset", "A method evaluated on a dataset (edge carries paper, page, passage)."),
           ("reports", "REPORTS", "Paper", "Result", "The paper reports this result."), ("ofMethod", "OF_METHOD", "Result", "Method", "The method the result was obtained with."),
           ("onDataset", "ON_DATASET", "Result", "Dataset", "The dataset the result was measured on."), ("measuredBy", "MEASURED_BY", "Result", "Metric", "The metric of the result."),
           ("states", "STATES", "Paper", "Limitation", "The paper states this limitation."), ("suggests", "SUGGESTS", "Paper", "FutureWork", "The paper suggests this future work."),
           ("mentions", "MENTIONS", "FutureWork", None, "Entities named in a future-work statement (a Method, Problem, Context or Dataset).")]
    for name, cy, dom, rng, doc in REL:
        p = type(name, (ObjectProperty,), {}); p.domain = [C[dom]]; p.range = [C[rng]] if rng else [ENTITY]; p.comment = [doc]; p.neo4jRelationshipType = [cy]

    def dprop(name, dom, rng, doc, functional=True):
        p = type(name, (DataProperty, FunctionalProperty) if functional else (DataProperty,), {}); p.domain = [dom]; p.range = [rng]; p.comment = [doc]; return p
    dprop("paperId", C["Paper"], str, "File-stem identifier of the paper (Neo4j property id)."); dprop("title", C["Paper"], str, "Paper title.")
    dprop("name", Or([C["Problem"], C["Method"], C["Dataset"], C["Metric"], C["Context"]]), str, "Canonical name of an entity after normalization.")
    dprop("value", C["Result"], float, "The reported number (e.g. 22.4)."); dprop("resultContext", C["Result"], str, "Context of the result (Neo4j property context), e.g. the language.")
    dprop("text", Or([C["Limitation"], C["FutureWork"]]), str, "Normalized statement text.")
    for n, t, d in [("page", int, "Page of the supporting passage."), ("passage", str, "Verbatim sentence from the paper that supports the fact."), ("role", str, "proposed | baseline | used")]:
        dprop(n, Thing, t, d + " In Neo4j this lives on the RELATIONSHIP (provenance); OWL would need reification to attach it to an edge.", False)

    aliases = json.loads((HERE / "aliases.json").read_text(encoding="utf-8"))
    for typ, ents in aliases.items():
        for canon, syns in ents.items():
            ind = C[typ](re.sub(r"\W+", "_", canon)); ind.label = [canon]; ind.altLabel = syns
onto.save(file=str(HERE / "gapfinder.owl"), format="rdfxml")

# consistency check against the real code
graph_src = (HERE / "graph.py").read_text(encoding="utf-8"); used = set(re.findall(r"\[\w*:([A-Z_]+(?:\|[A-Z_]+)*)", graph_src)) | set(re.findall(r"-\[(?:\w+):([A-Z_]+)", graph_src))
used = {x for u in used for x in u.split("|")}; declared = {r[1] for r in REL}
print("classes:", sorted(C), "\nrelationship types in graph.py not in ontology:", sorted(used - declared) or "none", "| in ontology but unused:", sorted(declared - used) or "none")
