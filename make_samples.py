"""Builds 6 SYNTHETIC one-page papers (PDF) + 'ideal LLM extraction' fixtures for offline testing.
Everything here is fictional (names of methods/datasets are real, papers and numbers are invented) and has planted gaps:
  A1 mBART never applied to domain adaptation; LoRA never applied to dialect translation
  A2 Yoruba never studied together with domain adaptation
  B1 two papers disagree on mBART / FLORES-200 / BLEU / Swahili (22.4 vs 25.6)
  B2 three future-work statements that no paper in the corpus addresses"""
import json, pymupdf
from pathlib import Path
HERE = Path(__file__).parent; (HERE / "samples").mkdir(exist_ok=True); (HERE / "fixtures").mkdir(exist_ok=True)
LRMT, DT, DA = "low-resource machine translation", "dialect translation", "domain adaptation"

P = {}  # stem -> dict(title, sections{name: [sentence keys]}, S{key: sentence}, facts)
P["synth_P1_transformer_vs_mbart"] = dict(title="Comparing Transformer and mBART for Low-Resource Machine Translation",
 S=dict(a1="We study low-resource machine translation for Swahili and Yoruba.", a2="We compare a Transformer baseline with mBART fine-tuning on the FLORES-200 dataset.",
        i1="Low-resource machine translation remains difficult because parallel data is scarce.", i2="Large multilingual pretrained models such as mBART may transfer knowledge to languages with little data.",
        m1="The Transformer baseline is trained from scratch on the available parallel data.", m2="mBART is fine-tuned on the same parallel data for each language.",
        r1="On Swahili, mBART reaches 22.4 BLEU on FLORES-200, while the Transformer baseline reaches 17.9 BLEU.", r2="On Yoruba, mBART reaches 14.2 BLEU on FLORES-200, while the Transformer baseline reaches 10.8 BLEU.",
        l1="We report only BLEU, which may not capture morphological errors.", f1="In future work we plan to study dialectal varieties such as Algerian Arabic."),
 sections={"Abstract": ["a1", "a2"], "1 Introduction": ["i1", "i2"], "2 Method": ["m1", "m2"], "3 Results": ["r1", "r2"], "4 Limitations": ["l1"], "5 Future Work": ["f1"]},
 facts=dict(problems=[(LRMT, "a1")], methods=[("Transformer", "baseline", "m1"), ("mBART", "used", "m2")], datasets=[("FLORES-200", "a2")], contexts=[("Swahili", "a1"), ("Yoruba", "a1")],
   applied=[("Transformer", LRMT, "a2"), ("mBART", LRMT, "a2")], evaluated=[("Transformer", "FLORES-200", "r1"), ("mBART", "FLORES-200", "r1")],
   results=[("mBART", "FLORES-200", "BLEU", 22.4, "Swahili", "r1"), ("Transformer", "FLORES-200", "BLEU", 17.9, "Swahili", "r1"),
            ("mBART", "FLORES-200", "BLEU", 14.2, "Yoruba", "r2"), ("Transformer", "FLORES-200", "BLEU", 10.8, "Yoruba", "r2")],
   limitations=[("Only BLEU is reported.", "l1")], future=[("Study dialectal varieties such as Algerian Arabic.", ["Algerian Arabic"], "f1")]))
P["synth_P2_backtranslation_swahili"] = dict(title="Back-Translation Augmentation for Low-Resource Machine Translation",
 S=dict(a1="We address low-resource machine translation for Swahili using back-translation with mBART.", a2="We evaluate on OPUS-100 and FLORES-200 with BLEU and chrF.",
        i1="Back-translation creates synthetic parallel data from monolingual text.", i2="Prior work reports gains for several languages.",
        m1="We generate synthetic Swahili data with a reverse mBART model.", m2="The synthetic data is mixed with the original parallel data for fine-tuning.",
        r1="mBART alone reaches 25.6 BLEU on FLORES-200 for Swahili.", r2="Back-translation reaches 27.0 BLEU on OPUS-100 for Swahili.", r3="Back-translation also reaches 51.3 chrF on OPUS-100 for Swahili.",
        l1="The quality of the synthetic data depends on the reverse translation model.", f1="We plan to investigate back-translation for domain adaptation in specialised text."),
 sections={"Abstract": ["a1", "a2"], "1 Introduction": ["i1", "i2"], "2 Method": ["m1", "m2"], "3 Results": ["r1", "r2", "r3"], "4 Limitations": ["l1"], "5 Future Work": ["f1"]},
 facts=dict(problems=[(LRMT, "a1")], methods=[("back-translation", "proposed", "a1"), ("mBART", "used", "a1")], datasets=[("OPUS-100", "a2"), ("FLORES-200", "a2")], contexts=[("Swahili", "a1")],
   applied=[("back-translation", LRMT, "a1"), ("mBART", LRMT, "a1")], evaluated=[("back-translation", "OPUS-100", "r2"), ("mBART", "FLORES-200", "r1")],
   results=[("mBART", "FLORES-200", "BLEU", 25.6, "Swahili", "r1"), ("back-translation", "OPUS-100", "BLEU", 27.0, "Swahili", "r2"), ("back-translation", "OPUS-100", "chrF", 51.3, "Swahili", "r3")],
   limitations=[("Synthetic data quality depends on the reverse model.", "l1")], future=[("Investigate back-translation for domain adaptation.", ["back-translation", DA], "f1")]))
P["synth_P3_lora_domain_adaptation"] = dict(title="LoRA Adapters for Domain Adaptation in Machine Translation",
 S=dict(a1="We study domain adaptation and low-resource machine translation for Amharic using LoRA adapters.", a2="Experiments use the FLORES-200 dataset.",
        i1="Full fine-tuning of large models is expensive for small teams.", i2="Low-rank adapters update only a small number of parameters.",
        m1="We insert LoRA adapters into a pretrained translation model and train only the adapters.", m2="The adapters transfer to the medical domain with small additional training.",
        r1="LoRA reaches 19.8 BLEU on FLORES-200 for Amharic.", l1="We evaluate only one language pair, Amharic to English.", f1="We plan to extend the study to dialectal data such as Algerian Arabic."),
 sections={"Abstract": ["a1", "a2"], "1 Introduction": ["i1", "i2"], "2 Method": ["m1", "m2"], "3 Results": ["r1"], "4 Limitations": ["l1"], "5 Future Work": ["f1"]},
 facts=dict(problems=[(DA, "a1"), (LRMT, "a1")], methods=[("LoRA", "proposed", "m1")], datasets=[("FLORES-200", "a2")], contexts=[("Amharic", "a1")],
   applied=[("LoRA", DA, "a1"), ("LoRA", LRMT, "a1")], evaluated=[("LoRA", "FLORES-200", "r1")], results=[("LoRA", "FLORES-200", "BLEU", 19.8, "Amharic", "r1")],
   limitations=[("Only one language pair is evaluated.", "l1")], future=[("Extend to dialectal data such as Algerian Arabic.", ["Algerian Arabic", DT], "f1")]))
P["synth_P4_algerian_arabic_dialect"] = dict(title="Translating Algerian Arabic: mBART and Back-Translation for Dialect Translation",
 S=dict(a1="We study dialect translation from Algerian Arabic to English using mBART and back-translation.", a2="We evaluate on the MADAR dataset with chrF.",
        i1="Dialects lack standard orthography and parallel data.", i2="Dialect translation is therefore harder than translation of standard languages.",
        m1="We fine-tune mBART on a small parallel corpus of Algerian Arabic.", m2="We then add back-translated data generated from monolingual dialect text.",
        r1="mBART reaches 38.5 chrF on MADAR for Algerian Arabic.", r2="Back-translation reaches 41.2 chrF on MADAR for Algerian Arabic.",
        l1="The parallel corpus is small, and we evaluate only with chrF.", f1="Future work will apply parameter-efficient methods such as LoRA to dialect translation."),
 sections={"Abstract": ["a1", "a2"], "1 Introduction": ["i1", "i2"], "2 Method": ["m1", "m2"], "3 Results": ["r1", "r2"], "4 Limitations": ["l1"], "5 Future Work": ["f1"]},
 facts=dict(problems=[(DT, "a1")], methods=[("mBART", "used", "m1"), ("back-translation", "used", "m2")], datasets=[("MADAR", "a2")], contexts=[("Algerian Arabic", "a1")],
   applied=[("mBART", DT, "a1"), ("back-translation", DT, "a1")], evaluated=[("mBART", "MADAR", "r1"), ("back-translation", "MADAR", "r2")],
   results=[("mBART", "MADAR", "chrF", 38.5, "Algerian Arabic", "r1"), ("back-translation", "MADAR", "chrF", 41.2, "Algerian Arabic", "r2")],
   limitations=[("Small parallel corpus and chrF-only evaluation.", "l1")], future=[("Apply parameter-efficient methods such as LoRA to dialect translation.", ["LoRA", DT], "f1")]))
P["synth_P5_backtranslation_medical"] = dict(title="Back-Translation for Domain Adaptation in Swahili Medical Translation",
 S=dict(a1="We study domain adaptation for Swahili medical translation using back-translation.", a2="We evaluate on OPUS-100 with BLEU.",
        i1="General-purpose translation systems perform poorly on medical text.", i2="Domain adaptation with synthetic data can reduce this gap.",
        m1="We back-translate Swahili medical monolingual text to create in-domain parallel data.", r1="Back-translation reaches 26.5 BLEU on OPUS-100 for Swahili.",
        l1="We evaluate only Swahili.", f1="We will study domain adaptation for other African languages such as Yoruba."),
 sections={"Abstract": ["a1", "a2"], "1 Introduction": ["i1", "i2"], "2 Method": ["m1"], "3 Results": ["r1"], "4 Limitations": ["l1"], "5 Future Work": ["f1"]},
 facts=dict(problems=[(DA, "a1")], methods=[("back-translation", "proposed", "a1")], datasets=[("OPUS-100", "a2")], contexts=[("Swahili", "a1")],
   applied=[("back-translation", DA, "a1")], evaluated=[("back-translation", "OPUS-100", "r1")], results=[("back-translation", "OPUS-100", "BLEU", 26.5, "Swahili", "r1")],
   limitations=[("Only Swahili is evaluated.", "l1")], future=[("Study domain adaptation for other African languages such as Yoruba.", [DA, "Yoruba"], "f1")]))
P["synth_P6_lora_yoruba"] = dict(title="LoRA Adapters for Yoruba Machine Translation",
 S=dict(a1="We apply LoRA adapters to low-resource machine translation for Yoruba.", a2="We evaluate on FLORES-200 with BLEU.",
        i1="Yoruba is a tonal language with limited parallel data.", m1="We train LoRA adapters on a pretrained multilingual model with Yoruba parallel data.",
        r1="LoRA reaches 15.1 BLEU on FLORES-200 for Yoruba.", l1="We only report BLEU, and only on one test set.", f1="We plan to evaluate adapter-based translation on Hausa."),
 sections={"Abstract": ["a1", "a2"], "1 Introduction": ["i1"], "2 Method": ["m1"], "3 Results": ["r1"], "4 Limitations": ["l1"], "5 Future Work": ["f1"]},
 facts=dict(problems=[(LRMT, "a1")], methods=[("LoRA", "proposed", "a1")], datasets=[("FLORES-200", "a2")], contexts=[("Yoruba", "a1")],
   applied=[("LoRA", LRMT, "a1")], evaluated=[("LoRA", "FLORES-200", "r1")], results=[("LoRA", "FLORES-200", "BLEU", 15.1, "Yoruba", "r1")],
   limitations=[("Only BLEU on one test set.", "l1")], future=[("Evaluate adapter-based translation on Hausa.", ["LoRA", "Hausa"], "f1")]))

def raw_json(p):   # what a good LLM would return, in the extraction schema
    S, f = p["S"], p["facts"]
    return dict(problems=[dict(name=n, passage=S[k]) for n, k in f["problems"]], methods=[dict(name=n, role=r, passage=S[k]) for n, r, k in f["methods"]],
        datasets=[dict(name=n, passage=S[k]) for n, k in f["datasets"]], contexts=[dict(name=n, passage=S[k]) for n, k in f["contexts"]],
        applications=[dict(method=m, problem=q, passage=S[k]) for m, q, k in f["applied"]], evaluations=[dict(method=m, dataset=d, passage=S[k]) for m, d, k in f["evaluated"]],
        results=[dict(method=m, dataset=d, metric=mt, value=v, context=c, passage=S[k]) for m, d, mt, v, c, k in f["results"]],
        limitations=[dict(text=t, passage=S[k]) for t, k in f["limitations"]], future_work=[dict(text=t, mentions=m, passage=S[k]) for t, m, k in f["future"]])

for stem, p in P.items():
    doc = pymupdf.open(); page = doc.new_page(); body = ["SYNTHETIC SAMPLE PAPER - fictional, for pipeline testing only", "", p["title"], ""]
    for sec, keys in p["sections"].items(): body += [sec, " ".join(p["S"][k] for k in keys), ""]
    assert page.insert_textbox(pymupdf.Rect(54, 54, 541, 788), "\n".join(body), fontsize=10, fontname="helv") >= 0
    doc.set_metadata({"title": p["title"]}); doc.save(HERE / "samples" / f"{stem}.pdf"); doc.close()
    (HERE / "fixtures" / f"{stem}.json").write_text(json.dumps(raw_json(p), indent=1))
print(f"wrote {len(P)} PDFs to samples/ and fixtures/")
