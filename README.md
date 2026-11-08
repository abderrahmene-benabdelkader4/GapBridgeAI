# GapFinder

Find credible research gaps in the literature you care about, with the evidence attached.
Type a topic (or add your PDFs), let the app read the papers, and review gap candidates that each point to the exact sentences and pages that support them.

## Quick start

1. Install Python 3.12 (3.10 to 3.13 also work). Get a **free Google Gemini key** at aistudio.google.com/apikey (Google account only, no card).
2. **Windows:** double-click `run.bat`. **macOS/Linux:** run `./run.sh`. The first start installs everything (a few minutes) and opens the app in your browser.
3. In the app: **Settings** -> provider *Google Gemini* -> paste your key -> *Test this model* -> *Save*. (Other options: Mistral, Groq, a local Ollama: see below.)
4. **1 · Build corpus** -> type your topic -> *Search* -> *Add selected papers* -> *Build corpus*. (No key yet? Press *Try the demo*.)
5. **2 · Explore gaps** -> read the cards, press **Keep / Later / Dismiss**, add notes.
6. **3 · Ask the literature** for cited answers, **4 · Export** for a report, CSV and BibTeX.

Your projects and settings live in `~/GapFinder` (set `GAPFINDER_HOME` to change). Updating the app never touches them.

## Choosing a language model (the part that costs tokens)
Reading papers needs a language model. The app supports four; pick in **Settings**, press **Test this model**, then **Save**.

| Provider | Best for | Honest notes |
|---|---|---|
| **Google Gemini, free tier (recommended)** | Most people. Google account only, no card, no phone. Very long context: the app sends a **whole paper per request**, so the free daily request limit goes a long way. Native JSON output. | Exact free quotas are not published by Google and vary by model and account (check AI Studio). The app picks the model with the biggest free allowance (Flash-Lite) by itself. When the daily quota runs out the build **pauses and keeps its progress**; quotas reset at midnight Pacific time. |
| Mistral free plan (via "OpenAI-compatible") | The biggest free token allowance (about 1 billion tokens a month, per Mistral's published plan: verify in your console). Base URL `https://api.mistral.ai/v1`, model `mistral-small-latest`. | Needs phone (SMS) verification. |
| Groq | Fast. | Small free daily token limits: they run out quickly on a full corpus. |
| Ollama (local) | Privacy, no limits. | Needs the Ollama app running. Models of 1B to 3B are **too weak** for reliable extraction; use 7B to 8B or larger if your computer can run them. Keep temperature at 0. |

**Privacy:** free tiers of Gemini and Mistral may use your prompts to improve their products (Google does so outside the UK/Switzerland/EEA). Public papers are fine. Do not process confidential or unpublished documents on a free tier: use Ollama for those.

**How the app stretches free limits:** results are cached per paper and per model (switching model re-reads, re-running costs nothing); big-context models read whole sections in one request; "Abstract only" depth costs about one request per paper, useful for a first scan; every paper is saved as it finishes, so you can stop and resume any time.

**Temperature:** keep it at 0. Higher values make the model worse at copying exact sentences, which is what the evidence check requires.

## How to read a gap card
* **Type**: method never tried on a problem, setting never studied with a problem, conflicting results, or future work nobody has done.
* **Evidence**: the sentences (with page and a link to the paper) that make it a candidate. A statement is only kept if its sentence really exists in the paper.
* **Confidence**: how likely the gap is to survive more papers. Low confidence means it may just reflect papers you have not added.
* **Already done elsewhere?** A quick OpenAlex check. It looks at the top 5 results only: a lead, not proof.
* **A gap always means "not found in this corpus", never "nobody has done it".** Verify before you build on it.

## Honest expectations
* Quality depends on the papers and the model. Check the **Extraction quality** panel after each build: it shows how many of the model's statements were backed by a real sentence. If it is low, use a larger model (Settings) or check the PDFs.
* Papers without an open-access PDF are read from their abstract only (marked in the table). Abstract-only corpora give shallower gaps: add the PDFs of the key papers.
* Scanned PDFs have no text layer and are skipped (use OCR, e.g. `ocrmypdf`, first).
* Free accounts have daily limits. When you hit one, the build **pauses and keeps its progress**; press *Resume* later (Gemini quotas reset at midnight Pacific time).
* Gap detection is rule-based and works best on method/dataset/result-style papers (empirical computer science and similar). Other fields will give fewer, weaker gaps.

## Troubleshooting
| Message | What to do |
|---|---|
| "rejected the API key" | Settings -> paste the key again (Gemini keys come from aistudio.google.com/apikey) |
| "Google blocks the Gemini API from your current network location" | Turn off any VPN/proxy and retry. Google lists Algeria as supported, but this error also appears for some networks; otherwise use Mistral or Groq |
| "model ... is not available / not found" | Settings -> *Test this model* shows a suggested model; copy it into Model and Save (for Gemini, leave Model empty for automatic) |
| "free daily quota ... used up" / "usage limit reached" | Progress is saved. Resume tomorrow, or switch model/provider in Settings |
| "Gemini returned no text" | Pick a different model in Settings (some models spend their whole budget thinking) |
| "full text unavailable" on a paper | Publisher blocks the download: add the PDF yourself in section B |
| Very few gaps | Add more papers on the topic, use "Key sections" depth, and clean up names in the Build page |

## Optional: Neo4j
Settings -> *Use Neo4j* -> URI/user/password -> *Test Neo4j* -> Save. The graph is then stored in Neo4j (Cypher queries do the detection) and you can explore it in Neo4j Browser:
`MATCH (n {project:'my-project'})-[r]->(m) RETURN n,r,m LIMIT 100`. Without Neo4j the app uses an equivalent built-in engine.

## For developers and the thesis
* `research_app.py` is the thesis workbench (conditions V/G/H, blinded rating via `annotate_app.py`, withholding experiment). Commands: `python cli.py --help`.
* Tests (all offline): `selftest.py` (16), `selftest_app.py` (26), `selftest_real.py` (37), `selftest_providers.py` (47: Gemini, OpenAI-compatible, Ollama, end-to-end builds, Settings page), `selftest_ui.py` (23), `selftest_cypher.py` (7, needs `falkordblite`, Linux/macOS).

### What the tests prove, and what they do not
Proven: the full pipeline (search -> download -> parse -> extract -> graph -> gaps) through the **real Groq SDK and real HTTP** (and the Gemini, OpenAI-compatible and Ollama protocols) against fake servers that return rate limits, daily quotas, malformed/truncated JSON, models that spend their whole output budget thinking, rejected JSON schemas, bad keys, 'location not supported', retired models, paywall pages, corrupt PDFs; resume after a crash or limit; the Cypher detectors on a **real openCypher engine** (FalkorDB) with the same results as the Python engine; hard PDF layouts (two columns, running headers, hyphenation, rotated stamps); concurrent status polling; the whole UI driven like a user.
**Not verified by the author:** real answers from Gemini/Mistral/Groq/Ollama (extraction quality on real papers), the real free-tier quotas (Google does not publish them and third-party figures disagree), the live OpenAlex service, a real Neo4j server (the Cypher passed on FalkorDB, which is openCypher but not Neo4j: run `python selftest.py --neo4j` once), real publisher PDFs, `sentence-transformers`. Run one real topic with 10 to 15 papers and read the Extraction quality panel before trusting a big run.
