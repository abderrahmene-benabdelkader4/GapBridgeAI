"""Providers (Gemini, OpenAI-compatible, Groq, Ollama) against fake servers that fail the way the real services do, plus end-to-end builds.   python selftest_providers.py"""
import os, sys, json, shutil, tempfile
from pathlib import Path
HERE = Path(__file__).parent; TMP = Path(tempfile.mkdtemp()); os.environ["GAPFINDER_HOME"] = str(TMP / "home")
for k in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "OLLAMA_HOST", "OLLAMA_MODEL", "NEO4J_PASSWORD"): os.environ.pop(k, None)
import testkit, core, llm, builder, pipeline
sl = []; llm._sleep = lambda s: sl.append(round(s, 2)); R = []
def chk(c, m): R.append(bool(c)); print(("PASS " if c else "FAIL ") + m)
def reset(): llm._AUTO.clear(); llm._GEM_THINK.clear(); sl.clear()
FIX = HERE / "fixtures"; smart = lambda msgs: testkit.smart_content({"messages": msgs}, FIX); A = lambda **k: llm.chat_json("sys", "usr", **k); lc = lambda d: {k.lower(): v for k, v in d.items()}
stems = [f.stem for f in sorted((HERE / "samples").glob("*.pdf"))]; full = lambda s: core.parse_pdf(str(HERE / "samples" / f"{s}.pdf"), chunk_chars=20000)["chunks"][0]["text"]

# ═════ Gemini ═════
gm = testkit.FakeGemini(default=smart); os.environ.update(LLM_PROVIDER="gemini", GEMINI_API_KEY="AIza-test", GEMINI_BASE_URL=gm.url); reset()
gm.queue = [gm.ok('{"a": 1}')]; r = A(max_tokens=1800, schema=core.EXTRACT_SCHEMA); q = gm.requests[-1]; g = q["body"]["generationConfig"]
chk(r == {"a": 1} and lc(q["headers"]).get("x-goog-api-key") == "AIza-test" and g["responseMimeType"] == "application/json" and g["responseJsonSchema"]["type"] == "object" and g["maxOutputTokens"] == 1800 + 2048 and g["temperature"] == 0 and q["body"]["systemInstruction"]["parts"][0]["text"] == "sys",
    "G1 request: key header, JSON mime type + JSON schema, output headroom for thinking, temperature 0, system instruction")
chk("gemini-3.5-flash-lite:generateContent" in q["path"], f"G2 no model configured: best FREE model picked from the account's list ({q['path'].split('/')[-1].split(':')[0]})")
gm.queue = [gm.ok('{"gaps": [{"title": "x", "evidence": [{"paper": "p", "passage": "hello wor', "MAX_TOKENS")]; u0 = llm.USAGE["truncated"]; chk("gaps" in A() and llm.USAGE["truncated"] == u0 + 1, "G3 truncated output (MAX_TOKENS) repaired and flagged")
reset(); gm.queue = [gm.empty("MAX_TOKENS"), gm.ok('{"b": 1}')]; r = A(); chk(r == {"b": 1} and gm.requests[-1]["body"]["generationConfig"].get("thinkingConfig") == {"thinkingLevel": "minimal"}, "G4 a thinking model that spent its whole budget thinking: retried with thinking turned down")
n = len(gm.requests); gm.queue = [gm.ok('{"b2": 1}')]; A(); chk(len(gm.requests) - n == 1 and gm.requests[-1]["body"]["generationConfig"].get("thinkingConfig") == {"thinkingLevel": "minimal"}, "G5 the working thinking setting is remembered (no wasted retry next time)")
reset(); gm.queue = [gm.empty("MAX_TOKENS"), gm.err(400, "Thinking level is not supported for this model", "INVALID_ARGUMENT"), gm.ok('{"c": 1}')]; A(); chk(gm.requests[-1]["body"]["generationConfig"].get("thinkingConfig") == {"thinkingBudget": 0}, "G6 models that reject thinkingLevel fall back to thinkingBudget 0")
reset(); gm.queue = [gm.err(400, "Invalid JSON payload received. Unknown name 'responseJsonSchema': Cannot find field.", "INVALID_ARGUMENT"), gm.ok('{"d": 1}')]; n = len(gm.requests); A(schema=core.EXTRACT_SCHEMA)
chk("responseJsonSchema" not in gm.requests[n + 1]["body"]["generationConfig"] and gm.requests[n + 1]["body"]["generationConfig"]["responseMimeType"] == "application/json", "G7 schema rejected -> retried with plain JSON mode")
gm.queue = [gm.err(429, "Resource has been exhausted. Please retry in 7.2s.", "RESOURCE_EXHAUSTED", [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]), gm.ok('{"e": 1}')]; sl.clear(); chk(A() == {"e": 1} and sl == [7.5], f"G8 per-minute limit: waits the server's retryDelay ({sl})")
gm.queue = [gm.err(429, "Quota exceeded", "RESOURCE_EXHAUSTED", [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}])]
try: A(); chk(False, "G9")
except llm.LLMRateLimit as e: chk("midnight Pacific" in str(e) and "Resume" in str(e), "G9 DAILY free quota -> pause with a clear message (no pointless waiting)")
for status, msg, st_, exc, word in [(403, "Permission denied", "PERMISSION_DENIED", llm.LLMConfigError, "403"), (404, "models/x is not found for API version v1beta", "NOT_FOUND", llm.LLMModelError, "Settings"), (400, "User location is not supported for the API use.", "FAILED_PRECONDITION", llm.LLMConfigError, "VPN")]:
    gm.queue = [gm.err(status, msg, st_)]
    try: A(); chk(False, f"G10 {status}")
    except exc as e: chk(word in str(e), f"G10 {status} -> {exc.__name__}: {str(e)[:60]}...")
os.environ["GEMINI_API_KEY"] = "wrong"
try: A(); chk(False, "G11")
except llm.LLMConfigError as e: chk("rejected the API key" in str(e), "G11 invalid key -> LLMConfigError pointing to aistudio.google.com/apikey" if "aistudio" in str(e) else "G11 invalid key")
os.environ["GEMINI_API_KEY"] = "AIza-test"; reset(); gm.queue = [(503, {}, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}), gm.ok('{"f": 1}')]; chk(A() == {"f": 1} and len(sl) == 1, "G12 503 overloaded is retried")
gm.queue = [(200, {}, {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})]
try: A(); chk(False, "G13")
except llm.LLMError as e: chk(not isinstance(e, core.FATAL) and "declined" in str(e), "G13 a safety block on one text is a per-chunk failure, not fatal")
n = len(gm.requests); llm.chat_text("s", "u"); chk("responseMimeType" not in gm.requests[n]["body"]["generationConfig"], "G14 plain-text calls do not force JSON")
reset(); c = llm.check(); chk(c["ok"] and "gemini-3.5-flash-lite" in c["message"] and "improve" in c["warning"], f"G15 Test button: works, shows the model, and warns about free-tier data use")
os.environ["GEMINI_MODEL"] = "gemini-retired"; c = llm.check(); chk(not c["ok"] and c["model"] == "gemini-3.5-flash-lite", "G16 a retired/unknown model is flagged with a suggested replacement"); os.environ.pop("GEMINI_MODEL")
os.environ["GEMINI_API_KEY"] = "wrong"; c = llm.check(); chk(not c["ok"] and "rejected" in c["message"], "G17 Test button with a wrong key explains it"); os.environ["GEMINI_API_KEY"] = "AIza-test"

# ═════ OpenAI-compatible (Mistral, Cerebras, OpenRouter...) via an OpenAI-format fake ═════
fo = testkit.FakeGroq(default=smart, models=("mistral-large-latest", "open-mistral-nemo")); os.environ.update(LLM_PROVIDER="openai", OPENAI_BASE_URL=fo.url + "/v1", OPENAI_API_KEY="k", OPENAI_MODEL="mistral-small-latest"); reset()
fo.queue = [(200, {}, fo.ok('{"o": 1}'))]; n = len(fo.requests); r = A(); chk(r == {"o": 1} and fo.requests[n]["model"] == "mistral-small-latest" and fo.requests[n]["response_format"] == {"type": "json_object"}, "O1 OpenAI-compatible request (model, JSON mode)")
fo.queue = [fo.err(400, "response_format is not supported by this model"), (200, {}, fo.ok('{"o2": 1}'))]; n = len(fo.requests); A(); chk("response_format" not in fo.requests[n + 1], "O2 JSON mode refused -> retried without it")
fo.queue = [(429, {"retry-after": "3"}, {"error": {"message": "Rate limit exceeded"}}), (200, {}, fo.ok('{"o3": 1}'))]; sl.clear(); chk(A() == {"o3": 1} and sl == [3.5], "O3 429 honours retry-after")
for status, exc in [(401, llm.LLMConfigError), (404, llm.LLMModelError)]:
    fo.queue = [fo.err(status, "bad" if status == 401 else "model not found")]
    try: A(); chk(False, f"O4 {status}")
    except exc: chk(True, f"O4 {status} -> {exc.__name__}")
fo.default = "OK"; c = llm.check(); chk(c["ok"] and "improve" in c["warning"], "O5 Test button works even when the provider omits an alias from its model list (a real call decides)")
fo.queue = [fo.err(404, "model not found")]; c = llm.check(); chk(not c["ok"] and c["model"] == "mistral-large-latest", "O6 a really missing model is flagged with a suggestion"); fo.close()

# ═════ Ollama ═════
ol = testkit.FakeOllama(default=smart); os.environ.update(LLM_PROVIDER="ollama", OLLAMA_HOST=ol.url.replace("http://", ""), OLLAMA_MODEL="llama3.2:1b"); reset()
ol.queue = [(200, {"message": {"content": '{"l": 1}'}, "done": True, "done_reason": "stop"})]; r = A(max_tokens=700, schema=core.EXTRACT_SCHEMA); q = ol.requests[-1]
chk(r == {"l": 1} and q["options"]["num_ctx"] == 8192 and q["options"]["temperature"] == 0 and q["options"]["num_predict"] == 700 and isinstance(q["format"], dict), "L1 Ollama request: context window raised from the tiny default, temperature 0, JSON-schema format")
ol.queue = [(404, {"error": "model 'llama3.2:1b' not found, try pulling it first"})]
try: A(); chk(False, "L2")
except llm.LLMModelError as e: chk("ollama pull" in str(e), "L2 model not installed -> tells the user the pull command")
ol.queue = [(400, {"error": "invalid format: unsupported schema"}), (200, {"message": {"content": '{"l2": 1}'}, "done": True})]; n = len(ol.requests); A(schema=core.EXTRACT_SCHEMA); chk(ol.requests[n + 1]["format"] == "json", "L3 older Ollama without schema support -> falls back to format=json")
ol.queue = [(200, {"message": {"content": "<think>let me think</think>OK"}, "done": True})]; chk(llm.chat_text("s", "u") == "OK", "L4 reasoning-model <think> blocks are stripped")
c = llm.check(); chk(c["ok"] and "small model" in c["warning"], "L5 Test button warns that a 1B model is too small for extraction")
os.environ["OLLAMA_MODEL"] = "missing:7b"; c = llm.check(); chk(not c["ok"] and c["model"] == "llama3.1:8b", "L6 missing model -> suggests the largest installed one"); os.environ["OLLAMA_MODEL"] = "llama3.2:1b"
chk(llm.ollama_host() == ol.url, "L7 host without scheme is normalized"); os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"; chk(llm.ollama_host() == "http://localhost:11434", "L8 0.0.0.0 is mapped to localhost"); os.environ["OLLAMA_HOST"] = ol.url
ol.close(); os.environ["OLLAMA_HOST"] = ol.url
try: A(); chk(False, "L9")
except llm.LLMConfigError as e: chk("ollama serve" in str(e), "L9 Ollama not running -> LLMConfigError with the fix")

# ═════ end-to-end builds ═════
def upload_all(project):
    for s in stems: builder.add_upload(project, s + ".pdf", (HERE / "samples" / f"{s}.pdf").read_bytes())
os.environ.update(LLM_PROVIDER="gemini", GEMINI_API_KEY="AIza-test"); reset(); gm.queue = []; n0 = len(gm.requests); upload_all("g1"); builder.start_build("g1", "standard", True); js = builder.wait("g1", 90); ext = gm.requests[n0:]
res = pipeline.load("g1"); gaps = core.detect_all_py(res["facts"]); gc = ext[0]["body"]["generationConfig"]
chk(js["state"] == "done" and len(ext) == 6 and len(res["facts"]) == 73 and len(gaps) == 7, f"E1 Gemini end to end: whole papers sent in {len(ext)} requests (one per paper), 73 facts, the 7 planted gaps")
chk("responseJsonSchema" in gc and gc["maxOutputTokens"] == 8000 + 2048 and "gemini-3.5-flash-lite" in ext[0]["path"], "E2 extraction uses the JSON schema, the big output budget and the auto-picked free model")
reset(); upload_all("g2"); gm.queue = [gm.ok(smart([{"content": "You extract facts"}, {"content": "TEXT:\n" + full(stems[0])}])), gm.ok(smart([{"content": "You extract facts"}, {"content": "TEXT:\n" + full(stems[1])}])),
                                        gm.err(429, "Quota exceeded", "RESOURCE_EXHAUSTED", [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}])]
builder.start_build("g2", "standard", True); js = builder.wait("g2", 60); done = sum(x["state"] == "done" for x in builder.list_papers("g2"))
chk(js["state"] == "paused" and "midnight Pacific" in js["message"] and done == 2, f"E3 daily free quota hit after 2 papers: build PAUSED with the reason, {done} papers saved")
gm.queue = []; n0 = len(gm.requests); builder.start_build("g2", "standard", True); js = builder.wait("g2", 90); chk(js["state"] == "done" and len(gm.requests) - n0 == 4 and len(core.detect_all_py(pipeline.load("g2")["facts"])) == 7, "E4 next day: Resume reads only the 4 remaining papers (nothing repeated) and finds the same 7 gaps")
os.environ["GEMINI_MODEL"] = "gemini-3.1-pro"; n0 = len(gm.requests); builder.start_build("g2", "standard", True); js = builder.wait("g2", 90); chk(js["state"] == "done" and len(gm.requests) - n0 == 6 and "gemini-3.1-pro" in gm.requests[-1]["path"], "E5 switching model re-reads the papers with the new model"); os.environ.pop("GEMINI_MODEL")
ol2 = testkit.FakeOllama(default=smart); os.environ.update(LLM_PROVIDER="ollama", OLLAMA_HOST=ol2.url, OLLAMA_MODEL="llama3.1:8b"); reset(); upload_all("o1"); builder.start_build("o1", "standard", True); js = builder.wait("o1", 90)
chk(js["state"] == "done" and len(core.detect_all_py(pipeline.load("o1")["facts"])) == 7 and all(r["options"]["num_ctx"] == 8192 for r in ol2.requests) and len(ol2.requests) >= 6, f"E6 Ollama end to end with small chunks ({len(ol2.requests)} requests), same 7 gaps"); ol2.close()

# ═════ the Settings page, driven like a user ═════
from streamlit.testing.v1 import AppTest
for k in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "OLLAMA_HOST", "OLLAMA_MODEL"): os.environ.pop(k, None)
os.environ["GAPFINDER_HOME"] = str(TMP / "home_ui"); reset(); at = AppTest.from_file(str(HERE / "app.py"), default_timeout=90).run(); at.sidebar.button(key="create_first").click().run()
chk(any("language model" in w.value for w in at.warning), "U1 first run: tells the researcher to set up a language model (Gemini suggested)")
at.sidebar.radio(key="nav").set_value("Settings").run(); chk(at.selectbox(key="prov").value == "gemini", "U2 Gemini is the default provider")
at.text_input(key="k_gem").set_value("wrong-key"); at.button(key="test_llm").click().run(); chk(any("rejected" in e.value for e in at.error), "U3 wrong key -> a plain explanation")
at.text_input(key="k_gem").set_value("AIza-test"); at.button(key="test_llm").click().run(); chk(any("Gemini works" in s.value for s in at.success) and any("improve" in w.value for w in at.warning), "U4 right key -> 'Gemini works' plus the free-tier privacy notice")
at.button(key="save_settings").click().run(); cur = json.loads((TMP / "home_ui" / "settings.json").read_text()); chk(cur["LLM_PROVIDER"] == "gemini" and cur["GEMINI_API_KEY"] == "AIza-test", "U5 saved in the user's data folder")
at.sidebar.radio(key="nav").set_value("1 · Build corpus").run(); chk(not any("language model" in w.value for w in at.warning) and "gemini" in at.sidebar.caption[-2].value.lower() + " ".join(c.value for c in at.sidebar.caption).lower(), "U6 Build page no longer asks for a model; the sidebar shows which one is active")
print(f"\n{sum(R)}/{len(R)} provider checks passed"); gm.close(); shutil.rmtree(TMP, ignore_errors=True); sys.exit(0 if all(R) else 1)
