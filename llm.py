"""Language-model layer with four providers and one interface:
  gemini  Google Gemini API, free tier (recommended: 1M-token context, native JSON schema, Google account only)
  openai  any OpenAI-compatible service (Mistral free plan, Cerebras, OpenRouter, ...)
  groq    Groq cloud
  ollama  a local Ollama server
Built for real use: tolerant JSON parsing/repair, retries with the server's own wait hints, daily-vs-minute quota handling, typed errors
the UI can explain, automatic model discovery. Nothing here is specific to one paper or one provider outside its own _call function."""
import os, re, json, time

USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "truncated": 0, "retries": 0}
_sleep = time.sleep                                               # patched in tests

class LLMError(Exception): pass
class LLMConfigError(LLMError): pass       # missing / rejected key, unreachable server, blocked location
class LLMModelError(LLMError): pass        # model not found or retired
class LLMRateLimit(LLMError): pass         # daily/long limit: stop, progress is saved, resume later
class LLMTooLarge(LLMError): pass          # single request exceeds the model limits

PROVIDERS = ("gemini", "openai", "groq", "ollama")
def provider():
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p in PROVIDERS: return p
    return "gemini" if os.getenv("GEMINI_API_KEY") else ("groq" if os.getenv("GROQ_API_KEY") else "gemini")
def available():
    return {"gemini": bool(os.getenv("GEMINI_API_KEY")), "groq": bool(os.getenv("GROQ_API_KEY")), "openai": bool(os.getenv("OPENAI_BASE_URL")), "ollama": True}[provider()]
_AUTO = {}
def model():
    p = provider()
    if p == "gemini": return _gemini_model()
    return {"openai": os.getenv("OPENAI_MODEL", "mistral-small-latest"), "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:8b"), "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")}[p]
def label(): return f"{provider()}:{model()}"
def temperature():
    try: return max(0.0, min(2.0, float(os.getenv("LLM_TEMPERATURE", "0"))))              # 0 = repeatable extraction (recommended)
    except ValueError: return 0.0
def chunk_chars():
    """Big-context models read whole sections (fewer requests = friendlier to daily limits); small local models get small chunks."""
    return int(os.getenv("LLM_CHUNK_CHARS", {"gemini": 20000, "openai": 12000, "groq": 3500, "ollama": 2000}[provider()]))
def output_tokens(): return int(os.getenv("LLM_OUTPUT_TOKENS", {"gemini": 8000, "openai": 4000, "groq": 1800, "ollama": 1800}[provider()]))
def ollama_host():
    h = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip().rstrip("/")
    if "://" not in h: h = "http://" + h
    return h.replace("//0.0.0.0", "//localhost")
def _num(s):
    m = re.search(r"([\d.]+)\s*(ms|s)", str(s)); return (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)) if m else None

# ───────── Groq ─────────
def _client():
    from groq import Groq
    kw = dict(api_key=os.environ["GROQ_API_KEY"], max_retries=0, timeout=float(os.getenv("GROQ_TIMEOUT", "90")))   # retries are handled here, with the server's hints
    if os.getenv("GROQ_BASE_URL"): kw["base_url"] = os.environ["GROQ_BASE_URL"]
    return Groq(**kw)
def _info(e):
    body = getattr(e, "body", None)
    if isinstance(body, dict): body = body.get("error", body)
    return getattr(e, "status_code", None), (body if isinstance(body, dict) else {}), str(e)
def _wait_hint(e, msg):
    try:
        h = e.response.headers.get("retry-after")
        if h: return float(h)
    except Exception: pass
    m = re.search(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?\s*(?:([\d.]+)ms)?", msg)
    if m and any(m.groups()): h, mi, s, ms = (float(x) if x else 0.0 for x in m.groups()); return h * 3600 + mi * 60 + s + ms / 1000
    return None
def _groq_call(messages, json_mode, max_tokens, model_name=None):
    c = _client(); m = model_name or model(); use_json = json_mode
    for attempt in range(8):
        try:
            kw = dict(model=m, messages=messages, temperature=temperature(), max_tokens=max_tokens)
            if use_json: kw["response_format"] = {"type": "json_object"}
            r = c.chat.completions.create(**kw); u = getattr(r, "usage", None); USAGE["calls"] += 1
            USAGE["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0; USAGE["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            ch = r.choices[0]; return ch.message.content or "", ch.finish_reason
        except Exception as e:
            code, body, msg = _info(e); name = type(e).__name__; low = msg.lower()
            if code in (401, 403) or name in ("AuthenticationError", "PermissionDeniedError"): raise LLMConfigError("Groq rejected the API key. Check it in Settings (keys start with 'gsk_').")
            if code == 404 or body.get("code") in ("model_not_found", "model_decommissioned") or ("model" in low and ("decommissioned" in low or "does not exist" in low or "not found" in low)):
                raise LLMModelError(f"The model '{m}' is not available on your Groq account (retired or renamed). Open Settings and pick another model.")
            if code == 400 and body.get("code") == "json_validate_failed":
                if body.get("failed_generation"): return body["failed_generation"], "failed_generation"
                use_json = False; continue
            if code == 400 and use_json and "json" in low: use_json = False; continue                 # this model has no JSON mode: ask in the prompt instead
            if code == 413 or "request too large" in low: raise LLMTooLarge("The request is larger than the model's per-minute token limit. Use a smaller chunk size or a model with a higher limit.")
            if code == 429 or name == "RateLimitError":
                w = _wait_hint(e, msg) or min(2 ** attempt, 30)
                if w > 180: raise LLMRateLimit(f"Groq usage limit reached (wait about {int(w // 60)} min). Progress is saved: press Resume later, or switch provider/model in Settings.")
                USAGE["retries"] += 1; _sleep(w + 0.5); continue
            if code in (500, 502, 503, 504) or name in ("APIConnectionError", "APITimeoutError", "InternalServerError"): USAGE["retries"] += 1; _sleep(min(2 ** attempt, 20)); continue
            raise LLMError(f"Groq error {code}: {msg[:300]}")
    raise LLMError("Groq did not answer after several retries (network or service problem).")

# ───────── Google Gemini (REST, no SDK needed) ─────────
def gemini_base(): return os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
def _gh(): return {"x-goog-api-key": os.environ.get("GEMINI_API_KEY", ""), "content-type": "application/json"}
_GEM_THINK = {}                                                   # model -> thinking setting that worked ({} = default)
_THINK_VARIANTS = [None, {"thinkingLevel": "minimal"}, {"thinkingBudget": 0}]

def pick_gemini(ids):
    """Best FREE-tier choice: newest stable Flash-Lite (largest free allowance), then Flash, then Pro. Skips image/audio/live/embedding models."""
    skip = ("image", "tts", "live", "audio", "embedding", "robotics", "computer-use", "imagen", "veo", "native", "learnlm", "aqa", "vision", "customtools", "gemma")
    def key(i):
        fam = 0 if "flash-lite" in i else 1 if "flash" in i else 3 if "pro" in i else 2; ver = tuple(int(x) for x in re.findall(r"\d+", i.split("flash")[0].split("pro")[0])[:2]) or (0,)
        return (fam, any(w in i for w in ("preview", "exp")), tuple(-v for v in ver))
    ok = [i for i in ids if i.startswith("gemini-") and not any(s in i for s in skip)]
    return min(ok, key=key) if ok else None
def _gemini_model():
    m = os.getenv("GEMINI_MODEL")
    if m: return m
    if "gemini" not in _AUTO:
        try: picked = pick_gemini(list_models())
        except Exception: return "gemini-flash-lite-latest"          # do not cache a fallback: the next call (with a fixed key/network) should discover the real model
        if not picked: return "gemini-flash-lite-latest"
        _AUTO["gemini"] = picked
    return _AUTO["gemini"]

def _gemini_err(r):
    try: e = r.json().get("error", {})
    except ValueError: e = {}
    return str(e.get("message", r.text))[:500], e.get("details") or [], e.get("status", "")
def _gemini_raise(r, m):
    """Raise the typed error for non-retryable statuses; return normally for the ones the caller handles (429, 5xx, schema/thinking 400)."""
    msg, details, status = _gemini_err(r); low = msg.lower()
    if r.status_code == 400 and ("api key not valid" in low or "api_key_invalid" in json.dumps(details).lower() or "api key expired" in low): raise LLMConfigError("Google rejected the API key. Check it in Settings (create one at aistudio.google.com/apikey).")
    if "location is not supported" in low: raise LLMConfigError("Google blocks the Gemini API from your current network location. This sometimes happens even in supported countries (VPN, proxy, some networks). Turn off any VPN and retry, or pick another provider in Settings.")
    if r.status_code == 403: raise LLMConfigError("Google denied access (403). Check that the key is valid and allowed for the Gemini API.")
    if r.status_code == 404: raise LLMModelError(f"The model '{m}' was not found for your key. Open Settings and press Test to see the available models.")
    return msg, details, status

def _gemini_call(messages, json_mode, max_tokens, schema=None):
    import requests
    m = _gemini_model(); sysmsg = next((x["content"] for x in messages if x["role"] == "system"), "")
    contents = [{"role": "user" if x["role"] == "user" else "model", "parts": [{"text": x["content"]}]} for x in messages if x["role"] != "system"]
    gen = dict(temperature=temperature(), maxOutputTokens=max_tokens + 2048)          # headroom: thinking tokens count against the output limit
    if json_mode: gen["responseMimeType"] = "application/json"
    schema_on = bool(json_mode and schema); ti = 0; net_fail = 0
    for attempt in range(10):
        g = dict(gen)
        if schema_on: g["responseJsonSchema"] = schema
        tv = _GEM_THINK[m] if m in _GEM_THINK else _THINK_VARIANTS[ti]
        if tv: g["thinkingConfig"] = tv
        body = dict(contents=contents, generationConfig=g)
        if sysmsg: body["systemInstruction"] = {"parts": [{"text": sysmsg}]}
        try: r = requests.post(f"{gemini_base()}/v1beta/models/{m}:generateContent", headers=_gh(), json=body, timeout=(10, float(os.getenv("GEMINI_TIMEOUT", "300"))))
        except (requests.ConnectionError, requests.Timeout):
            net_fail += 1
            if net_fail >= 3: raise LLMError("Cannot reach the Gemini API (network problem). Check your connection and try again.")
            _sleep(2 ** net_fail); continue
        if r.status_code == 200:
            j = r.json(); USAGE["calls"] += 1; u = j.get("usageMetadata") or {}; USAGE["prompt_tokens"] += u.get("promptTokenCount", 0) or 0; USAGE["completion_tokens"] += (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0)
            cand = (j.get("candidates") or [None])[0]
            if not cand: raise LLMError(f"Gemini declined to answer this text ({(j.get('promptFeedback') or {}).get('blockReason', 'no answer')}).")
            text = "".join(p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []) if not p.get("thought")); fin = cand.get("finishReason", "STOP")
            if not text.strip():
                if fin == "MAX_TOKENS" and m not in _GEM_THINK and ti < len(_THINK_VARIANTS) - 1: ti += 1; USAGE["retries"] += 1; continue      # thinking ate the budget: turn it down and retry
                raise LLMError(f"Gemini returned no text (finish reason: {fin}). Try a different model in Settings.")
            if m not in _GEM_THINK and ti: _GEM_THINK[m] = _THINK_VARIANTS[ti]                                                                     # remember what worked
            return text, ("length" if fin == "MAX_TOKENS" else "stop")
        msg, details, status = _gemini_raise(r, m); low = msg.lower()
        if r.status_code == 400 and schema_on and any(w in low for w in ("schema", "response_json", "responsejson", "response_schema")): schema_on = False; continue
        if r.status_code == 400 and "thinking" in low and m not in _GEM_THINK and ti < len(_THINK_VARIANTS) - 1: ti += 1; continue
        if r.status_code == 429:
            blob = (json.dumps(details) + msg).lower(); delay = next((_num(d.get("retryDelay")) for d in details if isinstance(d, dict) and d.get("retryDelay")), None) or _num(msg) or min(2 ** attempt * 2, 30)
            if "perday" in blob or "per day" in blob or delay > 180: raise LLMRateLimit(f"The free daily quota for '{m}' is used up (it resets at midnight Pacific time). Progress is saved: press Resume later, or choose another model/provider in Settings (Flash-Lite models have the largest free allowance).")
            USAGE["retries"] += 1; _sleep(delay + .5); continue
        if r.status_code in (500, 502, 503, 504): USAGE["retries"] += 1; _sleep(min(2 ** attempt, 20)); continue
        raise LLMError(f"Gemini error {r.status_code}: {msg[:300]}")
    raise LLMError("Gemini did not give a usable answer after several attempts.")

# ───────── OpenAI-compatible (Mistral, Cerebras, OpenRouter, ...) ─────────
def _openai_call(messages, json_mode, max_tokens, schema=None):
    import requests
    base = os.environ["OPENAI_BASE_URL"].rstrip("/"); key = os.getenv("OPENAI_API_KEY", ""); m = model(); use_json = json_mode; hdr = {"Authorization": f"Bearer {key}"} if key else {}
    for attempt in range(8):
        payload = dict(model=m, messages=messages, temperature=temperature(), max_tokens=max_tokens)
        if use_json: payload["response_format"] = {"type": "json_object"}
        try: r = requests.post(base + "/chat/completions", json=payload, headers=hdr, timeout=(10, float(os.getenv("OPENAI_TIMEOUT", "180"))))
        except (requests.ConnectionError, requests.Timeout): USAGE["retries"] += 1; _sleep(min(2 ** attempt, 20)); continue
        if r.status_code == 200:
            j = r.json(); USAGE["calls"] += 1; u = j.get("usage") or {}; USAGE["prompt_tokens"] += u.get("prompt_tokens", 0) or 0; USAGE["completion_tokens"] += u.get("completion_tokens", 0) or 0
            ch = (j.get("choices") or [{}])[0]; return _strip_think((ch.get("message") or {}).get("content") or ""), ch.get("finish_reason")
        try: e = r.json().get("error", r.text)
        except ValueError: e = r.text
        msg = e.get("message", "") if isinstance(e, dict) else str(e); low = msg.lower()
        if r.status_code in (401, 403): raise LLMConfigError("The provider rejected the API key. Check the key and the base URL in Settings.")
        if r.status_code == 404 or ("model" in low and ("not found" in low or "does not exist" in low)): raise LLMModelError(f"The model '{m}' is not available from this provider. Choose another in Settings (Test lists the available ones).")
        if r.status_code == 400 and use_json and ("response_format" in low or "json" in low): use_json = False; continue
        if r.status_code == 413: raise LLMTooLarge("The request is too large for this provider/model. Lower the chunk size.")
        if r.status_code == 429:
            try: w = float(r.headers.get("retry-after"))
            except (TypeError, ValueError): w = _num(msg) or min(2 ** attempt * 2, 30)
            if w > 180 or "per day" in low or "daily" in low: raise LLMRateLimit("The provider's usage limit is reached. Progress is saved: press Resume later, or switch provider/model in Settings.")
            USAGE["retries"] += 1; _sleep(w + .5); continue
        if r.status_code in (500, 502, 503, 504): USAGE["retries"] += 1; _sleep(min(2 ** attempt, 20)); continue
        raise LLMError(f"Provider error {r.status_code}: {msg[:300]}")
    raise LLMError("The provider did not answer after several attempts.")

# ───────── Ollama (local) ─────────
def _strip_think(t): return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()      # reasoning models (qwen3, deepseek-r1) add these blocks
def _ollama_call(messages, json_mode, max_tokens, schema=None):
    import requests
    host, m = ollama_host(), model(); fmt = (schema or "json") if json_mode else None
    for attempt in range(4):
        payload = dict(model=m, messages=messages, stream=False, keep_alive="15m", options=dict(temperature=temperature(), num_predict=max_tokens, num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192"))))
        if fmt: payload["format"] = fmt
        try: r = requests.post(host + "/api/chat", json=payload, timeout=(5, float(os.getenv("OLLAMA_TIMEOUT", "600"))))
        except requests.ConnectionError: raise LLMConfigError(f"Cannot reach Ollama at {host}. Start the Ollama app (or run 'ollama serve') and try again.")
        except requests.Timeout:
            if attempt == 0: continue                                   # the first request may include loading the model
            raise LLMError("Ollama took too long to answer. The model may be too large for this computer: pick a smaller model in Settings.")
        if r.status_code == 200:
            j = r.json(); USAGE["calls"] += 1; USAGE["prompt_tokens"] += j.get("prompt_eval_count", 0) or 0; USAGE["completion_tokens"] += j.get("eval_count", 0) or 0
            return _strip_think((j.get("message") or {}).get("content", "")), ("length" if j.get("done_reason") == "length" else "stop")
        try: err = r.json().get("error", r.text)
        except ValueError: err = r.text
        low = str(err).lower()
        if r.status_code == 404 or ("model" in low and "not found" in low): raise LLMModelError(f"The model '{m}' is not installed in Ollama. Run:  ollama pull {m}   (or choose an installed model in Settings).")
        if "memory" in low: raise LLMError(f"Ollama cannot load '{m}': not enough memory. Choose a smaller model in Settings.")
        if isinstance(fmt, dict) and r.status_code in (400, 500) and any(w in low for w in ("format", "schema", "grammar")): fmt = "json"; continue      # older Ollama: no JSON-schema support
        if r.status_code in (500, 502, 503): USAGE["retries"] += 1; _sleep(min(2 ** attempt, 10)); continue
        raise LLMError(f"Ollama error {r.status_code}: {str(err)[:300]}")
    raise LLMError("Ollama did not answer after several attempts.")

def _call(messages, json_mode, max_tokens, model_name=None, schema=None):
    p = provider()
    if p == "gemini": return _gemini_call(messages, json_mode, max_tokens, schema)
    if p == "openai": return _openai_call(messages, json_mode, max_tokens, schema)
    if p == "ollama": return _ollama_call(messages, json_mode, max_tokens, schema)
    return _groq_call(messages, json_mode, max_tokens, model_name)

def parse_json(text):
    t = (text or "").strip(); t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    for cand in (t, (re.search(r"\{.*\}", t, re.S) or [None])[0] if re.search(r"\{.*\}", t, re.S) else None):
        if cand:
            try: obj = json.loads(cand); return obj if isinstance(obj, dict) else {"_value": obj}
            except ValueError: pass
    import json_repair                                             # truncated or slightly malformed output
    obj = json_repair.loads(t)
    if isinstance(obj, dict): return obj
    raise ValueError("model reply is not a JSON object")

def chat_json(system, user, max_tokens=1800, schema=None):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        text, finish = _call(msgs, True, max_tokens, schema=schema)
        if finish == "length": USAGE["truncated"] += 1
        try: return parse_json(text)
        except ValueError:
            msgs = msgs + [{"role": "assistant", "content": text[:2000]}, {"role": "user", "content": "That was not valid JSON. Reply again with ONE JSON object only, no prose, no code fences."}]
    raise LLMError("The model did not return valid JSON twice in a row. Try a larger model in Settings" + (" (small local models often cannot follow the format)." if provider() == "ollama" else "."))

def chat_text(system, user, max_tokens=400): return _call([{"role": "system", "content": system}, {"role": "user", "content": user}], False, max_tokens)[0].strip()

# ───────── model discovery and the Settings "Test" button ─────────
BAD = ("whisper", "guard", "tts", "playai", "orpheus", "embed", "safeguard", "vision-preview")
def _tags():
    import requests
    return requests.get(ollama_host() + "/api/tags", timeout=8).json().get("models", [])
def _size_b(t):
    m = re.search(r"([\d.]+)\s*([BMbm])", str((t.get("details") or {}).get("parameter_size", "")) or t.get("name", ""))
    return (float(m.group(1)) / (1000 if m.group(2) in "Mm" else 1)) if m else 0.0
def list_models():
    import requests
    p = provider()
    if p == "ollama": return sorted(t["name"] for t in _tags())
    if p == "gemini":
        r = requests.get(gemini_base() + "/v1beta/models", headers=_gh(), params={"pageSize": 1000}, timeout=15)
        if r.status_code != 200: _gemini_raise(r, "?"); raise LLMError(f"Could not list Gemini models (HTTP {r.status_code}): {_gemini_err(r)[0][:200]}")
        return sorted(x["name"].split("/", 1)[-1] for x in r.json().get("models", []) if "generateContent" in (x.get("supportedGenerationMethods") or []))
    if p == "openai":
        r = requests.get(os.environ["OPENAI_BASE_URL"].rstrip("/") + "/models", headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}, timeout=15)
        if r.status_code in (401, 403): raise LLMConfigError("The provider rejected the API key. Check the key and the base URL in Settings.")
        return sorted(x["id"] for x in r.json().get("data", []))
    return sorted(m.id for m in _client().models.list().data)
def pick_model(ids, tags=None):
    """Groq: a large instruct model. Ollama: the largest installed chat model. Never speech, guard or embedding models."""
    ok = [i for i in ids if not any(b in i.lower() for b in BAD + ("nomic", "bge-", "minilm"))]
    if tags is not None:
        sizes = {t["name"]: _size_b(t) for t in tags}; ok.sort(key=lambda n: -sizes.get(n, 0)); return ok[0] if ok else None
    if provider() == "gemini": return pick_gemini(ids)
    for pref in ("llama-3.3-70b-versatile", "gpt-oss-120b", "mistral-small", "70b", "gpt-oss-20b", "versatile", "instant", "8b"):
        hit = next((i for i in ok if pref in i.lower()), None)
        if hit: return hit
    return ok[0] if ok else None

PRIVACY = "Free tiers may use your prompts to improve the provider's products (Google does this outside the UK/Switzerland/EEA). Public papers are fine; do not process confidential or unpublished documents on a free tier."
def check():
    """Used by Settings: dict(ok, message, model, models, warning)."""
    p = provider(); out = lambda ok, msg, mdl=None, models=(), warn="": dict(ok=ok, message=msg, model=mdl, models=list(models), warning=warn)
    if not available():
        return out(False, {"gemini": "No Gemini API key set. Create a free one at aistudio.google.com/apikey.", "groq": "No Groq API key set.", "openai": "No base URL set."}.get(p, "Not configured."))
    try:
        if p == "ollama":
            import requests
            try: tags = _tags()
            except requests.RequestException: return out(False, f"Cannot reach Ollama at {ollama_host()}. Start the Ollama app (or run 'ollama serve').")
            names = [t["name"] for t in tags]; m = model()
            if m not in names and m + ":latest" not in names: alt = pick_model(names, tags); return out(False, f"Model '{m}' is not installed. " + (f"Installed: {', '.join(names)}. Suggested: {alt}." if names else f"Run:  ollama pull {m}"), alt, names)
            chat_text("Reply with one word.", "Say OK.", 8); size = next((_size_b(t) for t in tags if t["name"] in (m, m + ":latest")), 0.0); warn = ""
            if 0 < size <= 3.5: warn = f"'{m}' is a small model ({size:g}B). It often cannot follow the extraction format, so expect few facts. For reliable results use a 7B-8B model or larger."
            if temperature() > .3: warn += (" " if warn else "") + f"Temperature {temperature():g} makes extraction less reliable; 0 is recommended."
            return out(True, "Ollama works.", m, names, warn)
        ids = list_models(); explicit = {"gemini": os.getenv("GEMINI_MODEL"), "openai": os.getenv("OPENAI_MODEL"), "groq": os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"}[p]
        if explicit and explicit not in ids and p != "openai": alt = pick_model(ids); return out(False, f"Model '{explicit}' is not available. Suggested: {alt}", alt, ids)
        if explicit and explicit not in ids and p == "openai":                          # some providers omit valid aliases from /models: trust a real test call instead
            try: chat_text("Reply with one word.", "Say OK.", 8)
            except LLMModelError: alt = pick_model(ids); return out(False, f"Model '{explicit}' is not available. Suggested: {alt}", alt, ids)
        chat_text("Reply with one word.", "Say OK.", 8); return out(True, {"gemini": f"Gemini works (model {model()}).", "groq": "Groq works.", "openai": "The provider works."}[p], model(), ids, PRIVACY if p in ("gemini", "openai") else "")
    except LLMError as e: return out(False, str(e))
    except Exception as e: return out(False, f"{type(e).__name__}: {str(e)[:200]}")
