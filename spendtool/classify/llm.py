"""Optional LLM classification of the highest-spend distinct lines.

Off unless a key/endpoint is available. Providers:
  anthropic : needs ANTHROPIC_API_KEY (or classification.llm.api_key_env)
  openai    : any OpenAI-compatible endpoint (OpenAI, Groq, Together, a local Ollama/vLLM server).
              Set base_url (default https://api.openai.com/v1) and OPENAI_API_KEY (not needed for Ollama).
Results are cached on disk so re-runs cost nothing. Any failure is logged and the pipeline continues
with the local classifiers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}
KEY_ENVS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def _enabled(c: dict) -> tuple[bool, str | None]:
    en = c.get("enabled", "auto")
    prov = c.get("provider", "anthropic")
    key = os.environ.get(c.get("api_key_env") or KEY_ENVS.get(prov, "")) or os.environ.get("SPENDTOOL_LLM_API_KEY")
    if os.environ.get("SPENDTOOL_DISABLE_LLM"):
        return False, None
    if en is True or str(en).lower() == "true":
        return True, key
    if str(en).lower() == "auto":
        return bool(key) or bool(c.get("base_url") and "localhost" in str(c.get("base_url"))), key
    return False, None


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _call(c: dict, key: str | None, system: str, user: str) -> str:
    prov = c.get("provider", "anthropic")
    model = c.get("model") or DEFAULT_MODELS.get(prov)
    timeout = float(c.get("timeout", 60))
    if prov == "anthropic":
        url = (c.get("base_url") or "https://api.anthropic.com") + "/v1/messages"
        r = _post(url, {"x-api-key": key or "", "anthropic-version": "2023-06-01"},
                  {"model": model, "max_tokens": 4000, "temperature": 0, "system": system,
                   "messages": [{"role": "user", "content": user}]}, timeout)
        return "".join(b.get("text", "") for b in r.get("content", []))
    url = (c.get("base_url") or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    hdr = {"Authorization": f"Bearer {key}"} if key else {}
    r = _post(url, hdr, {"model": model, "temperature": 0,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}, timeout)
    return r["choices"][0]["message"]["content"]


SYSTEM = ("You are a procurement spend analyst. You assign purchase invoice lines to the single best "
          "category from a company's category tree. Lines may be abbreviated, misspelt or in any language. "
          "Reply with JSON only.")


def _prompt(tree_txt: str, batch: list[dict]) -> str:
    return ("Category tree (leaf id: path):\n" + tree_txt + "\n\nLines (i = line number, d = description, "
            "g = GL account text, s = supplier, x = other attributes):\n" + json.dumps(batch, ensure_ascii=False) +
            "\n\nReturn a JSON array with one object per line: {\"i\": line number, \"c\": leaf id, "
            "\"p\": confidence between 0 and 1}. Always choose the closest leaf; use a low p when unsure.")


def _parse(text: str, n_leaves: int, ids: set) -> dict:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for o in arr:
        try:
            i, c, p = int(o["i"]), int(o["c"]), float(o.get("p", 0.7))
        except (KeyError, TypeError, ValueError):
            continue
        if i in ids and 0 <= c < n_leaves:
            out[i] = (c, min(max(p, 0.0), 1.0))
    return out


def llm_classify(units, unit_spend: np.ndarray, tax, c: dict):
    n, L = len(units), tax.n
    P = np.zeros((n, L))
    got = np.zeros(n, dtype=bool)
    on, key = _enabled(c)
    if not on:
        return P, got
    tree_txt = "\n".join(f"{i}: {' > '.join(lf)}" for i, lf in enumerate(tax.leaves))
    tree_hash = hashlib.md5(tree_txt.encode()).hexdigest()[:12]
    cache_path = Path(c.get("cache_file") or ".cache/llm_cache.json")
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    order = np.argsort(-unit_spend)[: int(c.get("max_unique", 4000))]
    todo, keys = [], {}
    for u in order:
        r = units.iloc[u]
        if not (r["description"] or r["gl_description"] or r["extra_text"]):
            continue
        k = tree_hash + "|" + "|".join(str(r[f]) for f in ["description", "gl_description", "supplier_name", "extra_text"])
        keys[u] = k
        if k not in cache:
            todo.append(u)
    bs = int(c.get("batch_size", 40))
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    log.info("LLM: %d distinct lines selected, %d cached, %d batches to send", len(keys), len(keys) - len(todo), len(batches))

    import threading
    dead = threading.Event()
    state = {"fail_streak": 0, "ok": 0}
    lock = threading.Lock()
    t0 = time.time()
    budget = float(c.get("max_seconds", 900))

    def run(batch):
        if dead.is_set() or time.time() - t0 > budget:
            return {}
        items = [{"i": j, "d": units.iloc[u]["description"][:200], "g": units.iloc[u]["gl_description"][:80],
                  "s": units.iloc[u]["supplier_name"][:60], "x": units.iloc[u]["extra_text"][:80]} for j, u in enumerate(batch)]
        for attempt in range(3):
            if dead.is_set():
                return {}
            try:
                res = _parse(_call(c, key, SYSTEM, _prompt(tree_txt, items)), L, set(range(len(batch))))
                with lock:
                    state["fail_streak"] = 0
                    state["ok"] += 1
                return {batch[j]: v for j, v in res.items()}
            except Exception as e:
                if attempt == 2:
                    log.warning("LLM batch failed: %s", e)
                time.sleep(2 * (attempt + 1))
        with lock:
            state["fail_streak"] += 1
            # circuit breaker: the endpoint is down or the key is wrong - stop trying
            if state["fail_streak"] >= 3 and state["ok"] == 0:
                dead.set()
                log.warning("LLM: endpoint failing, skipping remaining batches (local classifiers continue)")
        return {}

    with ThreadPoolExecutor(max_workers=int(c.get("workers", 4))) as ex:
        for res in ex.map(run, batches):
            for u, v in res.items():
                cache[keys[u]] = v
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    except OSError:
        pass
    for u, k in keys.items():
        if k in cache:
            leaf, p = cache[k]
            P[u, int(leaf)] = 1.0
            # spread the remaining mass uniformly so the LLM's confidence is respected
            P[u] = p * P[u] + (1 - p) / L
            got[u] = True
    log.info("LLM: classified %d distinct lines in %.0fs", int(got.sum()), time.time() - t0)
    return P, got
