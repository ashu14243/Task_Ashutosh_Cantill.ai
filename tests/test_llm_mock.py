"""The optional LLM step, exercised against a local fake server (no network, no key)."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd

from spendtool.classify.llm import llm_classify
from spendtool.classify.taxonomy import Taxonomy


class _Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path.endswith("/v1/messages"):
            user = body["messages"][0]["content"]
        else:
            user = body["messages"][1]["content"]
        lines = json.loads(re.search(r"(\[\{.*\}\])", user, re.S).group(1))
        ans = [{"i": o["i"], "c": 1 if "laptop" in o["d"].lower() else 0, "p": 0.9} for o in lines]
        txt = "Here you go:\n" + json.dumps(ans)
        if self.path.endswith("/v1/messages"):
            payload = {"content": [{"type": "text", "text": txt}]}
        else:
            payload = {"choices": [{"message": {"content": txt}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _units():
    return pd.DataFrame({"description": ["Dell laptop", "Cleaning service", ""], "gl_description": ["", "", ""],
                         "supplier_name": ["", "", ""], "extra_text": ["", "", ""]})


def test_openai_compatible(tmp_path, monkeypatch):
    monkeypatch.delenv("SPENDTOOL_DISABLE_LLM", raising=False)
    srv = _serve()
    tax = Taxonomy(leaves=[("A", "Cleaning"), ("A", "Laptops")], max_depth=2)
    c = {"enabled": True, "provider": "openai", "base_url": f"http://127.0.0.1:{srv.server_port}/v1",
         "cache_file": str(tmp_path / "c.json"), "batch_size": 10}
    P, got = llm_classify(_units(), np.array([3.0, 2.0, 1.0]), tax, c)
    assert got.tolist() == [True, True, False]          # blank line is not sent
    assert P[0].argmax() == 1 and P[1].argmax() == 0
    srv.shutdown()
    # second call is served from the cache with the server gone
    P2, got2 = llm_classify(_units(), np.array([3.0, 2.0, 1.0]), tax, c)
    assert got2.tolist() == [True, True, False]


def test_anthropic_format(tmp_path, monkeypatch):
    monkeypatch.delenv("SPENDTOOL_DISABLE_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    srv = _serve()
    tax = Taxonomy(leaves=[("A", "Cleaning"), ("A", "Laptops")], max_depth=2)
    c = {"enabled": "auto", "provider": "anthropic", "base_url": f"http://127.0.0.1:{srv.server_port}",
         "cache_file": str(tmp_path / "c.json")}
    P, got = llm_classify(_units(), np.array([3.0, 2.0, 1.0]), tax, c)
    assert got.tolist() == [True, True, False]
    srv.shutdown()


def test_disabled_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SPENDTOOL_LLM_API_KEY", raising=False)
    tax = Taxonomy(leaves=[("A", "Cleaning"), ("A", "Laptops")], max_depth=2)
    P, got = llm_classify(_units(), np.array([3.0, 2.0, 1.0]), tax, {"enabled": "auto", "provider": "anthropic"})
    assert not got.any()


def test_unreachable_server_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("SPENDTOOL_DISABLE_LLM", raising=False)
    tax = Taxonomy(leaves=[("A", "Cleaning"), ("A", "Laptops")], max_depth=2)
    c = {"enabled": True, "provider": "openai", "base_url": "http://127.0.0.1:9/v1", "timeout": 1,
         "cache_file": str(tmp_path / "c.json")}
    P, got = llm_classify(_units(), np.array([3.0, 2.0, 1.0]), tax, c)
    assert not got.any()
