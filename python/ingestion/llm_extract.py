"""
Backends for constrained edge extraction, behind one interface.

  LlamaCppBackend   local GGUF via llama-cpp-python, constrained by GBNF grammar
  HttpBackend       any OpenAI-compatible endpoint, constrained by JSON Schema
  NullBackend       returns nothing, for testing the pipeline without inference

Constrained decoding makes off-vocabulary tokens unreachable, so parse_edges()
defaults to strict: if it starts raising, the constraint is not being applied.
"""

from __future__ import annotations

import os
import time

from .extraction_schema import (ITEM_CODES, SYSTEM_PROMPT, build_prompt,
                                gbnf_grammar, json_schema, parse_edges)


class Backend:
    """Interface: text in, list of validated edge dicts out."""

    def extract(self, text: str, filer: str | None = None,
                items: list[str] | None = None,
                candidates: list[str] | None = None) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullBackend(Backend):
    """Extracts nothing. Lets the runner, sharding and I/O be tested for free."""

    def extract(self, text: str, filer: str | None = None,
                items: list[str] | None = None,
                candidates: list[str] | None = None) -> list[dict]:
        return []


class LlamaCppBackend(Backend):
    """
    Local GGUF through llama-cpp-python with a GBNF grammar.

    n_ctx is deliberately modest. Prompt prefill dominates wall-clock on CPU, so
    a large context window costs memory and buys nothing when the input is a
    truncated filing body of a few hundred tokens.

    Temperature is 0. This is an extraction task with a closed output space;
    sampling only introduces run-to-run variation in a dataset that later
    analysis will treat as fixed.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int | None = None,
                 max_chars: int = 6000, max_edges: int = 12, verbose: bool = False):
        from llama_cpp import Llama, LlamaGrammar

        self.max_chars = max_chars
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or (os.cpu_count() or 4),
            verbose=verbose,
        )
        self.grammar = LlamaGrammar.from_string(gbnf_grammar(max_edges),
                                                verbose=verbose)

    def extract(self, text: str, filer: str | None = None,
                items: list[str] | None = None,
                candidates: list[str] | None = None) -> list[dict]:
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(text, filer, self.max_chars, items, ITEM_CODES, candidates)},
            ],
            grammar=self.grammar,
            temperature=0.0,
            max_tokens=1024,
        )
        return parse_edges(response["choices"][0]["message"]["content"])

    def close(self) -> None:
        self.model = None


class HttpBackend(Backend):
    """
    Any OpenAI-compatible chat endpoint, over plain `requests`.

    Covers llama-server, vLLM and hosted APIs with one code path, and
    deliberately does not use the `openai` package. The cluster this runs on has
    numpy and requests in its module Python and nothing else, and pip-installing
    into a shared environment on a login node is exactly the kind of setup step
    that breaks silently three months later. `requests` is enough: the protocol
    is one POST.

    Retries on transport errors because a 12-hour array task must not die
    because a locally-hosted server was still loading its weights.

    The read timeout is 900s and must stay generous, because per-request latency
    scales with concurrency. With eight in-flight requests against an eight-slot
    server, a filing that needs ~85s alone waits for its share of a saturated
    batch and can legitimately take ten minutes wall clock. Lowering the ceiling
    to 180s on the theory that a slow request is a doomed one was wrong and
    measurable: failures went from 40 in 9,950 filings (0.4%) to 1,587 in 2,500
    (63.5%), all of them requests that would have completed given time.
    """

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8080/v1",
                 api_key: str | None = None, max_chars: int = 6000,
                 max_edges: int = 12, timeout: int = 900, retries: int = 4,
                 max_tokens: int = 1024):
        import requests

        self._requests = requests
        self.model = model
        self.max_chars = max_chars
        self.schema = json_schema(max_edges)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.session = requests.Session()
        key = api_key or os.getenv("OPENAI_API_KEY")
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"

    def extract(self, text: str, filer: str | None = None,
                items: list[str] | None = None,
                candidates: list[str] | None = None) -> list[dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(text, filer, self.max_chars, items, ITEM_CODES, candidates)},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "edges", "strict": True,
                                "schema": self.schema},
            },
        }

        last = None
        for attempt in range(self.retries):
            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)
                # A 4xx is the request being wrong, not the server being busy --
                # a prompt too long for the context window returns 400 on every
                # attempt. Retrying it four times with backoff turns a fast
                # failure into fifteen wasted seconds per filing, which is what
                # made the first pilot look three times slower than it was.
                if 400 <= response.status_code < 500:
                    detail = response.text[:200].replace("\n", " ")
                    raise RuntimeError(
                        f"backend rejected the request ({response.status_code}): "
                        f"{detail}")
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return parse_edges(content)
            except self._requests.RequestException as error:
                last = error
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"backend unreachable after {self.retries} attempts: {last}")

    def wait_until_ready(self, seconds: int = 300) -> bool:
        """Poll until the server answers, so a job can start it and then block."""
        health = self.url.rsplit("/v1/", 1)[0] + "/health"
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                if self.session.get(health, timeout=5).status_code == 200:
                    return True
            except self._requests.RequestException:
                pass
            time.sleep(2)
        return False


def build_backend(kind: str, model: str | None = None, **kwargs) -> Backend:
    """Dispatch by name so the runner takes a single --backend flag."""
    if kind == "null":
        return NullBackend()
    if kind == "llama-cpp":
        if not model:
            raise ValueError("--model is required for the llama-cpp backend")
        return LlamaCppBackend(model, **kwargs)
    if kind in ("http", "server", "openai"):
        if not model:
            raise ValueError(f"--model is required for the {kind} backend")
        return HttpBackend(model, **kwargs)
    raise ValueError(f"unknown backend {kind!r}")
