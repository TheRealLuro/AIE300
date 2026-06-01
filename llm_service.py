"""
Tiny provider-agnostic LLM layer.

Supports OpenAI, Anthropic, and a local Ollama server. Each provider exposes
the same two methods used by main.py:
    - chat(messages, model, max_tokens, temperature) -> str
    - complete_json(system, user, model, max_tokens) -> str   (low-temp, JSON only)

The frontend / API caller picks the provider and model per-request. If none is
given we fall back to DEFAULT_LLM_PROVIDER from the environment.
"""

import os
import json
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Catalog of models exposed to the frontend dropdown. Keep this small.
# ---------------------------------------------------------------------------
MODEL_CATALOG = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "default": "gpt-4o-mini",
    },
    "anthropic": {
        "label": "Anthropic",
        "models": ["claude-haiku-4-5", "claude-sonnet-4-5"],
        "default": "claude-haiku-4-5",
    },
    "ollama": {
        "label": "Ollama (local)",
        # llama3.1 is the default: it chains multiple tool calls far more
        # reliably than the smaller llama3.2 (which is fine for single calls).
        "models": ["llama3.1", "llama3.2", "mistral"],
        "default": "llama3.1",
    },
}

DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()


class LLMError(Exception):
    """Raised when a provider call fails for any reason worth surfacing."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class _OpenAIProvider:
    name = "openai"

    def __init__(self):
        from openai import OpenAI  # imported lazily so the dep is optional
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key)

    def chat(self, messages, model, max_tokens, temperature):
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def chat_with_tools(self, messages, model, tools, system=None,
                        max_tokens=1024, temperature=0.3):
        kwargs = dict(model=model, messages=messages,
                      max_tokens=max_tokens, temperature=temperature)
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"text": msg.content, "tool_calls": tool_calls}

    def embed(self, text, model="text-embedding-3-small"):
        resp = self.client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding


class _AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=key)

    def chat(self, messages, model, max_tokens, temperature):
        # Anthropic puts the system prompt in its own field, not in messages.
        system = ""
        cleaned = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                cleaned.append({"role": m["role"], "content": m["content"]})

        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=cleaned,
        )
        return resp.content[0].text

    def chat_with_tools(self, messages, model, tools, system=None,
                        max_tokens=1024, temperature=0.3):
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=messages,
        )
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        resp = self.client.messages.create(**kwargs)
        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name,
                                   "arguments": block.input or {}})
        return {"text": "\n".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls}


class _OllamaProvider:
    name = "ollama"

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def chat(self, messages, model, max_tokens, temperature):
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=120,
            )
        except requests.RequestException as e:
            raise LLMError(f"Could not reach Ollama at {self.base_url}: {e}")
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text}")
        return resp.json()["message"]["content"]

    def chat_with_tools(self, messages, model, tools, system=None,
                        max_tokens=1024, temperature=0.3):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
        except requests.RequestException as e:
            raise LLMError(f"Could not reach Ollama at {self.base_url}: {e}")
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text}")
        msg = resp.json().get("message", {})
        tool_calls = []
        for idx, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            # Ollama does not return tool-call ids; synthesize a stable one.
            tool_calls.append({
                "id": f"call_{idx}_{fn.get('name', 'tool')}",
                "name": fn.get("name"),
                "arguments": args,
            })
        return {"text": msg.get("content") or None, "tool_calls": tool_calls}

    def embed(self, text, model="nomic-embed-text"):
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60,
            )
        except requests.RequestException as e:
            raise LLMError(f"Could not reach Ollama at {self.base_url}: {e}")
        if resp.status_code != 200:
            raise LLMError(
                f"Ollama embeddings error {resp.status_code}: {resp.text}. "
                f"Did you run `ollama pull {model}`?"
            )
        return resp.json().get("embedding", [])


_PROVIDER_CLASSES = {
    "openai": _OpenAIProvider,
    "anthropic": _AnthropicProvider,
    "ollama": _OllamaProvider,
}

# Provider instances are cached so we don't rebuild the SDK client per request.
_provider_cache = {}


def get_provider(name: Optional[str]):
    name = (name or DEFAULT_PROVIDER).lower()
    if name not in _PROVIDER_CLASSES:
        raise LLMError(f"Unknown provider '{name}'. Use one of: {list(_PROVIDER_CLASSES)}")
    if name not in _provider_cache:
        _provider_cache[name] = _PROVIDER_CLASSES[name]()
    return _provider_cache[name]


def resolve_model(provider_name: str, requested: Optional[str]) -> str:
    """Use the requested model if it's in the catalog, else the provider default."""
    cat = MODEL_CATALOG.get(provider_name, {})
    if requested and requested in cat.get("models", []):
        return requested
    return cat.get("default") or requested or ""


# ---------------------------------------------------------------------------
# High-level helpers used by FastAPI
# ---------------------------------------------------------------------------
def chat(messages, provider=None, model=None, max_tokens=512, temperature=0.7) -> str:
    p = get_provider(provider)
    model = resolve_model(p.name, model)
    return p.chat(messages, model, max_tokens, temperature)


def complete_json(system: str, user: str, provider=None, model=None, max_tokens=512) -> dict:
    """
    Ask the model for JSON, parse it, and retry once with a stricter nudge if
    the first response isn't valid JSON. Raises LLMError if both attempts fail.
    """
    p = get_provider(provider)
    model = resolve_model(p.name, model)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = p.chat(messages, model, max_tokens, temperature=0.2)
    parsed = _try_parse_json(raw)
    if parsed is not None:
        return parsed

    # Retry once: tell the model exactly what went wrong.
    messages.append({"role": "assistant", "content": raw})
    messages.append({
        "role": "user",
        "content": "That was not valid JSON. Reply again with ONLY the JSON object, no prose, no markdown fences.",
    })
    raw2 = p.chat(messages, model, max_tokens, temperature=0.0)
    parsed = _try_parse_json(raw2)
    if parsed is None:
        raise LLMError(f"Model did not return valid JSON after retry. Last response: {raw2[:300]}")
    return parsed


def _try_parse_json(text: str):
    if not text:
        return None
    # Models sometimes wrap JSON in ```json ... ``` fences. Strip them.
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # Or trail extra prose — grab the outermost {...} block.
    if not s.startswith("{"):
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Tool calling: provider-neutral schema + per-provider adapters.
#
# A "tool" is the canonical dict {name, description, parameters(json-schema)}.
# The agent (agent.py) keeps a provider-neutral "transcript" of typed events
# and converts it to each provider's message shape via build_messages() right
# before the API call, so the agent loop never branches on provider.
# ---------------------------------------------------------------------------
def to_openai_tools(tools):
    """Canonical tools -> OpenAI / Ollama function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def to_anthropic_tools(tools):
    """Canonical tools -> Anthropic tool format (parameters -> input_schema)."""
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


def build_messages(transcript, provider_name, system):
    """Convert a canonical transcript into a provider's message list.

    Transcript events: {type: user|assistant_text|tool_call|tool_result, ...}.
    OpenAI/Ollama embed the system prompt as a message; Anthropic takes it as a
    separate `system=` argument so it is omitted from the returned messages.
    """
    if provider_name == "anthropic":
        return _build_anthropic_messages(transcript)
    if provider_name == "ollama":
        return _build_ollama_messages(transcript, system)
    return _build_openai_messages(transcript, system)


def _build_openai_messages(transcript, system):
    msgs = [{"role": "system", "content": system}]
    i, n = 0, len(transcript)
    while i < n:
        ev = transcript[i]
        t = ev["type"]
        if t == "user":
            msgs.append({"role": "user", "content": ev["content"]})
            i += 1
        elif t in ("assistant_text", "tool_call"):
            text_parts, tool_calls = [], []
            while i < n and transcript[i]["type"] in ("assistant_text", "tool_call"):
                e = transcript[i]
                if e["type"] == "assistant_text":
                    text_parts.append(e["content"])
                else:
                    tool_calls.append({
                        "id": e["id"],
                        "type": "function",
                        "function": {"name": e["name"],
                                     "arguments": json.dumps(e["arguments"])},
                    })
                i += 1
            m = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                m["tool_calls"] = tool_calls
            msgs.append(m)
        elif t == "tool_result":
            msgs.append({"role": "tool", "tool_call_id": ev["id"],
                         "content": ev["content"]})
            i += 1
        else:
            i += 1
    return msgs


def _build_ollama_messages(transcript, system):
    # Ollama mirrors OpenAI but wants tool-call arguments as an object (not a
    # JSON string), uses `tool_name` on tool results, and ignores call ids.
    msgs = [{"role": "system", "content": system}]
    i, n = 0, len(transcript)
    while i < n:
        ev = transcript[i]
        t = ev["type"]
        if t == "user":
            msgs.append({"role": "user", "content": ev["content"]})
            i += 1
        elif t in ("assistant_text", "tool_call"):
            text_parts, tool_calls = [], []
            while i < n and transcript[i]["type"] in ("assistant_text", "tool_call"):
                e = transcript[i]
                if e["type"] == "assistant_text":
                    text_parts.append(e["content"])
                else:
                    tool_calls.append({
                        "function": {"name": e["name"], "arguments": e["arguments"]},
                    })
                i += 1
            m = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                m["tool_calls"] = tool_calls
            msgs.append(m)
        elif t == "tool_result":
            msgs.append({"role": "tool", "tool_name": ev.get("name", ""),
                         "content": ev["content"]})
            i += 1
        else:
            i += 1
    return msgs


def _build_anthropic_messages(transcript):
    msgs = []
    i, n = 0, len(transcript)
    while i < n:
        ev = transcript[i]
        t = ev["type"]
        if t == "user":
            msgs.append({"role": "user", "content": ev["content"]})
            i += 1
        elif t in ("assistant_text", "tool_call"):
            content = []
            while i < n and transcript[i]["type"] in ("assistant_text", "tool_call"):
                e = transcript[i]
                if e["type"] == "assistant_text":
                    if e["content"]:
                        content.append({"type": "text", "text": e["content"]})
                else:
                    content.append({"type": "tool_use", "id": e["id"],
                                    "name": e["name"], "input": e["arguments"]})
                i += 1
            msgs.append({"role": "assistant", "content": content})
        elif t == "tool_result":
            # Coalesce consecutive tool results into one user message.
            results = []
            while i < n and transcript[i]["type"] == "tool_result":
                e = transcript[i]
                results.append({"type": "tool_result", "tool_use_id": e["id"],
                                "content": e["content"]})
                i += 1
            msgs.append({"role": "user", "content": results})
        else:
            i += 1
    return msgs


def chat_with_tools(messages, tools, provider=None, model=None, system=None,
                    max_tokens=1024, temperature=0.3) -> dict:
    """High-level one-shot tool call. Returns {text, tool_calls}."""
    p = get_provider(provider)
    model = resolve_model(p.name, model)
    return p.chat_with_tools(messages, model, tools, system=system,
                             max_tokens=max_tokens, temperature=temperature)


# Local-first embeddings for the RAG tool. Defaults to Ollama nomic-embed-text
# (no API key, runs on the user's machine); set EMBED_MODEL to override.
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def embed_text(text: str):
    """Return an embedding vector for `text`. Uses the local Ollama embedder
    regardless of which chat provider is selected, so RAG works offline."""
    provider = get_provider("ollama")
    return provider.embed(text, model=EMBED_MODEL)
