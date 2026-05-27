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
        "models": ["llama3.2", "llama3.1", "mistral"],
        "default": "llama3.2",
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
