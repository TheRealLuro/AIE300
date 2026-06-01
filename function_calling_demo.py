"""
Part 1 - Function Calling Basics (standalone, no server / DB needed).

Demonstrates the full mechanic end-to-end and prints the trace:

    user message
      -> model decides which tool to call (and with what arguments)
      -> OUR code executes the function
      -> we send the result back
      -> model produces the final answer

Key point: the model only *decides* what to call. WE run the function and feed
the result back. Three tiny tools are defined below (get_weather, calculate,
search_items) so the demo runs without Mongo or any app state.

Usage:
    python function_calling_demo.py                       # uses DEFAULT_LLM_PROVIDER
    python function_calling_demo.py --provider ollama --model llama3.2
    python function_calling_demo.py --provider anthropic --prompt "What's 19*23?"
"""

import argparse
import ast
import json
import operator

import llm_service

# ---------------------------------------------------------------------------
# 1. Tool definitions (the schemas the model sees)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression like '19 * 23 + 4'.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "search_items",
        "description": "Search a small demo catalog of items by keyword.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

_DEMO_CATALOG = [
    {"name": "Python Cookbook", "description": "Recipes for the Python language"},
    {"name": "Wireless Mouse", "description": "Ergonomic, long battery life"},
    {"name": "Espresso Machine", "description": "Makes coffee"},
]

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


# ---------------------------------------------------------------------------
# 2. Tool implementations (OUR code runs these)
# ---------------------------------------------------------------------------
def get_weather(city):
    fake = {"london": "15°C, drizzle", "tokyo": "22°C, clear",
            "new york": "18°C, cloudy"}
    return fake.get((city or "").lower(), f"20°C, sunny (stub for {city})")


def calculate(expression):
    return str(_safe_eval(ast.parse(expression, mode="eval").body))


def search_items(query):
    ql = (query or "").lower()
    hits = [i for i in _DEMO_CATALOG
            if ql in i["name"].lower() or ql in i["description"].lower()]
    if not hits:
        return f"No items found for '{query}'."
    return "; ".join(f"{i['name']} - {i['description']}" for i in hits)


FUNCTIONS = {"get_weather": get_weather, "calculate": calculate,
             "search_items": search_items}

SYSTEM = ("You are a helpful assistant. Use the provided tools when they help "
          "answer the user, then give a short final answer.")


# ---------------------------------------------------------------------------
# 3. The flow
# ---------------------------------------------------------------------------
def run(prompt, provider_name, model, max_steps=5):
    provider = llm_service.get_provider(provider_name)
    model = llm_service.resolve_model(provider.name, model)

    print(f"\n=== provider={provider.name} model={model} ===")
    print(f"USER: {prompt}\n")

    # provider-neutral transcript; llm_service converts it per provider
    transcript = [{"type": "user", "content": prompt}]

    for step in range(max_steps):
        messages = llm_service.build_messages(transcript, provider.name, SYSTEM)
        reply = provider.chat_with_tools(messages, model, TOOLS, system=SYSTEM)
        text, calls = reply.get("text"), reply.get("tool_calls") or []

        if text:
            transcript.append({"type": "assistant_text", "content": text})

        # No tool call -> the model gave its final answer.
        if not calls:
            print(f"MODEL (final answer): {text}\n")
            return text

        # The model asked to call one or more tools.
        for c in calls:
            transcript.append({"type": "tool_call", "id": c["id"],
                               "name": c["name"], "arguments": c["arguments"]})
            print(f"MODEL wants tool -> {c['name']}({json.dumps(c['arguments'])})")
            fn = FUNCTIONS.get(c["name"])
            try:
                result = fn(**c["arguments"]) if fn else f"unknown tool {c['name']}"
            except Exception as e:  # noqa: BLE001
                result = f"error: {e}"
            print(f"  OUR CODE ran it  -> {result}")
            transcript.append({"type": "tool_result", "id": c["id"],
                               "name": c["name"], "content": result})
        print()

    print("Reached step limit without a final answer.\n")
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Part 1 function-calling demo")
    ap.add_argument("--provider", default=None,
                    help="openai | anthropic | ollama (default: env)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default="What's the weather in Tokyo, and what is 19 * 23?")
    args = ap.parse_args()

    try:
        run(args.prompt, args.provider, args.model)
    except llm_service.LLMError as e:
        print(f"LLM error: {e}")
        raise SystemExit(1)
