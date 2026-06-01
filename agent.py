"""
The agent: a from-scratch (Path A) tool-calling loop built on the raw provider
function-calling added in llm_service.py.

Design
------
- We keep a provider-neutral *transcript* of typed events (user / assistant_text
  / tool_call / tool_result). llm_service.build_messages() converts it to each
  provider's message shape right before every API call, so this file never
  branches on which provider is in use.
- The transcript (plus a small `pending` queue and the running `steps` trace) is
  the entire agent state. It is JSON-serializable and handed back to the
  frontend on every response, which makes pause/resume trivial and stateless on
  the server: to confirm a destructive action or answer a clarifying question,
  the frontend just echoes the state back with a decision.

Guardrails
----------
1. Max iterations  - the main loop is capped at AGENT_MAX_STEPS.
2. Tool confirmation - create_item / delete_item pause the loop and wait for an
   explicit approval before they run.
3. Error handling  - every tool runs inside _safe_execute(); failures become a
   plain-text tool result the model can read and recover from, never a crash.

There is also an `ask_user` tool: when the model is missing a required argument
it asks a structured question, which pauses the loop the same way confirmation
does, and resumes once the user answers.
"""

import os
import re

from bson import ObjectId

import llm_service
import rag
from db import items_collection

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))

# Tools that mutate data -> require explicit user approval before they run.
DESTRUCTIVE = {"create_item", "delete_item"}


# ---------------------------------------------------------------------------
# Tool schemas (canonical, provider-neutral). One definition, adapted per
# provider by llm_service.to_openai_tools / to_anthropic_tools.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "list_items",
        "description": "List every item in the collection and how many there "
                       "are. Use for 'what do I have', 'show/list my items', or "
                       "'how many items do I have'.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "search_items",
        "description": "Search the item collection for items whose name or "
                       "description contains the query keyword. Use this to "
                       "check whether something already exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "keyword to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_item",
        "description": "Create a new item in the collection. This changes data, "
                       "so it must be confirmed by the user first.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "short item name"},
                "description": {"type": "string", "description": "optional details"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "query_knowledge_base",
        "description": "Ask a natural-language question about the existing items "
                       "and get an answer grounded in them (RAG). Use this for "
                       "'what do I have about X' / summarising questions, not for "
                       "exact keyword lookups.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}
            },
            "required": ["question"],
        },
    },
    {
        "name": "predict_flower",
        "description": "Predict the iris species (setosa/versicolor/virginica) "
                       "from four measurements in centimetres.",
        "parameters": {
            "type": "object",
            "properties": {
                "sepal_length": {"type": "number"},
                "sepal_width": {"type": "number"},
                "petal_length": {"type": "number"},
                "petal_width": {"type": "number"},
            },
            "required": ["sepal_length", "sepal_width",
                         "petal_length", "petal_width"],
        },
    },
    {
        "name": "delete_item",
        "description": "Delete an item by its id. Destructive: must be confirmed "
                       "by the user first.",
        "parameters": {
            "type": "object",
            "properties": {"item_id": {"type": "string"}},
            "required": ["item_id"],
        },
    },
    {
        "name": "ask_user",
        "description": "Ask the user a single structured clarifying question when "
                       "you are missing information you need to proceed (e.g. the "
                       "name of an item to create). Prefer this over guessing.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "optional suggested answers",
                },
            },
            "required": ["question"],
        },
    },
]


SYSTEM_PROMPT = (
    "You are the assistant inside Boring API, an item-manager + iris flower "
    "predictor. You hold a normal conversation AND take actions by calling "
    "tools. You remember earlier messages in this conversation.\n\n"
    "Tools:\n"
    "- list_items(): show everything in the collection (and how many).\n"
    "- search_items(query): find items by keyword.\n"
    "- create_item(name, description?): add an item (the app asks the user to "
    "confirm first).\n"
    "- query_knowledge_base(question): grounded answer about existing items "
    "(RAG).\n"
    "- predict_flower(...): classify an iris from 4 measurements.\n"
    "- delete_item(item_id): remove an item by id (the app asks to confirm).\n"
    "- ask_user(question, options?): ask ONE structured question when you need a "
    "detail to run a tool.\n\n"
    "WHEN TO USE TOOLS:\n"
    "- Greetings, thanks, small talk, or general/how-does-this-work questions "
    "need NO tool - just reply in plain language. Calling list_items (or any "
    "tool) to say hi or to acknowledge a message is WRONG.\n"
    "- Use a tool ONLY when the user actually asks about their items (list, "
    "search, count, look up) or asks for an action (create/delete/predict).\n\n"
    "HANDLING RESULTS (be helpful, never robotic):\n"
    "1. If the collection is EMPTY or a search finds NOTHING, say so plainly and "
    "OFFER a next step, e.g. \"You don't have any items yet - would you like me "
    "to add one?\". This offer is normal conversation; do NOT auto-create.\n"
    "2. Only create something WITHOUT being asked when the user's request itself "
    "says to (e.g. 'find X and if none exist, create one'). Then: search_items "
    "first, and if empty, immediately call create_item.\n"
    "3. When the user agrees to add an item but hasn't given a name, call "
    "ask_user to get the name before create_item.\n"
    "4. To delete or change an item the user names (not by id), first "
    "search_items to find its id, then call delete_item with that id.\n"
    "5. NEVER invent or assume required values (measurements, names, ids). To "
    "predict a flower, just call predict_flower; if the user hasn't given the "
    "four measurements, the app will ask them for you - do NOT make any up.\n"
    "6. If a tool result starts with 'Error', don't pretend it worked - briefly "
    "explain the problem and suggest what to do next.\n\n"
    "STYLE:\n"
    "- Trust tool results: if a tool already ran and returned data (a "
    "prediction, a list, a search result), report it directly. Never tell the "
    "user they didn't give you inputs that the tool already used.\n"
    "- To actually DO something you MUST call its tool; never claim you "
    "searched, created, or deleted without calling it.\n"
    "- NEVER write a tool call as text (e.g. predict_flower(...) or a JSON "
    "object with 'name'/'parameters'). Either call the tool for real, or reply "
    "in plain words.\n"
    "- Don't put tool-argument questions in plain text - use ask_user for those. "
    "(Friendly offers and follow-ups are fine.)\n"
    "- Take one step at a time; read each result before the next call.\n"
    "- Finish with a short, friendly plain-text reply (no JSON, no tool syntax), "
    "and offer an obvious next step when it helps.\n\n"
    "Example (greeting - NO tool):\n"
    "  user: 'hi'\n"
    "  -> reply: 'Hi! I can manage your items or predict an iris flower - what "
    "would you like to do?'\n\n"
    "Example (items - use a tool):\n"
    "  user: 'do I have anything about cats?'\n"
    "  -> search_items(query='cats') -> 'No items match cats. You have 0 items.'\n"
    "  -> reply: \"You don't have any items about cats yet. Want me to add one?\"\n"
    "  user: 'yes, call it Cat Care'\n"
    "  -> create_item(name='Cat Care', description='Notes about cats')\n"
    "  -> reply: 'Added \"Cat Care\" for you.'"
)


# ---------------------------------------------------------------------------
# Tool implementations. Each takes (args, ctx) and returns a STRING result.
# ctx carries the selected provider/model so the RAG tool can reuse them.
# ---------------------------------------------------------------------------
_nn_service = None


def _nn():
    global _nn_service
    if _nn_service is None:
        from neural_network import ModelService
        _nn_service = ModelService()
    return _nn_service


def _format_items(items, header):
    lines = [header]
    for i in items[:50]:
        desc = i.get("description") or "(no description)"
        lines.append(f"- id={str(i['_id'])} | {i.get('name', '')}: {desc}")
    return "\n".join(lines)


def _tool_list_items(args, ctx):
    items = list(items_collection.find())
    if not items:
        return "The collection is empty - you have no items yet."
    return _format_items(items, f"You have {len(items)} item(s):")


def _tool_search_items(args, ctx):
    query = (args.get("query") or "").strip()
    all_items = list(items_collection.find())
    if not all_items:
        return "The collection is empty - you have no items yet."
    if not query:
        return _format_items(all_items, f"You have {len(all_items)} item(s):")
    ql = query.lower()
    matched = [
        i for i in all_items
        if ql in (i.get("name", "") or "").lower()
        or ql in (i.get("description", "") or "").lower()
    ]
    if not matched:
        return (f"No items match '{query}'. You have {len(all_items)} item(s) "
                "in total, but none about that.")
    return _format_items(matched, f"Found {len(matched)} item(s) matching '{query}':")


def _tool_create_item(args, ctx):
    name = (args.get("name") or "").strip()
    if not name:
        return "Error: 'name' is required to create an item."
    result = items_collection.insert_one(
        {"name": name, "description": args.get("description")}
    )
    return f"Created item '{name}' with id={str(result.inserted_id)}."


def _tool_query_kb(args, ctx):
    question = (args.get("question") or "").strip()
    if not question:
        return "Error: 'question' is required."
    res = rag.answer(question, provider=ctx.get("provider"),
                     model=ctx.get("model"))
    out = res["answer"]
    if res["sources"]:
        srcs = ", ".join(f"{s['name']} ({s['score']})" for s in res["sources"])
        out += f"\n\n[sources: {srcs}]"
    return out


def _tool_predict_flower(args, ctx):
    feats = [args.get("sepal_length"), args.get("sepal_width"),
             args.get("petal_length"), args.get("petal_width")]
    if any(f is None for f in feats):
        return ("Error: need sepal_length, sepal_width, petal_length and "
                "petal_width (all in cm).")
    res = _nn().predict([float(f) for f in feats])
    return (f"Predicted {res['prediction']} "
            f"(confidence {round(res['confidence'] * 100, 1)}%).")


def _tool_delete_item(args, ctx):
    item_id = (args.get("item_id") or "").strip()
    try:
        oid = ObjectId(item_id)
    except Exception:
        return f"Error: '{item_id}' is not a valid item id."
    if items_collection.delete_one({"_id": oid}).deleted_count == 0:
        return f"No item with id={item_id} found."
    return f"Deleted item id={item_id}."


TOOL_IMPLS = {
    "list_items": _tool_list_items,
    "search_items": _tool_search_items,
    "create_item": _tool_create_item,
    "query_knowledge_base": _tool_query_kb,
    "predict_flower": _tool_predict_flower,
    "delete_item": _tool_delete_item,
}


def _safe_execute(name, args, ctx):
    """Run a tool, turning any failure into a readable string (guardrail #3)."""
    fn = TOOL_IMPLS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'."
    try:
        return fn(args, ctx)
    except Exception as e:  # noqa: BLE001 - the model recovers from words, not stack traces
        return f"Error running {name}: {e}"


# ---------------------------------------------------------------------------
# Required-input gate: never fabricate required tool arguments. If a required
# value is missing - or, for predict_flower, a measurement the user never
# actually typed - we pause and collect it via a form instead of guessing.
# ---------------------------------------------------------------------------
REQUIRED = {
    "search_items": ["query"],
    "query_knowledge_base": ["question"],
    "create_item": ["name"],
    "predict_flower": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
}

_FIELD_LABELS = {
    "query": ("Search keyword", "text"),
    "question": ("Your question", "text"),
    "name": ("Item name", "text"),
    "sepal_length": ("Sepal length (cm)", "number"),
    "sepal_width": ("Sepal width (cm)", "number"),
    "petal_length": ("Petal length (cm)", "number"),
    "petal_width": ("Petal width (cm)", "number"),
}

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _field_spec(key):
    label, ftype = _FIELD_LABELS.get(key, (key, "text"))
    return {"name": key, "label": label, "type": ftype}


def _user_numbers(transcript):
    """Every number the USER typed in this conversation (for grounding checks)."""
    nums = set()
    for ev in transcript:
        if ev.get("type") == "user":
            for tok in _NUM_RE.findall(ev.get("content", "") or ""):
                try:
                    nums.add(round(float(tok), 4))
                except ValueError:
                    pass
    return nums


def _grounded_number(value, user_nums):
    if value is None:
        return False
    try:
        f = round(float(value), 4)
    except (TypeError, ValueError):
        return False
    return any(abs(f - u) < 1e-6 for u in user_nums)


def _required_inputs(name, args, transcript):
    """Missing required fields for a fresh tool call. predict_flower also
    rejects measurements the user never actually provided (anti-fabrication)."""
    args = args or {}
    if name == "predict_flower":
        user_nums = _user_numbers(transcript)
        return [_field_spec(k) for k in REQUIRED[name]
                if not _grounded_number(args.get(k), user_nums)]
    return [_field_spec(k) for k in REQUIRED.get(name, [])
            if not str(args.get(k) or "").strip()]


def _missing_presence(name, args):
    """Required fields still empty after a collect form (no grounding - the
    user just typed these values into the form, so they count as provided)."""
    args = args or {}
    return [_field_spec(k) for k in REQUIRED.get(name, [])
            if not str(args.get(k) or "").strip()]


def _collect_question(tool):
    return {
        "predict_flower": "I need the flower's measurements (in cm):",
        "create_item": "What should the item be called?",
        "search_items": "What should I search for?",
        "query_knowledge_base": "What's your question about your items?",
    }.get(tool, "I need a bit more information:")


def _set_tool_call_args(transcript, call_id, new_args):
    """Point a recorded tool_call at the real (user-supplied) args so the trace
    shows what actually ran, not the model's placeholder."""
    for ev in transcript:
        if ev.get("type") == "tool_call" and ev.get("id") == call_id:
            ev["arguments"] = new_args
            return


def _record_provided(transcript, field_values):
    """After a collect form, fold the values the user typed into their most
    recent message, so the model knows the user supplied them and won't claim
    the inputs were never given."""
    if not field_values:
        return
    parts = ", ".join(
        f"{_FIELD_LABELS.get(k, (k,))[0].lower()}: {v}"
        for k, v in field_values.items()
    )
    for ev in reversed(transcript):
        if ev.get("type") == "user":
            ev["content"] = (ev.get("content", "") +
                             f"\n(I provided - {parts})").strip()
            return
    transcript.append({"type": "user", "content": f"(I provided - {parts})"})


# Pure greetings / acknowledgements that must never trigger a tool. Matched
# exactly (after trimming), so this never swallows a real request.
_SMALLTALK = {
    "hi", "hii", "hiya", "hello", "hey", "heya", "yo", "sup", "hi there",
    "hey there", "hello there", "howdy", "greetings",
    "thanks", "thank you", "thanks!", "thx", "ty", "cheers", "much appreciated",
    "ok", "okay", "k", "kk", "cool", "nice", "great", "awesome", "perfect",
    "good morning", "good afternoon", "good evening", "good night", "gm",
    "how are you", "how's it going", "hows it going", "how are you doing",
    "what's up", "whats up", "wassup", "bye", "goodbye", "see ya", "later",
    "lol", "haha", "nvm", "never mind",
}


def _is_smalltalk(message):
    if not message:
        return False
    return message.strip().lower().strip("!.?,") in _SMALLTALK


# ---------------------------------------------------------------------------
# Loop helpers
# ---------------------------------------------------------------------------
def _drain(calls, transcript, steps, ctx):
    """Execute queued tool calls in order. Returns a pause descriptor the moment
    it hits a tool that needs user input/approval, else None when all ran.
    Pause kinds: 'ask' (model ask_user), 'collect' (missing required inputs),
    'confirm' (destructive action)."""
    while calls:
        call = calls[0]
        name = call["name"]
        if name == "ask_user":
            return {"_pause": "ask", "call": call, "remaining": calls}
        # Gate: never run a tool with missing/fabricated required inputs.
        missing = _required_inputs(name, call.get("arguments", {}), transcript)
        if missing:
            return {"_pause": "collect", "call": call, "fields": missing,
                    "remaining": calls}
        if name in DESTRUCTIVE:
            return {"_pause": "confirm", "call": call, "remaining": calls}
        output = _safe_execute(name, call["arguments"], ctx)
        transcript.append({"type": "tool_result", "id": call["id"],
                           "name": name, "content": output})
        steps.append({"tool": name, "input": call["arguments"], "output": output})
        calls.pop(0)
    return None


def _resolve_paused_call(call, kind, transcript, steps, approvals, user_input, ctx):
    """Apply the user's decision to an 'ask' or 'confirm' pause, append result."""
    name = call["name"]
    if kind == "ask":
        out = (user_input or "").strip() or "(no answer provided)"
    else:  # confirm: destructive tool awaiting approval
        approved = bool(approvals and approvals.get(call["id"]))
        if approved:
            out = _safe_execute(name, call["arguments"], ctx)
        else:
            out = (f"The user DECLINED to run {name}. Do not retry it; "
                   "acknowledge that and ask how they would like to proceed.")
    transcript.append({"type": "tool_result", "id": call["id"],
                       "name": name, "content": out})
    steps.append({"tool": name, "input": call["arguments"], "output": out})


# Weaker local models sometimes emit a tool call as plain TEXT instead of a real
# tool call (e.g. 'predict_flower(...)' or a JSON blob with name/parameters).
# We detect that and nudge the model to either call the tool or speak plainly.
MAX_CORRECTIONS = 2
_CORRECTION_MSG = (
    "Your previous message looked like a tool call written as plain text, which "
    "does nothing. If you still need info from the user, call the ask_user tool. "
    "If you have what you need, call the proper tool for real. Otherwise reply to "
    "the user in plain language only - no tool names, no JSON."
)


def _looks_like_botched_tool(text):
    if not text:
        return False
    t = text.strip()
    low = t.lower()
    if "ask_user(" in low:
        return True
    if t.startswith("{") and '"name"' in t and ('"parameters"' in t or '"arguments"' in t):
        return True
    for name in TOOL_IMPLS:  # e.g. "predict_flower(" / "create_item("
        if f"{name}(" in low:
            return True
    return False


def _state(transcript, pending, steps, pending_meta=None):
    return {"transcript": transcript, "pending": pending, "steps": steps,
            "pending_meta": pending_meta or {}}


def _completed(result, transcript, steps, provider_name, model, limit_reached=False):
    return {
        "status": "completed",
        "result": result,
        "steps": steps,
        "state": _state(transcript, [], steps),
        "provider": provider_name,
        "model": model,
        "limit_reached": limit_reached,
    }


def _pause_response(paused, transcript, steps, provider_name, model):
    call = paused["call"]
    kind = paused["_pause"]
    meta = {"kind": kind}
    resp = {"steps": steps, "provider": provider_name, "model": model}
    if kind == "confirm":
        resp.update({
            "status": "needs_confirmation",
            "pending": {"id": call["id"], "tool": call["name"],
                        "input": call["arguments"]},
            "message": f"The agent wants to run '{call['name']}'. "
                       "Approve to let it continue.",
        })
    elif kind == "collect":
        meta["fields"] = paused["fields"]
        resp.update({
            "status": "needs_input",
            "question": _collect_question(call["name"]),
            "fields": paused["fields"],
            "pending": {"id": call["id"], "tool": call["name"],
                        "input": call["arguments"]},
        })
    else:  # ask (model-driven ask_user)
        args = call.get("arguments", {})
        resp.update({
            "status": "needs_input",
            "question": args.get("question", "Could you clarify?"),
            "options": args.get("options"),
            "pending": {"id": call["id"], "tool": "ask_user", "input": args},
        })
    resp["state"] = _state(transcript, paused["remaining"], steps, meta)
    return resp


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_agent(message=None, state=None, approvals=None, user_input=None,
              provider=None, model=None, max_steps=None, field_values=None):
    """Run (or resume) the agent. Returns one of three status-tagged dicts:
    completed / needs_confirmation / needs_input."""
    max_steps = max_steps or MAX_STEPS
    state = state or {}
    transcript = list(state.get("transcript", []))
    pending = list(state.get("pending", []))
    steps = list(state.get("steps", []))
    pending_meta = state.get("pending_meta") or {}

    provider_obj = llm_service.get_provider(provider)
    model = llm_service.resolve_model(provider_obj.name, model)
    ctx = {"provider": provider_obj.name, "model": model}

    if pending:
        call = pending[0]
        rest = pending[1:]
        kind = pending_meta.get("kind") or (
            "ask" if call["name"] == "ask_user"
            else "confirm" if call["name"] in DESTRUCTIVE else "collect")

        if kind == "collect":
            # Merge the form answers into the call, point the trace at the real
            # values, then re-gate: re-ask if still empty, confirm if destructive,
            # otherwise run it.
            merged = dict(call.get("arguments") or {})
            for k, v in (field_values or {}).items():
                if v is not None and str(v).strip() != "":
                    merged[k] = v
            call["arguments"] = merged
            _set_tool_call_args(transcript, call["id"], merged)
            _record_provided(transcript, field_values)

            still = _missing_presence(call["name"], merged)
            if still:
                return _pause_response(
                    {"_pause": "collect", "call": call, "fields": still,
                     "remaining": pending}, transcript, steps,
                    provider_obj.name, model)
            if call["name"] in DESTRUCTIVE:
                return _pause_response(
                    {"_pause": "confirm", "call": call, "remaining": pending},
                    transcript, steps, provider_obj.name, model)
            out = _safe_execute(call["name"], merged, ctx)
            transcript.append({"type": "tool_result", "id": call["id"],
                               "name": call["name"], "content": out})
            steps.append({"tool": call["name"], "input": merged, "output": out})
            paused = _drain(rest, transcript, steps, ctx)
            if paused:
                return _pause_response(paused, transcript, steps,
                                       provider_obj.name, model)
        else:
            # 'ask' or 'confirm': settle the call, then drain the rest of its batch.
            _resolve_paused_call(call, kind, transcript, steps,
                                 approvals, user_input, ctx)
            paused = _drain(rest, transcript, steps, ctx)
            if paused:
                return _pause_response(paused, transcript, steps,
                                       provider_obj.name, model)
    elif message:
        transcript.append({"type": "user", "content": message})
        # Deterministic guard: a plain greeting/acknowledgement never needs a
        # tool. Answer it directly (no tools available) so it can't call one.
        if _is_smalltalk(message):
            messages = llm_service.build_messages(transcript, provider_obj.name,
                                                  SYSTEM_PROMPT)
            reply = provider_obj.chat_with_tools(messages, model, [],
                                                 system=SYSTEM_PROMPT)
            text = (reply.get("text") or "").strip() or "Hi! How can I help?"
            transcript.append({"type": "assistant_text", "content": text})
            return _completed(text, transcript, steps, provider_obj.name, model)

    corrections = 0
    for _ in range(max_steps):
        messages = llm_service.build_messages(transcript, provider_obj.name,
                                              SYSTEM_PROMPT)
        reply = provider_obj.chat_with_tools(messages, model, TOOL_SCHEMAS,
                                             system=SYSTEM_PROMPT)
        text = reply.get("text")
        calls = reply.get("tool_calls") or []
        if text:
            transcript.append({"type": "assistant_text", "content": text})
        if not calls:
            # Recover if the model wrote a tool call as text instead of calling it.
            if corrections < MAX_CORRECTIONS and _looks_like_botched_tool(text):
                corrections += 1
                transcript.append({"type": "user", "content": _CORRECTION_MSG})
                continue
            return _completed(text or "(no answer)", transcript, steps,
                              provider_obj.name, model)
        for c in calls:
            transcript.append({"type": "tool_call", "id": c["id"],
                               "name": c["name"], "arguments": c["arguments"]})
        paused = _drain(list(calls), transcript, steps, ctx)
        if paused:
            return _pause_response(paused, transcript, steps,
                                   provider_obj.name, model)

    # Guardrail #1: ran out of steps without a final answer.
    return _completed(
        "I reached my step limit before fully finishing. The trace above shows "
        "what I managed to do.",
        transcript, steps, provider_obj.name, model, limit_reached=True)
