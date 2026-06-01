"""
Embeddings-based RAG over the MongoDB items collection.

The "knowledge base" is just your items. We embed each item once (lazily, and
cache the vector back into its Mongo document), embed the incoming question,
rank items by cosine similarity, then ask the LLM to answer using ONLY the
retrieved items. Embeddings come from a local Ollama model (nomic-embed-text by
default) so retrieval works without any cloud API key.

Powers both the standalone POST /ask endpoint and the agent's
`query_knowledge_base` tool.
"""

import numpy as np

import llm_service
from db import items_collection


def _item_text(item) -> str:
    name = item.get("name", "") or ""
    desc = item.get("description", "") or ""
    return f"{name}. {desc}".strip()


def _ensure_embedding(item):
    """Return the item's embedding, computing + persisting it if missing/stale.

    We store the source text alongside the vector (`embed_src`) so edits to an
    item transparently trigger a re-embed on the next retrieval.
    """
    src = _item_text(item)
    if not src:
        return None
    cached = item.get("embedding")
    if cached and item.get("embed_src") == src:
        return cached
    vec = llm_service.embed_text(src)
    items_collection.update_one(
        {"_id": item["_id"]},
        {"$set": {"embedding": vec, "embed_src": src}},
    )
    return vec


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve(query: str, k: int = 4):
    """Top-k items most similar to `query`, each as
    {id, name, description, score}. Embeddings are computed/cached on demand."""
    q_vec = llm_service.embed_text(query)
    scored = []
    for item in items_collection.find():
        vec = _ensure_embedding(item)
        if not vec:
            continue
        scored.append((
            _cosine(q_vec, vec),
            {
                "id": str(item["_id"]),
                "name": item.get("name", ""),
                "description": item.get("description") or "",
            },
        ))
    scored.sort(key=lambda s: s[0], reverse=True)
    out = []
    for score, doc in scored[:k]:
        doc = dict(doc)
        doc["score"] = round(score, 4)
        out.append(doc)
    return out


RAG_SYSTEM_PROMPT = (
    "You answer questions about the user's item collection using ONLY the "
    "retrieved items provided below. Cite the item names you used. If the "
    "answer is not contained in the retrieved items, say you don't have an "
    "item about that rather than guessing. Be concise."
)


def answer(question: str, provider=None, model=None, k: int = 4) -> dict:
    """Retrieve relevant items and produce a grounded answer + sources."""
    sources = retrieve(question, k=k)

    if sources:
        context = "\n".join(
            f"- {s['name']}: {s['description']}" + f" (similarity {s['score']})"
            for s in sources
        )
    else:
        context = "(no items in the collection yet)"

    user_prompt = (
        f"Retrieved items:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the retrieved items above."
    )
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    reply = llm_service.chat(messages, provider=provider, model=model,
                             max_tokens=512, temperature=0.2)
    return {"answer": reply, "sources": sources}
