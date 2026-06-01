import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, List
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from bson import ObjectId
from db import items_collection
from neural_network import ModelService
from fastapi.responses import FileResponse
import llm_service
import rag
import agent

# Log through uvicorn's logger so our messages match the server's clean format.
logger = logging.getLogger("uvicorn")


def _log_routes(app: FastAPI):
    """Print a tidy table of the registered API routes at startup."""
    routes = [
        (r.path, ",".join(sorted(m for m in r.methods if m not in ("HEAD", "OPTIONS"))))
        for r in app.routes if isinstance(r, APIRoute)
    ]
    routes.sort()
    logger.info("Boring API ready - %d routes registered:", len(routes))
    for path, methods in routes:
        logger.info("    %-6s %s", methods, path)
    logger.info("Pages:  /  |  /chat-page (agent)  |  /predict-page  |  /docs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log_routes(app)
    yield


app = FastAPI(title="Boring API", lifespan=lifespan)
nn_service = ModelService()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Browsers auto-request this; return 204 so it doesn't spam 404s in the log.
    return Response(status_code=204)
# need this so the frontend can talk to the api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pydantic model for when we create/update an item
class ItemCreate(BaseModel):
    name: str
    description: Optional[str] = None

# pydantic model for returning items (includes the id)
class ItemRead(BaseModel):
    id: str
    name: str
    description: Optional[str] = None

class PredictionRequest(BaseModel):
    features: list[float]


# helper to turn a mongo document into our ItemRead format
def item_to_dict(item):
    return {
        "id": str(item["_id"]),
        "name": item["name"],
        "description": item.get("description")
    }

@app.get("/predict-page")
def predict_page():
    return FileResponse("static/predict.html")

@app.get("/chat-page")
def chat_page():
    return FileResponse("static/chat.html")


@app.post("/predict")
def predict(req: PredictionRequest):

    result = nn_service.predict(req.features)

    return result

# --- CRUD routes ---

# get all items
@app.get("/items", response_model=List[ItemRead])
def get_items():
    items = []
    for item in items_collection.find():
        items.append(item_to_dict(item))
    return items

# get one item by id
@app.get("/items/{item_id}", response_model=ItemRead)
def get_item(item_id: str):
    try:
        item = items_collection.find_one({"_id": ObjectId(item_id)})
    except:
        raise HTTPException(status_code=404, detail="Item not found")
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item_to_dict(item)

# create a new item
@app.post("/items", response_model=ItemRead, status_code=201)
def create_item(item: ItemCreate):
    new_item = {"name": item.name, "description": item.description}
    result = items_collection.insert_one(new_item)
    new_item["_id"] = result.inserted_id
    return item_to_dict(new_item)

# update an existing item
@app.put("/items/{item_id}", response_model=ItemRead)
def update_item(item_id: str, item: ItemCreate):
    try:
        oid = ObjectId(item_id)
    except:
        raise HTTPException(status_code=404, detail="Item not found")
    result = items_collection.update_one(
        {"_id": oid},
        {"$set": {"name": item.name, "description": item.description}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    updated = items_collection.find_one({"_id": oid})
    return item_to_dict(updated)

# delete an item
@app.delete("/items/{item_id}")
def delete_item(item_id: str):
    try:
        oid = ObjectId(item_id)
    except:
        raise HTTPException(status_code=404, detail="Item not found")
    result = items_collection.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"detail": "Item deleted"}


# --- LLM routes ---

CHAT_SYSTEM_PROMPT = (
    "You are the in-app assistant for Boring API, a small item-manager and "
    "iris-flower-prediction tool. You help the user manage their items "
    "(create/edit/delete), explain what fields mean, and answer questions "
    "about the flower predictor (sepal/petal measurements, the three iris "
    "classes: setosa, versicolor, virginica). Be concise, friendly, and "
    "answer in 1-3 short paragraphs unless the user asks for more detail. "
    "If a question is unrelated to the app, answer briefly and steer back."
)

ANALYZE_SYSTEM_PROMPT = """You are a data analysis assistant for an item-manager app.
Given an item description, return ONLY a JSON object in this exact shape:
{
  "categories": ["category1", "category2"],
  "tags": ["tag1", "tag2", "tag3"],
  "sentiment": "positive" | "negative" | "neutral",
  "summary": "one sentence summary"
}
Rules:
- 1-3 broad categories (e.g. "electronics", "kitchen", "book", "tool", "clothing").
- 3-6 short lowercase tags (single words or short phrases).
- sentiment reflects the tone of the description text itself.
- summary is one sentence, under 20 words.
- Output JSON only. No prose, no markdown fences."""

ANALYZE_FEW_SHOT = """Example 1:
Input: "Brand new noise-cancelling headphones, sound is crisp and battery lasts 30 hours. Worth every penny."
Output: {"categories": ["electronics", "audio"], "tags": ["headphones", "noise-cancelling", "battery", "review"], "sentiment": "positive", "summary": "Highly positive review of long-battery noise-cancelling headphones."}

Example 2:
Input: "Cheap plastic spatula, melted on the first use. Returning it."
Output: {"categories": ["kitchen", "utensil"], "tags": ["spatula", "plastic", "defective", "return"], "sentiment": "negative", "summary": "Negative review of a spatula that melted on first use."}

Now analyze this:
"""


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []
    provider: Optional[str] = None
    model: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    conversation_history: List[ChatMessage]
    provider: str
    model: str

class AnalyzeRequest(BaseModel):
    content: str
    provider: Optional[str] = None
    model: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    provider: Optional[str] = None
    model: Optional[str] = None
    k: int = 4


class AgentRequest(BaseModel):
    # message starts a task; state/approvals/user_input/field_values resume a paused one.
    message: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    state: Optional[dict] = None
    approvals: Optional[dict] = None
    user_input: Optional[str] = None
    field_values: Optional[dict] = None


# The agent defaults to a local, tool-capable Ollama model, but the frontend
# can override provider/model per request from the existing dropdown.
AGENT_DEFAULT_PROVIDER = os.getenv("AGENT_DEFAULT_PROVIDER", "ollama")
AGENT_DEFAULT_MODEL = os.getenv("AGENT_DEFAULT_MODEL", "llama3.1")


@app.get("/llm/models")
def list_models():
    """Tells the frontend which providers/models are available for the dropdown."""
    return {
        "default_provider": llm_service.DEFAULT_PROVIDER,
        "providers": llm_service.MODEL_CATALOG,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    messages.extend([m.model_dump() for m in request.conversation_history])
    messages.append({"role": "user", "content": request.message})

    try:
        provider = llm_service.get_provider(request.provider)
        model = llm_service.resolve_model(provider.name, request.model)
        reply = provider.chat(messages, model, max_tokens=512, temperature=0.7)
    except llm_service.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM provider failed: {e}")

    updated = request.conversation_history + [
        ChatMessage(role="user", content=request.message),
        ChatMessage(role="assistant", content=reply),
    ]
    return ChatResponse(
        reply=reply,
        conversation_history=updated,
        provider=provider.name,
        model=model,
    )


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")

    user_prompt = ANALYZE_FEW_SHOT + request.content

    try:
        result = llm_service.complete_json(
            system=ANALYZE_SYSTEM_PROMPT,
            user=user_prompt,
            provider=request.provider,
            model=request.model,
            max_tokens=400,
        )
    except llm_service.LLMError as e:
        # 422 = the LLM returned something we can't parse; client can retry.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM provider failed: {e}")

    required = ["categories", "tags", "sentiment", "summary"]
    missing = [f for f in required if f not in result]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"LLM response is missing required fields: {missing}",
        )
    if result["sentiment"] not in ("positive", "negative", "neutral"):
        raise HTTPException(
            status_code=422,
            detail=f"sentiment must be positive|negative|neutral, got: {result['sentiment']}",
        )
    return result


# --- RAG + Agent routes ---

@app.post("/ask")
def ask(request: AskRequest):
    """Embeddings-based RAG over your items: retrieve the most relevant items
    and answer the question grounded in them. Returns {answer, sources}."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return rag.answer(
            request.question,
            provider=request.provider,
            model=request.model,
            k=request.k,
        )
    except llm_service.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RAG failed: {e}")


@app.post("/agent")
def run_agent(request: AgentRequest):
    """Run (or resume) the tool-using agent over the app's API.

    Returns a status-tagged result with a reasoning trace:
      - completed:          {status, result, steps, state, ...}
      - needs_confirmation: {status, pending, steps, state, message}  (destructive tool)
      - needs_input:        {status, question, options, steps, state} (clarification)
    Resume a paused run by POSTing back `state` plus `approvals` or `user_input`.
    """
    provider = request.provider or AGENT_DEFAULT_PROVIDER
    model = request.model or AGENT_DEFAULT_MODEL
    try:
        return agent.run_agent(
            message=request.message,
            state=request.state,
            approvals=request.approvals,
            user_input=request.user_input,
            field_values=request.field_values,
            provider=provider,
            model=model,
        )
    except llm_service.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent failed: {e}")


# serve the frontend html - this has to be AFTER the api routes
# or else it will catch everything
app.mount("/static", StaticFiles(directory="static"), name="static")
