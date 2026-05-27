# Boring API

> Full-stack app for AIE300. FastAPI + MongoDB + PyTorch + LLM chat, served from one container.

The app does three things:

1. **Item manager** - simple CRUD over a MongoDB collection.
2. **Flower predictor** - a PyTorch classifier trained on the iris dataset.
3. **AI assistant** - chat + structured-data extraction backed by your choice of OpenAI, Anthropic, or a local Ollama server.

All three share the same FastAPI app and the same plain HTML/JS frontend.

---

## Quick start

1. Install Docker Desktop and make sure it's running.
2. Copy the env template and fill in the providers you plan to use:

   ```
   cp .env.example .env
   ```

   At minimum, set one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (or leave both blank if you only want Ollama). Pick a default in `DEFAULT_LLM_PROVIDER`.

3. Start everything:

   ```
   docker-compose up --build
   ```

4. Open the frontend:

   | Page | URL |
   |------|-----|
   | Items   | http://localhost:9878 |
   | AI Chat | http://localhost:9878/chat-page |
   | Flower Predictor | http://localhost:9878/predict-page |
   | Swagger / API docs | http://localhost:9878/docs |

   (The container listens on internal port 8000; `docker-compose.yml` publishes it on host port 9878.)

To stop: `docker-compose down`. Mongo data persists in the `mongodata` named volume so items survive restarts.

---

## Architecture

```
Browser (localhost:9878)
    |
    | fetch()
    v
FastAPI (web container)
    |-- /items       --> pymongo --> MongoDB (db container, volume: mongodata)
    |-- /predict     --> PyTorch model (loaded from model.pth + scaler.pkl)
    |-- /chat        --> llm_service --> OpenAI / Anthropic / Ollama
    |-- /analyze     --> llm_service (JSON-only, low temperature, with retry)
    |-- /llm/models  --> provider + model catalog for the frontend dropdown
    |-- /static/*    --> index.html, chat.html, predict.html
```

Provider selection happens per request - the chat UI has a dropdown, and `/chat` / `/analyze` accept optional `provider` and `model` fields.

---

## Tech stack

- Python 3.11 / FastAPI / Uvicorn
- MongoDB 7 + pymongo
- PyTorch + scikit-learn (StandardScaler, iris dataset)
- OpenAI, Anthropic, Ollama SDKs
- Plain HTML / CSS / JS (no framework)
- Docker + docker-compose

---

## Part 1 - Item manager

I used **MongoDB** because it's the database I've used the most across coursework and personal projects, and `pymongo` plays nicely with FastAPI. Item IDs are Mongo `ObjectId` hex strings, not integers.

### Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET    | `/items`          | List every item. |
| GET    | `/items/{id}`     | Get one item (404 if not found). |
| POST   | `/items`          | Create an item (returns 201). |
| PUT    | `/items/{id}`     | Update an item. |
| DELETE | `/items/{id}`     | Delete an item. |

Request body for POST/PUT:

```json
{ "name": "something", "description": "optional" }
```

---

## Part 2 - Flower predictor

A small PyTorch classifier trained on the sklearn iris dataset. It predicts one of three classes - **setosa**, **versicolor**, **virginica** - from four float inputs (sepal length/width, petal length/width).

The model:

- A `nn.Sequential` MLP: `Linear(4,32) -> ReLU -> Linear(32,16) -> ReLU -> Linear(16,3)`.
- `CrossEntropyLoss` + `Adam(lr=0.01)` + `DataLoader` batching.
- `StandardScaler` for feature normalization (scaler is saved alongside the model).
- Trained for 200 epochs on app startup if `model.pth` / `scaler.pkl` don't already exist; otherwise loaded from disk.
- Inference uses `model.eval()` + `torch.no_grad()`.

### Endpoint

| Method | Path | What it does |
|--------|------|--------------|
| POST | `/predict` | Predicts flower class from 4 floats. |

Request / response:

```json
// request
{ "features": [5.1, 3.5, 1.4, 0.2] }

// response
{
  "prediction": "setosa",
  "confidence": 0.9987,
  "probabilities": { "setosa": 0.9987, "versicolor": 0.0011, "virginica": 0.0002 }
}
```

There's also a `pytorch_basics.py` script that demonstrates tensor creation, tensor math, matrix multiplication, and autograd - separate from the trained model, kept for reference.

---

## Part 3 - AI assistant (chat + analyze)

Two LLM endpoints plus a chat UI. **Provider and model are selectable per request** - OpenAI, Anthropic, or a local Ollama server - so you can A/B different models from the dropdown without restarting anything.

### Setup

`.env.example` lists every supported variable:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_PROVIDER=anthropic   # one of: openai, anthropic, ollama
MONGO_URL=mongodb://localhost:27017
```

`.env` is in `.gitignore`, so real keys never get committed. `docker-compose.yml` forwards the variables into the `web` container at runtime - they are never baked into the image.

If you change `.env`, restart the stack so compose picks up the new values:

```
docker-compose down
docker-compose up --build
```

### Sanity check (no server needed)

```
python test_llm.py                                                # uses DEFAULT_LLM_PROVIDER
python test_llm.py --provider openai
python test_llm.py --provider anthropic --model claude-haiku-4-5
python test_llm.py --provider ollama   --model llama3.2 --prompt "say hi"
```

It prints the provider, model, and the reply. Exits non-zero if the provider is misconfigured.

### Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET  | `/llm/models` | Returns the provider/model catalog used by the dropdown. |
| POST | `/chat`       | Multi-turn chat. Body: `{message, conversation_history, provider?, model?}`. Returns `{reply, conversation_history, provider, model}`. |
| POST | `/analyze`    | Extracts structured JSON from an item description. Body: `{content, provider?, model?}`. Returns `{categories, tags, sentiment, summary}`. |
| GET  | `/chat-page`  | Serves the chat + analyze UI. |

`provider` and `model` are optional. If omitted, the server uses `DEFAULT_LLM_PROVIDER` and that provider's default model from the catalog (e.g. `gpt-4o-mini`, `claude-haiku-4-5`, `llama3.2`).

### Switching models from the API

```
curl -X POST http://localhost:9878/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hi","provider":"anthropic","model":"claude-haiku-4-5"}'
```

### Switching models from the UI

Open `/chat-page`. The "Model" card has a **provider** dropdown and a **model** dropdown that repopulates whenever you change the provider. Both the chat and the analyze panel use whichever provider/model is selected at send time. The "Reset conversation" button clears the local history.

### Provider abstraction

`llm_service.py` keeps the three providers behind a single interface, so `main.py` doesn't care which one is in use:

```python
# main.py snippet
provider = llm_service.get_provider(request.provider)        # OpenAI / Anthropic / Ollama
model    = llm_service.resolve_model(provider.name, request.model)
reply    = provider.chat(messages, model, max_tokens=512, temperature=0.7)
```

Provider clients are lazily constructed and cached, so the SDK only gets initialized for providers you actually use.

---

## Prompt documentation

Both system prompts and the few-shot examples live at the top of `main.py` (`CHAT_SYSTEM_PROMPT`, `ANALYZE_SYSTEM_PROMPT`, `ANALYZE_FEW_SHOT`) so they're easy to find and edit.

### `/chat` system message

> You are the in-app assistant for Boring API, a small item-manager and iris-flower-prediction tool. You help the user manage their items (create/edit/delete), explain what fields mean, and answer questions about the flower predictor (sepal/petal measurements, the three iris classes: setosa, versicolor, virginica). Be concise, friendly, and answer in 1-3 short paragraphs unless the user asks for more detail. If a question is unrelated to the app, answer briefly and steer back.

**Why this prompt:** It pins the assistant to the two things this app actually does (items + flower predictor), so generic questions get short answers and the model doesn't go off-topic. The "1-3 short paragraphs" rule keeps replies usable inside the small chat bubbles.

The frontend tracks `conversation_history` and sends it back on every request, so the model has full multi-turn context. The system message is re-prepended on the server each call - the client never needs to know about it.

### `/analyze` system message

> You are a data analysis assistant for an item-manager app. Given an item description, return ONLY a JSON object in this exact shape:
> ```json
> {
>   "categories": ["category1", "category2"],
>   "tags": ["tag1", "tag2", "tag3"],
>   "sentiment": "positive" | "negative" | "neutral",
>   "summary": "one sentence summary"
> }
> ```
> Rules:
> - 1-3 broad categories (e.g. "electronics", "kitchen", "book", "tool", "clothing").
> - 3-6 short lowercase tags (single words or short phrases).
> - sentiment reflects the tone of the description text itself.
> - summary is one sentence, under 20 words.
> - Output JSON only. No prose, no markdown fences.

### `/analyze` few-shot examples

Two examples are bundled into the user turn - one positive, one negative - so the model sees both the JSON shape and the sentiment polarity choices before the real input arrives:

```
Example 1:
Input: "Brand new noise-cancelling headphones, sound is crisp and battery lasts 30 hours. Worth every penny."
Output: {"categories": ["electronics", "audio"], "tags": ["headphones", "noise-cancelling", "battery", "review"], "sentiment": "positive", "summary": "Highly positive review of long-battery noise-cancelling headphones."}

Example 2:
Input: "Cheap plastic spatula, melted on the first use. Returning it."
Output: {"categories": ["kitchen", "utensil"], "tags": ["spatula", "plastic", "defective", "return"], "sentiment": "negative", "summary": "Negative review of a spatula that melted on first use."}
```

**Why few-shot:** Without examples, smaller and local models drift - they wrap JSON in markdown fences, add prose ("Sure! Here is..."), or invent extra fields. Two examples lock the shape and the tone.

### Failure handling

- **Temperature 0.2** for `/analyze` so structured output stays consistent across runs.
- `llm_service.complete_json` parses the response and, if it isn't valid JSON, strips common wrappers (markdown fences, leading prose, surrounding text) and tries again.
- If it still can't parse, it makes **one retry** with a stricter follow-up - *"That was not valid JSON. Reply with ONLY the JSON object, no prose, no markdown fences."* - at temperature 0.
- `/analyze` then validates that the required fields are present and that `sentiment` is in the allowed enum. If anything is off, it returns **HTTP 422** with the reason.
- Missing API key returns **HTTP 400** with a clear message (e.g. `"ANTHROPIC_API_KEY is not set"`).
- Any other SDK/provider exception returns **HTTP 502** with the underlying error string.

---

## File layout

```
.
├── main.py              # FastAPI app: items, predict, chat, analyze, page routes
├── db.py                # MongoDB connection (pymongo)
├── neural_network.py    # SimpleClassifier + ModelService (train/load/predict)
├── pytorch_basics.py    # Standalone PyTorch tensor/autograd demo
├── llm_service.py       # Provider abstraction (OpenAI/Anthropic/Ollama) + JSON retry
├── test_llm.py          # CLI sanity check: --provider, --model, --prompt
├── static/
│   ├── index.html       # Items UI
│   ├── chat.html        # AI chat + analyze UI with model dropdown
│   └── predict.html     # Flower predictor UI
├── model.pth            # Trained PyTorch weights
├── scaler.pkl           # Fitted StandardScaler
├── requirements.txt     # Python deps (UTF-8 - keep it that way)
├── Dockerfile
├── docker-compose.yml   # web + db services; forwards LLM env vars
├── .env.example         # Placeholders for keys + default provider + Mongo URL
└── .gitignore           # Includes .env so keys never get committed
```
