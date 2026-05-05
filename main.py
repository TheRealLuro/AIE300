from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
from db import items_collection
from neural_network import ModelService
from fastapi.responses import FileResponse

app = FastAPI(title="Boring API")
nn_service = ModelService()
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


# serve the frontend html - this has to be AFTER the api routes
# or else it will catch everything
app.mount("/static", StaticFiles(directory="static"), name="static")
