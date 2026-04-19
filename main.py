from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from db import ItemModel, init_db, get_db

app = FastAPI(title="Boring API")

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
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True

# make the tables when the app starts up
@app.on_event("startup")
def on_startup():
    init_db()


# --- CRUD routes ---

# get all items
@app.get("/items", response_model=List[ItemRead])
def get_items(db: Session = Depends(get_db)):
    items = db.query(ItemModel).all()
    return items

# get one item by id
@app.get("/items/{item_id}", response_model=ItemRead)
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

# create a new item
@app.post("/items", response_model=ItemRead, status_code=201)
def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    new_item = ItemModel(name=item.name, description=item.description)
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item

# update an existing item
@app.put("/items/{item_id}", response_model=ItemRead)
def update_item(item_id: int, item: ItemCreate, db: Session = Depends(get_db)):
    existing = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if existing is None:
        raise HTTPException(status_code=404, detail="Item not found")
    existing.name = item.name
    existing.description = item.description
    db.commit()
    db.refresh(existing)
    return existing

# delete an item
@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    existing = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if existing is None:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(existing)
    db.commit()
    return {"detail": "Item deleted"}


# serve the frontend html - this has to be AFTER the api routes
# or else it will catch everything
app.mount("/", StaticFiles(directory="static", html=True), name="static")
