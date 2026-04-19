from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from db import ItemModel, init_db, get_db

app = FastAPI(title="Boring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ItemCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ItemRead(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/items", response_model=List[ItemRead])
def get_items(db: Session = Depends(get_db)):
    return db.query(ItemModel).all()


@app.get("/items/{item_id}", response_model=ItemRead)
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.post("/items", response_model=ItemRead, status_code=201)
def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    db_item = ItemModel(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


@app.put("/items/{item_id}", response_model=ItemRead)
def update_item(item_id: int, item: ItemCreate, db: Session = Depends(get_db)):
    db_item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    db_item.name = item.name
    db_item.description = item.description
    db.commit()
    db.refresh(db_item)
    return db_item


@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(db_item)
    db.commit()
    return {"detail": "Item deleted successfully"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
