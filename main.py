from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Boring API ")

# Pydantic model
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    quantity: int


items_db: dict[int, dict] = {}
next_id: int = 1

@app.get("/items", response_model=List[Item])
def get_items():
    return list(items_db.values())


@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]


@app.post("/create_item", response_model=Item, status_code=201)
def create_item(item: Item):
    global next_id
    item_id = next_id
    next_id += 1
    item_dict = item.dict()
    item_dict["id"] = item_id
    items_db[item_id] = item_dict
    return item_dict


@app.put("/update_item/{item_id}", response_model=Item)
def update_item(item_id: int, updated_item: Item):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    item_dict = updated_item.dict()
    item_dict["id"] = item_id
    items_db[item_id] = item_dict
    return item_dict


@app.delete("/delete_item/{item_id}", status_code=200)
def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]
    return {"detail": "Item deleted successfully"}