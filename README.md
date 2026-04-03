# ITEM API
> Just a simple item api for AIE300.


## HOW TO SETUP
* Install docker desktop
* navigate to this projects dir in CMD or any terminal
* run:
    - docker build -t boring_api .
    - docker run --rm --name boring_api -p 8000:8000 -d boring_api
    > make sure Docker Desktop is running first.
* call http://127.0.0.1:8000 with the proper /[call]
    - example: http://127.0.0.1:8000/create_item

### CHECK http://127.0.0.1:8000/docs for info on how to use the apis


| Path | Type | Info needed |
| :--------: | :--------: | :--------: |
| /items | GET | Nothing |
| /items/{item_id} | GET | Item_id |
| /create_item | POST | {"name": "string","description": "string","price": 0,"tax": 0} |
| /update_item/{item_id} | PUT | Item_id, and edited info {"name": "string", "description": "string", "price": 0, "tax": 0} |
| /delete_item/{item_id} | DELETE | Item_id |
