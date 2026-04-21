# Boring API
> Item manager app for AIE300. Full stack with FastAPI + MongoDB + Docker.

## Database
I went with **MongoDB** because I prefer working with it over SQL databases. Its more simple and just overall easy to scale and everything. Also pymongo is super straightforward to use with FastAPI. (Ive used it the most, between classes and personal projects.)

## How to Run
1. Make sure Docker Desktop is installed and running
2. Open a terminal and navigate to this project folder
3. Run:

```
docker-compose up --build
```

4. Go to http://localhost:8000 in your browser

To stop it:

```
docker-compose down
```

Your data will still be there when you start it back up because of the docker volume. 

## Architecture
```
Browser (localhost:8000)
    |
    | fetch() calls
    v
FastAPI (web container, port 8000)
    - serves static/index.html
    - REST API endpoints (/items)
    - pymongo driver
    |
    | port 27017
    v
MongoDB (db container)
    - stores items as documents
    - data saved in mongodata volume
```

## Endpoints

| Method | Path | What it does |
|--------|------|-------------|
| GET | /items | get all items |
| GET | /items/{id} | get one item (404 if not found) |
| POST | /items | create a new item (returns 201) |
| PUT | /items/{id} | update an item (404 if not found) |
| DELETE | /items/{id} | delete an item (404 if not found) |

Note: item ids are MongoDB ObjectId hex strings (like `6789abc...`), not integers.

Item body for POST/PUT:

```json
{
  "name": "something",
  "description": "optional"
}
```


## Tech Stack
- Python 3.11 / FastAPI
- MongoDB 7
- pymongo
- HTML/CSS/JS (no frameworks)
- Docker + docker-compose
