# Boring API
> Item manager app for AIE300. Full stack with FastAPI + PostgreSQL + Docker.

## Database
I went with **PostgreSQL** because its what most people use in production and it works well with SQLAlchemy. Also it runs as its own container in docker-compose which is pretty cool for learning how services talk to each other.

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
    - SQLAlchemy ORM
    |
    | port 5432
    v
PostgreSQL (db container)
    - stores items in a table
    - data saved in pgdata volume
```

## Endpoints

| Method | Path | What it does |
|--------|------|-------------|
| GET | /items | get all items |
| GET | /items/{id} | get one item (404 if not found) |
| POST | /items | create a new item (returns 201) |
| PUT | /items/{id} | update an item (404 if not found) |
| DELETE | /items/{id} | delete an item (404 if not found) |

Item body for POST/PUT:
```json
{
  "name": "something",
  "description": "optional"
}
```

## Screenshot
*(add screenshot here)*

## Tech Stack
- Python 3.11 / FastAPI
- PostgreSQL 16
- SQLAlchemy + psycopg2
- HTML/CSS/JS (no frameworks)
- Docker + docker-compose
