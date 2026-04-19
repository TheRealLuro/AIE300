# Boring API - Item Manager

A full-stack CRUD application built with FastAPI, PostgreSQL, and vanilla HTML/CSS/JS, containerized with Docker.

## Database Choice

**PostgreSQL** — chosen for being production-realistic, widely used, and having excellent Python support via SQLAlchemy + psycopg2. Running it as a separate Docker service also demonstrates service networking with docker-compose.

## How to Run

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Quick Start
```bash
# Clone the repo and cd into it
git clone <your-repo-url>
cd AIE300

# Build and start the full stack
docker-compose up --build
```

Open **http://localhost:8000** in your browser.

### Stopping
```bash
docker-compose down
```

Data persists across restarts thanks to the named Docker volume (`pgdata`).

### Verify Persistence
```bash
docker-compose down
docker-compose up
# Your items should still be there!
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                   Browser                    │
│          (http://localhost:8000)              │
└──────────────┬───────────────────────────────┘
               │  HTTP (fetch API calls)
               ▼
┌──────────────────────────────────────────────┐
│            FastAPI  (web container)           │
│                  Port 8000                   │
│  ┌────────────┐  ┌─────────────────────────┐ │
│  │ Static Files│  │  REST API  (/items/*)   │ │
│  │ index.html  │  │  SQLAlchemy ORM         │ │
│  └────────────┘  └──────────┬──────────────┘ │
└─────────────────────────────┼────────────────┘
                              │  TCP :5432
                              ▼
               ┌──────────────────────────┐
               │   PostgreSQL 16          │
               │   (db container)         │
               │   Volume: pgdata         │
               └──────────────────────────┘
```

## API Endpoints

| Method | Path             | Description              | Status Codes |
|--------|------------------|--------------------------|--------------|
| GET    | `/items`         | List all items           | 200          |
| GET    | `/items/{id}`    | Get a single item        | 200, 404     |
| POST   | `/items`         | Create a new item        | 201          |
| PUT    | `/items/{id}`    | Update an existing item  | 200, 404     |
| DELETE | `/items/{id}`    | Delete an item           | 200, 404     |

### Item Schema

**Request body** (POST / PUT):
```json
{
  "name": "string (required)",
  "description": "string (optional)"
}
```

**Response body**:
```json
{
  "id": 1,
  "name": "string",
  "description": "string or null"
}
```

## Screenshot

> *(Add a screenshot of the running app here)*

## Tech Stack

- **Backend**: Python 3.11, FastAPI, SQLAlchemy, psycopg2-binary
- **Database**: PostgreSQL 16
- **Frontend**: Vanilla HTML, CSS, JavaScript
- **Infrastructure**: Docker, docker-compose
