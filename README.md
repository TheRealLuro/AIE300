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


# -----------------------------------------------------------------------------------------------------
## Flower predictor

Same setup as the item manager stuff, except this part uses PyTorch + a neural network to predict iris flowers. The page to test it is:

http://localhost:8000/predict-page

I used the Iris dataset from sklearn because its simple, works well for classification, and was good for learning how PyTorch training/inference works without making the project super overcomplicated.

The model predicts between all 3 flower classes:

- setosa  
- versicolor  
- virginica  

The neural network was built using PyTorch with:
- multiple linear layers
- ReLU activation functions
- CrossEntropyLoss
- Adam optimizer
- DataLoader batching

I also used a StandardScaler so the feature values are normalized before training/inference. The model trains on startup if there isnt already a saved model file, otherwise it just loads the existing trained model.

---

### API Endpoint

| Method | Path | What it does |
|--------|------|-------------|
| POST | /predict | predicts flower type from 4 float inputs |

---

### Example Request

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

### Example Response
{
  "prediction": "setosa",
  "confidence": 0.9987,
  "probabilities": {
    "setosa": 0.9987,
    "versicolor": 0.0011,
    "virginica": 0.0002
  }
}

## What the model uses
- train/test split
- DataLoader batching
- autograd/backpropagation
model saving/loading with torch.save
model.eval() and torch.no_grad() during inference
PyTorch basics section

I also included a separate script that demonstrates: ( pytorch_basics.py)

- tensor creation
- tensor math
- matrix multiplication
- autograd gradients

# Swagger docs for testing:

> http://localhost:8000/docs

You can test predictions there directly or use the frontend prediction page. (http://localhost:8000/predict-page)


