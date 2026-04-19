## this file handles all the database stuff
## using pymongo to talk to mongodb

import os
from pymongo import MongoClient

# get the mongo url from environment, default to localhost for local dev
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")

client = MongoClient(MONGO_URL)

# our database and collection
db = client["items_db"]
items_collection = db["items"]
