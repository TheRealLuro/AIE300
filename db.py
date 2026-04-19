## this file handles all the database stuff
## using sqlalchemy to talk to postgres

import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# get the database url from environment, default to localhost for local dev
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/items")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# this is our item table in the database
class ItemModel(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)

# creates the tables if they dont exist yet
def init_db():
    Base.metadata.create_all(bind=engine)

# this gives us a database session to use in our routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
