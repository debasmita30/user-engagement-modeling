import pandas as pd
from sqlalchemy import create_engine

# Change password if your postgres password is different
DATABASE_URL = "postgresql+psycopg2://postgres:Aarush227@localhost:5432/teams_analytics"

engine = create_engine(DATABASE_URL)

print("Loading users...")
users = pd.read_csv("data/users.csv")
users.to_sql("users", engine, if_exists="replace", index=False)
print("Users loaded.")

print("Loading events...")
events = pd.read_csv("data/events_small.csv")
events.to_sql("events", engine, if_exists="replace", index=False, chunksize=5000)
print("Events loaded.")
