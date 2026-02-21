import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hospyra:Rds12345@hospyra.postgres.database.azure.com:5432/postgres?sslmode=require")

engine = create_engine(DATABASE_URL)
