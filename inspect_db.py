import os
from sqlalchemy import text
from db import engine

def get_schema():
    with engine.connect() as conn:
        tables_res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
        tables = [row[0] for row in tables_res]
        
        for table in tables:
            print(f"\nTable: {table}")
            cols_res = conn.execute(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';"))
            for row in cols_res:
                print(f"  - {row[0]}: {row[1]}")

if __name__ == "__main__":
    get_schema()
