import os
import argparse
from sqlalchemy import text
from db import engine

def read_database(restaurant_id: int = None):
    print(f"[LOG] Connecting to database...")
    try:
        with engine.connect() as conn:
            # Get all tables in the public schema
            tables_query = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = [row[0] for row in conn.execute(tables_query)]
            
            print(f"[LOG] Found {len(tables)} tables: {', '.join(tables)}")
            
            # Dictionary to store table data records
            db_data = {}
            
            for table in tables:
                print(f"\\n" + "="*50)
                print(f"--- Reading Table: {table} ---")
                
                # Check for restaurant_id column
                cols_query = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}';")
                columns = [row[0] for row in conn.execute(cols_query)]
                
                query_str = f"SELECT * FROM {table}"
                params = {}
                
                has_restaurant_id = "restaurant_id" in columns
                
                if restaurant_id is not None:
                    if has_restaurant_id:
                        print(f"[LOG] Table '{table}' has 'restaurant_id' column. Filtering for restaurant_id={restaurant_id}.")
                        query_str += " WHERE restaurant_id = :restaurant_id"
                        params = {"restaurant_id": restaurant_id}
                    else:
                        print(f"[LOG] Table '{table}' does NOT have 'restaurant_id' column. Fetching all rows.")
                else:
                    print(f"[LOG] Fetching all rows for table '{table}'.")
                
                # Fetch data
                try:
                    result = conn.execute(text(query_str), params)
                    rows = [dict(row) for row in result.mappings()]
                    db_data[table] = rows
                    print(f"[LOG] Successfully fetched {len(rows)} rows.")
                    
                    if len(rows) > 0:
                        print(f"--- First 5 rows of {table} ---")
                        for i, row in enumerate(rows[:5]):
                            print(f"Row {i+1}: {row}")
                    else:
                        print(f"Table {table} is empty for this query.")
                except Exception as e:
                    print(f"[ERROR] Failed to read table {table}: {e}")
            
            print(f"\\n[LOG] Finished reading database. Processed {len(tables)} tables.")
            return db_data

    except Exception as e:
        print(f"[ERROR] Failed to connect or read database: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read entire PostgreSQL database and optionally filter by restaurant_id.")
    parser.add_argument("-r", "--restaurant_id", type=int, help="Optional: Restaurant ID to filter tables by", default=None)
    args = parser.parse_args()
    
    print("=" * 50)
    print("DATABASE READER SCRIPT")
    print("=" * 50)
    if args.restaurant_id is not None:
        print(f"Mode: Filtered by restaurant_id = {args.restaurant_id}")
    else:
        print("Mode: Read All tables")
    print("=" * 50)
    
    read_database(restaurant_id=args.restaurant_id)
