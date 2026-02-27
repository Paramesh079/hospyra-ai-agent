from sqlalchemy import text
from db import engine

def get_schema():
    try:
        with engine.connect() as conn:
            print("--- dish_review_analytics ---")
            result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'dish_review_analytics';"))
            for row in result:
                print(f"{row[0]}: {row[1]}")
                
            print("\n--- reviews ---")
            result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'reviews';"))
            for row in result:
                print(f"{row[0]}: {row[1]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_schema()
