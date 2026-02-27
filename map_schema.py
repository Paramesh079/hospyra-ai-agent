from sqlalchemy import text
from db import engine

def map_db():
    print("Mapping database tables to find hotel_id relationships...")
    with engine.connect() as conn:
        tables_res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = [row[0] for row in tables_res]
        
        for table in tables:
            cols_res = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"))
            cols = [col[0] for col in cols_res]
            
            if 'hotel_id' in cols or 'restaurant_id' in cols:
                print(f"[HAS HOTEL ID] {table}: {', '.join(cols)}")
            elif 'menu_item_id' in cols or 'category_id' in cols:
                print(f"[LINKS TO MENU] {table}: {', '.join(cols)}")
            elif table in ['menu_items', 'menu_categories']:
                print(f"[MENU TABLE] {table}: {', '.join(cols)}")

if __name__ == "__main__":
    map_db()
