import os
import csv
from datetime import datetime
from sqlalchemy import text
from db import engine

import argparse

def insert_menu_data(csv_file, hotel_id):
    try:
        with engine.begin() as conn:
            # 1. Verify Restaurant exists
            print(f"Verifying restaurant ID '{hotel_id}' exists...")
            result = conn.execute(
                text("SELECT id, name FROM restaurants WHERE id = :hid LIMIT 1"),
                {"hid": hotel_id}
            ).fetchone()
            
            if not result:
                print(f"Error: No restaurant found with ID {hotel_id}. Please use a valid ID or create the restaurant first.")
                return
                
            hotel_name = result[1]
            print(f"Found restaurant: {hotel_name} (ID: {hotel_id})")

            # 2. Track menus and categories to avoid duplicates
            menu_cache = {} # name -> id
            category_cache = {} # (menu_id, category_name) -> id

            print(f"Reading CSV: {csv_file}")
            with open(csv_file, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                items_inserted = 0
                for row in reader:
                    menu_name = row.get('menu', '').strip()
                    category_name = row.get('menu_category', '').strip()
                    item_name = row.get('item_name', '').strip()
                    price_str = row.get('price', '').strip()
                    is_veg_str = row.get('is_vegetarian', '').strip()
                    
                    if not menu_name or not item_name:
                        continue # Skip invalid rows

                    # 3. Handle Menu
                    if menu_name not in menu_cache:
                        menu_res = conn.execute(
                            text("SELECT id FROM menus WHERE hotel_id = :hid AND name = :name LIMIT 1"),
                            {"hid": hotel_id, "name": menu_name}
                        ).fetchone()
                        
                        if menu_res:
                            menu_id = menu_res[0]
                        else:
                            print(f"Creating menu: {menu_name}")
                            insert_menu = text("""
                                INSERT INTO menus (hotel_id, name, is_active, created_at, updated_at)
                                VALUES (:hid, :name, true, :now, :now)
                                RETURNING id
                            """)
                            menu_id = conn.execute(
                                insert_menu,
                                {"hid": hotel_id, "name": menu_name, "now": datetime.utcnow()}
                            ).fetchone()[0]
                        menu_cache[menu_name] = menu_id
                    
                    menu_id = menu_cache[menu_name]

                    # 4. Handle Menu Category
                    cat_key = (menu_id, category_name)
                    if cat_key not in category_cache:
                        cat_res = conn.execute(
                            text("SELECT id FROM menu_categories WHERE menu_id = :mid AND name = :name LIMIT 1"),
                            {"mid": menu_id, "name": category_name}
                        ).fetchone()
                        
                        if cat_res:
                            cat_id = cat_res[0]
                        else:
                            print(f"Creating category: {category_name} (under {menu_name})")
                            insert_cat = text("""
                                INSERT INTO menu_categories (menu_id, name, is_active, created_at, updated_at)
                                VALUES (:mid, :name, true, :now, :now)
                                RETURNING id
                            """)
                            cat_id = conn.execute(
                                insert_cat,
                                {"mid": menu_id, "name": category_name, "now": datetime.utcnow()}
                            ).fetchone()[0]
                        category_cache[cat_key] = cat_id
                    
                    cat_id = category_cache[cat_key]

                    # 5. Handle Menu Item
                    # Parse price: remove anything that isn't a digit or decimal
                    clean_price = ''.join(c for c in price_str if c.isdigit() or c == '.')
                    price_val = float(clean_price) if clean_price else 0.0
                    
                    # Parse is_vegetarian
                    is_veg = str(is_veg_str).lower() == 'true'
                    
                    item_res = conn.execute(
                        text("SELECT id FROM menu_items WHERE category_id = :cid AND name = :name LIMIT 1"),
                        {"cid": cat_id, "name": item_name}
                    ).fetchone()
                    
                    if not item_res:
                        insert_item = text("""
                            INSERT INTO menu_items (category_id, name, price, is_vegetarian, is_active, created_at, updated_at)
                            VALUES (:cid, :name, :price, :is_veg, true, :now, :now)
                        """)
                        conn.execute(
                            insert_item,
                            {
                                "cid": cat_id, 
                                "name": item_name, 
                                "price": price_val, 
                                "is_veg": is_veg,
                                "now": datetime.utcnow()
                            }
                        )
                        items_inserted += 1
                        
                print(f"Successfully inserted {items_inserted} new menu items into the database!")
                
    except Exception as e:
        print(f"Database insertion failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert menu data from CSV into the database.")
    parser.add_argument("csv_path", default="extracted_menu.csv", nargs="?", help="Path to the extracted CSV file")
    parser.add_argument("--hotel-id", type=int, required=True, help="The ID of the restaurant (hotel) in the database")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: {args.csv_path} not found.")
    else:
        insert_menu_data(args.csv_path, args.hotel_id)
