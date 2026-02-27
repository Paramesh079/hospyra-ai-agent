import os
import json
import csv
import argparse
import sys
import base64
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime, timezone
from sqlalchemy import text
from db import engine

load_dotenv()

# Azure OpenAI credentials (same .env as sql_agent.py)
_azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
)
_azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "Hospyra")

def encode_image(image_path):
    """
    Encode the image in base64.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Could not find image at {image_path}")

def analyze_menu_image_local(image_path, model_name=None, api_url=None):
    """
    Calls Azure OpenAI vision deployment to extract structured menu data from an image.
    model_name and api_url are accepted for API compatibility but ignored (Azure config comes from .env).
    """
    print(f"Reading image: {image_path}")
    base64_image = encode_image(image_path)

    # Detect image mime type from extension
    ext = os.path.splitext(image_path)[-1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    prompt_text = (
        "You are an expert data extractor. Analyze this restaurant menu image "
        "and extract ALL items — including food, drinks, beverages, desserts, and any other listed items — strictly into a JSON array. "
        "Do NOT skip any section. Every item visible on the image must be included. "
        "Each object in the array MUST have the exact following keys:\n"
        "- 'menu': The main section heading exactly as printed on the image (e.g., 'SOUPS', 'BEVERAGES', 'DESSERTS').\n"
        "- 'menu_category': Follow this priority order:\n"
        "  1. If a sub-category label is EXPLICITLY printed on the image (e.g., 'Western', 'Pan Asian'), use that exact text.\n"
        "  2. If NO sub-category label is visible, assign a category based on EACH ITEM'S OWN nature:\n"
        "     - Use 'Beverages' if the item is a drink (juice, water, soda, tea, coffee, mocktail, cocktail, milkshake, lassi, etc.).\n"
        "     - Use 'Non-Veg' if the item is a food that contains or is named after meat, poultry (chicken, mutton, lamb, beef, pork), seafood, or eggs.\n"
        "     - Use 'Veg' if the item is a food that contains no meat, poultry, seafood, or eggs.\n"
        "     - Every item must independently get its own category — do NOT group all items under a single category.\n"
        "     - Never use cuisine names (like 'Indian', 'Chinese') unless they are explicitly printed on the image.\n"
        "- 'item_name': The exact name of the item as printed.\n"
        "- 'price': The price of the item as a string (e.g. '199/-', '299/-').\n"
        "- 'is_vegetarian': true if vegetarian (green symbol or Veg label), false if non-vegetarian (red symbol or Non-Veg label). "
        "For beverages and desserts with no symbol, set true by default.\n\n"
        "IMPORTANT RULES FOR VEG/NON-VEG:\n"
        "1. If an item indicates BOTH Veg and Non-Veg options (e.g., it has both a green and red symbol, or says 'Veg/Non-Veg', or has two prices separated by a slash like '299/349/-'), you MUST split it into TWO separate objects in the array.\n"
        "2. For the split items, append '(Veg)' or '(Non-Veg)' to the 'item_name' if not already present.\n"
        "3. Assign the first price to the Veg option and the second price to the Non-Veg option.\n"
        "4. Set 'is_vegetarian' to true for the Veg option and false for the Non-Veg option.\n\n"
        "Do not include any other text, reasoning, or markdown format blocks like ```json. "
        "Output strictly valid JSON starting with [ and ending with ]."
    )


    print(f"Sending image to Azure OpenAI deployment '{_azure_deployment}'...")
    response = _azure_client.chat.completions.create(
        model=_azure_deployment,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        temperature=0.1,
        max_tokens=8192,
    )
    return response.choices[0].message.content


def insert_menu_data_to_db(data, hotel_id, save_csv=None):
    """
    Inserts the extracted JSON data dictionary list into the PostgreSQL database.
    Optionally saves it to a CSV file as well.
    """
    try:
        with engine.begin() as conn:
            # 1. Verify Restaurant exists
            print(f"\nVerifying restaurant ID '{hotel_id}' exists...")
            result = conn.execute(
                text("SELECT id, name FROM restaurants WHERE id = :hid LIMIT 1"),
                {"hid": hotel_id}
            ).fetchone()
            
            if not result:
                raise ValueError(f"Error: No restaurant found with ID {hotel_id}. Please use a valid ID or create the restaurant first.")
                
            hotel_name = result[1]
            print(f"Found restaurant: {hotel_name} (ID: {hotel_id})")

            # 2. Track menus and categories to avoid duplicates
            menu_cache = {} # name -> id
            category_cache = {} # (menu_id, category_name) -> id
            
            items_inserted = 0

            # Optional CSV Writer setup
            csv_file = None
            csv_writer = None
            if save_csv:
                print(f"Also writing extracted items to '{save_csv}'...")
                csv_file = open(save_csv, mode='w', newline='', encoding='utf-8')
                fieldnames = ['menu', 'menu_category', 'item_name', 'price', 'is_vegetarian']
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()

            print(f"Processing {len(data)} items to insert into DB...")
            for item in data:
                # Handle menu_category defaulting to 'Regular'
                menu_name = item.get('menu', '').strip()
                category_name = item.get('menu_category', '').strip()
                item_name = item.get('item_name', '').strip()
                price_str = str(item.get('price', '')).strip()
                is_veg_raw = item.get('is_vegetarian', '')
                
                if not category_name or str(category_name).lower() == 'none' or str(category_name).strip() == '':
                    category_name = 'Regular'
                
                if not menu_name or not item_name:
                    continue # Skip invalid rows
                    
                # Write to CSV if enabled
                if csv_writer:
                    row = {
                        'menu': menu_name,
                        'menu_category': category_name,
                        'item_name': item_name,
                        'price': price_str,
                        'is_vegetarian': is_veg_raw
                    }
                    csv_writer.writerow(row)

                # 3. Handle Database Menus
                if menu_name not in menu_cache:
                    menu_res = conn.execute(
                        text("SELECT id FROM menus WHERE hotel_id = :hid AND name = :name LIMIT 1"),
                        {"hid": hotel_id, "name": menu_name}
                    ).fetchone()
                    
                    if menu_res:
                        menu_id = menu_res[0]
                    else:
                        print(f"Creating new menu: '{menu_name}'")
                        insert_menu = text("""
                            INSERT INTO menus (hotel_id, name, is_active, created_at, updated_at)
                            VALUES (:hid, :name, true, :now, :now)
                            RETURNING id
                        """)
                        menu_id = conn.execute(
                            insert_menu,
                            {"hid": hotel_id, "name": menu_name, "now": datetime.now(timezone.utc)}
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
                        print(f"Creating new category: '{category_name}' (under menu '{menu_name}')")
                        insert_cat = text("""
                            INSERT INTO menu_categories (menu_id, name, is_active, created_at, updated_at)
                            VALUES (:mid, :name, true, :now, :now)
                            RETURNING id
                        """)
                        cat_id = conn.execute(
                            insert_cat,
                            {"mid": menu_id, "name": category_name, "now": datetime.now(timezone.utc)}
                        ).fetchone()[0]
                    category_cache[cat_key] = cat_id
                
                cat_id = category_cache[cat_key]

                # 5. Handle Menu Item
                # Parse price: remove anything that isn't a digit or decimal
                clean_price = ''.join(c for c in price_str if c.isdigit() or c == '.')
                price_val = float(clean_price) if clean_price else 0.0
                
                # Parse is_vegetarian
                is_veg = str(is_veg_raw).lower() == 'true'
                
                item_res = conn.execute(
                    text("SELECT id FROM menu_items WHERE category_id = :cid AND name = :name LIMIT 1"),
                    {"cid": cat_id, "name": item_name}
                ).fetchone()
                
                if not item_res:
                    insert_item = text("""
                        INSERT INTO menu_items (category_id, name, price, is_vegetarian, created_at, updated_at)
                        VALUES (:cid, :name, :price, :is_veg, :now, :now)
                    """)
                    conn.execute(
                        insert_item,
                        {
                            "cid": cat_id, 
                            "name": item_name, 
                            "price": price_val, 
                            "is_veg": is_veg,
                            "now": datetime.now(timezone.utc)
                        }
                    )
                    items_inserted += 1
                    
            print(f"\nSuccess! Inserted {items_inserted} new menu items into the database!")
            
            if csv_file:
                csv_file.close()
                print(f"Extraction results also saved to {save_csv}")
                
    except Exception as e:
        raise RuntimeError(f"Database insertion failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract menu from image and insert into PostgreSQL database using Azure OpenAI.")
    parser.add_argument("image_path", nargs="?", default="/home/paramesh/Downloads/17_restaurant.jpg", help="Path to the menu image")
    parser.add_argument("--hotel-id", default=17, type=int, help="The ID of the restaurant (hotel) in the database")
    parser.add_argument("--out", default="extracted_menu.csv", help="Optional: output CSV filename to also save the extracted data (default: extracted_menu.csv)")

    args = parser.parse_args()

    # 1. Analyze image via Azure OpenAI
    result_text = analyze_menu_image_local(args.image_path)

    
    # 2. Clean the string to extract raw JSON
    clean_text = result_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()
    
    # 3. Parse JSON
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from model output. Raw output was:\n{result_text}")
        sys.exit(1)
        
    if not isinstance(data, list):
        print("Model did not return a list of items. Format error.")
        sys.exit(1)
        
    # 4. Insert data into Database (and optionally save to CSV)
    insert_menu_data_to_db(data, args.hotel_id, args.out)

if __name__ == "__main__":
    main()
