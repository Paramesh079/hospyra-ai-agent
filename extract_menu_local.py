import os
import json
import csv
import argparse
import sys
import base64
import requests

def encode_image(image_path):
    """
    Encode the image in base64.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        sys.exit(1)

def analyze_menu_image_local(image_path, model_name, api_url):
    """
    Calls a local LLM through an Ollama-compatible API to extract structured menu data.
    """
    print(f"Reading image: {image_path}")
    base64_image = encode_image(image_path)

    prompt_text = (
        "You are an expert data extractor. Analyze this restaurant menu image "
        "and extract all the food items strictly into a JSON array. "
        "Each object in the array MUST have the exact following keys:\n"
        "- 'menu': The main section heading (e.g., 'SOUPS').\n"
        "- 'menu_category': The sub-category under the main section (e.g., 'Western', 'Mexican', 'Pan Asian').\n"
        "- 'item_name': The exact name of the dish.\n"
        "- 'price': The price of the dish as a int (e.g. '199/-', '299/-').\n"
        "- 'is_vegetarian': true if vegetarian (green symbol or Veg label), false if non-vegetarian (red symbol or Non-Veg label).\n\n"
        "IMPORTANT RULES FOR VEG/NON-VEG:\n"
        "1. If an item indicates BOTH Veg and Non-Veg options (e.g., it has both a green and red symbol, or says 'Veg/Non-Veg', or has two prices separated by a slash like '299/349/-'), you MUST split it into TWO separate objects in the array.\n"
        "2. For the split items, append '(Veg)' or '(Non-Veg)' to the 'item_name' if not already present.\n"
        "3. Assign the first price to the Veg option and the second price to the Non-Veg option.\n"
        "4. Set 'is_vegetarian' to true for the Veg option and false for the Non-Veg option.\n\n"
        "Do not include any other text, reasoning, or markdown format blocks like ```json. "
        "Output strictly valid JSON starting with [ and ending with ]."
    )

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "menu": {"type": "string"},
                "menu_category": {"type": "string"},
                "item_name": {"type": "string"},
                "price": {"type": "integer"},
                "is_vegetarian": {"type": "boolean"}
            },
            "required": ["menu", "menu_category", "item_name", "price", "is_vegetarian"]
        }
    }

    # Standard payload for Ollama POST /api/chat
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": [base64_image]
            }
        ],
        "format": schema,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    print(f"Sending request to local model '{model_name}' at {api_url}...")
    try:
        response = requests.post(f"{api_url}/api/chat", json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('message', {}).get('content', '')
        
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to local API: {e}")
        print(f"Make sure your local model server is running at {api_url}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract menu from image to CSV using a purely local model.")
    parser.add_argument("image_path", nargs="?", default="/home/paramesh/Downloads/samsgriddle_menu_items/images/3.png", help="Path to the menu image")
    parser.add_argument("--out", default="extracted_menu.csv", help="Output CSV filename")
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Local Model ID to use (e.g., llava, ministral, pixtral)")
    parser.add_argument("--api-url", default="http://localhost:11434", help="URL of the local inference server (default Ollama port)")
    
    args = parser.parse_args()
    
    # 1. Analyze image
    result_text = analyze_menu_image_local(args.image_path, args.model, args.api_url)
    
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
        
    # 4. Write data to CSV
    print(f"Writing {len(data)} items to '{args.out}'...")
    with open(args.out, mode='w', newline='', encoding='utf-8') as f:
        # Define fieldnames to handle if a key is missing
        fieldnames = ['menu', 'menu_category', 'item_name', 'price', 'is_vegetarian']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in data:
            # Handle menu_category defaulting to 'Regular'
            menu_cat = item.get('menu_category', '')
            if not menu_cat or str(menu_cat).lower() == 'none' or str(menu_cat).strip() == '':
                menu_cat = 'regular'
                
            row = {
                'menu': item.get('menu', ''),
                'menu_category': menu_cat,
                'item_name': item.get('item_name', ''),
                'price': item.get('price', ''),
                'is_vegetarian': item.get('is_vegetarian', '')
            }
            writer.writerow(row)
            
    print("Done! Data successfully extracted using local model.")

if __name__ == "__main__":
    main()
