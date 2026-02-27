import os
import json
import csv
import argparse
import sys
import boto3
from botocore.exceptions import ClientError

def analyze_menu_image_bedrock(image_path, model_id):
    """
    Calls AWS Bedrock with a Mistral vision model to extract structured menu data.
    """
    print(f"Reading image: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        sys.exit(1)

    # Determine image format (Converse API requires jpeg, png, webp, or gif)
    ext = image_path.lower().split('.')[-1]
    image_format = ext if ext in ['png', 'jpeg', 'gif', 'webp'] else 'jpeg'
    if image_format == 'jpg':
        image_format = 'jpeg'

    # Initialize Bedrock Runtime client. It respects typical AWS env vars.
    client = boto3.client("bedrock-runtime")

    prompt_text = (
        "You are an expert data extractor. Analyze this restaurant menu image "
        "and extract all the food items strictly into a JSON array. "
        "Each object in the array MUST have the exact following keys:\n"
        "- 'menu': The main section heading (e.g., 'SOUPS').\n"
        "- 'menu_category': The sub-category under the main section (e.g., 'Western', 'Mexican', 'Pan Asian').\n"
        "- 'item_name': The exact name of the dish.\n"
        "- 'price': The price of the dish as text (e.g. '199/-', '299/349/-').\n"
        "Do not include any other text, reasoning, or markdown format blocks like ```json. "
        "Output strictly valid JSON starting with [ and ending with ]."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": image_format,
                        "source": {"bytes": image_bytes}
                    }
                },
                {"text": prompt_text}
            ]
        }
    ]

    print(f"Sending request to Bedrock using Mistral model: {model_id}...")
    try:
        # converse API is the standard for multimodal on bedrock
        response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={"temperature": 0.1}
        )
        
        output_text = response['output']['message']['content'][0]['text']
        return output_text
    except ClientError as e:
        print(f"\nAWS Bedrock Error: {e}")
        print("\nNOTE: Ensure that you have enabled access to the Mistral vision models ")
        print("(like Pixtral Large) in the AWS Bedrock Console in your region.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract menu from image to CSV using Bedrock Mistral model.")
    parser.add_argument("image_path", help="Path to the menu image (e.g., restaurant_menu.jpg)")
    parser.add_argument("--out", default="extracted_menu.csv", help="Output CSV filename")
    # Using pixtral model for vision capabilities (Mistral's vision models)
    parser.add_argument("--model", default="mistral.pixtral-large-2411-v1:0", 
                        help="Mistral Bedrock Model ID (must support vision, e.g., Pixtral)")
    
    args = parser.parse_args()
    
    # 1. Analyze image
    result_text = analyze_menu_image_bedrock(args.image_path, args.model)
    
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
        writer = csv.DictWriter(f, fieldnames=['menu', 'menu_category', 'item_name', 'price'])
        writer.writeheader()
        for item in data:
            row = {
                'menu': item.get('menu', ''),
                'menu_category': item.get('menu_category', ''),
                'item_name': item.get('item_name', ''),
                'price': item.get('price', '')
            }
            writer.writerow(row)
            
    print("Done! Data successfully extracted.")

if __name__ == "__main__":
    main()
