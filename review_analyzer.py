from decimal import Decimal
import asyncio
import re
import json
import time
from sqlalchemy import text
from db import engine
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "Hospyra")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
print(f"[LOG] Initializing Azure OpenAI LLM for review analysis with deployment: {deployment_name}")
llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    api_version=api_version,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)


async def analyze_reviews_with_agent(limit: int = 20):
    """
    LLM-based review analysis (Streaming).
    Flow:
    1. Load reviews from PostgreSQL (Limit to 200)
    2. LLM extracts dishes + sentiment (Parallel execution, streaming as completed)
    3. Validate dish names against menu table
    4. Yield valid positive dishes as Server-Sent Events (SSE)
    """

    try:
        total_start_time = time.time()
        
        # 1️⃣ Load a limited number of reviews
        fetch_start_time = time.time()
        with engine.connect() as connection:
            review_query = text("""
                SELECT id, comment
                FROM reviews
                WHERE comment IS NOT NULL
                ORDER BY id DESC
                LIMIT 200
            """)
            review_results = connection.execute(review_query).fetchall()

        if not review_results:
            return

        chunk_size = 4
        total_chunks = (len(review_results) + chunk_size - 1) // chunk_size
        
        system_prompt = """
You are a Restaurant Review Analysis Agent.

Task:
1. Identify ALL dishes mentioned.
2. Determine sentiment for EACH dish.
3. Count positive and negative mentions.

Return ONLY valid JSON:

{
  "dishes": [
    {
      "name": "Dish Name",
      "positive_mentions": 2,
      "negative_mentions": 0,
      "sentiment_score": 0.85
    }
  ]
}
"""
        prompts = []
        for i in range(0, len(review_results), chunk_size):
            chunk = review_results[i:i + chunk_size]
            reviews_text = "\n\n".join([f"Review #{row[0]}:\n{row[1]}" for row in chunk])
            final_prompt = system_prompt + "\n\nREVIEWS:\n\n" + reviews_text
            prompts.append(final_prompt)

        # Pre-load menu items for fast in-memory validation
        menu_items = []
        with engine.connect() as connection:
            menu_query = text("""
                SELECT 
                    m.id,
                    m.name AS item_name,
                    c.name AS item_category,
                    m.price AS item_price
                FROM menu_items m
                LEFT JOIN menu_categories c ON m.category_id = c.id
            """)
            menu_items = connection.execute(menu_query).fetchall()

        fetch_end_time = time.time()
        print(f"[TIME] Data Fetching Time: {fetch_end_time - fetch_start_time:.2f} seconds")

        # 2️⃣ Execute LLM calls and yield as they complete
        processing_start_time = time.time()
        tasks = [llm.ainvoke(prompt) for prompt in prompts]
        completed_count = 0
        
        for task in asyncio.as_completed(tasks):
            try:
                response = await task
                completed_count += 1
                
                json_match = re.search(r'\{[\s\S]*\}', response.content)
                if not json_match:
                    continue

                agent_analysis = json.loads(json_match.group())
                chunk_dishes = agent_analysis.get("dishes", [])
                
                valid_dishes = []
                for dish in chunk_dishes:
                    if dish.get("sentiment_score", 0) <= 0 or dish.get("positive_mentions", 0) <= dish.get("negative_mentions", 0):
                        continue # Skip non-positive right away
                        
                    dish_name_lower = dish["name"].lower()
                    if not dish_name_lower:
                        continue
                        
                    name_words = dish_name_lower.split()
                    
                    # Simple in-memory matching
                    best_match = None
                    for item in menu_items:
                        item_name_lower = str(item[1]).lower()
                        if len(name_words) == 1:
                            if name_words[0] in item_name_lower:
                                best_match = item
                                break
                        else:
                            if name_words[0] in item_name_lower and name_words[-1] in item_name_lower:
                                best_match = item
                                break
                    
                    if best_match:
                        valid_dishes.append({
                            "dish_name": best_match[1],
                            "category": best_match[2],
                            "price": float(best_match[3]) if isinstance(best_match[3], Decimal) else best_match[3],
                            "sentiment_score": dish.get("sentiment_score", 0),
                            "positive_mentions": dish.get("positive_mentions", 0),
                            "negative_mentions": dish.get("negative_mentions", 0)
                        })
                
                if valid_dishes:
                    # Sort dishes from this chunk
                    valid_dishes.sort(key=lambda x: (x["sentiment_score"], x["positive_mentions"]), reverse=True)
                    # Limit dishes from chunk if needed (e.g. up to 'limit' per chunk to avoid massive payload)
                    limited_dishes = valid_dishes[:limit]
                    
                    # Yield results in chunks of 4 at a time
                    response_start_time = time.time()
                    for i in range(0, len(limited_dishes), 4):
                        chunk = limited_dishes[i:i + 4]
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0.05)
                    response_end_time = time.time()
                    print(f"[TIME] Response Streaming Time (Chunk): {response_end_time - response_start_time:.2f} seconds")
            
            except Exception as e:
                import traceback
                traceback.print_exc()

        processing_end_time = time.time()
        print(f"[TIME] Total Data Processing & LLM Time: {processing_end_time - processing_start_time:.2f} seconds")
        print(f"[TIME] Total Execution Time: {time.time() - total_start_time:.2f} seconds")

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
