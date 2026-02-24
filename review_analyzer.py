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


async def analyze_reviews_with_agent(hotel_id: int, limit: int = 20):
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
                WHERE comment IS NOT NULL AND hotel_id = :hotel_id
                ORDER BY id DESC
            """)
            review_results = connection.execute(review_query, {"hotel_id": hotel_id}).fetchall()

        if not review_results:
            return

        chunk_size = 4
        total_chunks = (len(review_results) + chunk_size - 1) // chunk_size
        
        system_prompt = """
You are a Restaurant Review Analysis Agent.

Task:
1. Identify ALL dishes mentioned.
2. Determine sentiment for EACH dish.
3. Count positive, negative, and neutral mentions.
4. Extract top keywords (e.g., "spicy", "cold", "perfect") and taste descriptors.
5. Provide a short summary of what people are saying about the dish.

Return ONLY valid JSON:

{
  "dishes": [
    {
      "name": "Dish Name",
      "positive_mentions": 2,
      "negative_mentions": 0,
      "neutral_mentions": 1,
      "sentiment_score": 0.85,
      "top_keywords": ["crispy", "fresh"],
      "taste_descriptors": ["salty", "rich"],
      "summary": "People loved the crispy texture but a few found it slightly too salty."
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
                LEFT JOIN menus mn ON c.menu_id = mn.id
                WHERE mn.hotel_id = :hotel_id
            """)
            menu_items = connection.execute(menu_query, {"hotel_id": hotel_id}).fetchall()

        fetch_end_time = time.time()
        print(f"[TIME] Data Fetching Time: {fetch_end_time - fetch_start_time:.2f} seconds")

        # 2️⃣ Execute LLM calls and yield as they complete
        processing_start_time = time.time()
        tasks = [llm.ainvoke(prompt) for prompt in prompts]
        completed_count = 0
        aggregated_dishes = {}
        
        for task in asyncio.as_completed(tasks):
            try:
                response = await task
                completed_count += 1
                
                json_match = re.search(r'\{[\s\S]*\}', response.content)
                if not json_match:
                    continue

                agent_analysis = json.loads(json_match.group())
                chunk_dishes = agent_analysis.get("dishes", [])
                
                for dish in chunk_dishes:
                    dish_name_lower = dish.get("name", "").lower()
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
                        item_id = best_match[0]
                        if item_id not in aggregated_dishes:
                            aggregated_dishes[item_id] = {
                                "id": item_id,
                                "dish_name": best_match[1],
                                "category": best_match[2],
                                "price": float(best_match[3]) if isinstance(best_match[3], Decimal) else best_match[3],
                                "sentiment_score": 0.0,
                                "positive_mentions": 0,
                                "negative_mentions": 0,
                                "neutral_mentions": 0,
                                "top_keywords": [],
                                "taste_descriptors": [],
                                "summary_parts": [],
                                "occurrence_count": 0
                            }
                        
                        ad = aggregated_dishes[item_id]
                        ad["positive_mentions"] += dish.get("positive_mentions", 0)
                        ad["negative_mentions"] += dish.get("negative_mentions", 0)
                        ad["neutral_mentions"] += dish.get("neutral_mentions", 0)
                        ad["sentiment_score"] += dish.get("sentiment_score", 0)
                        ad["occurrence_count"] += 1
                        
                        if dish.get("top_keywords"):
                            ad["top_keywords"].extend(dish.get("top_keywords"))
                        if dish.get("taste_descriptors"):
                            ad["taste_descriptors"].extend(dish.get("taste_descriptors"))
                        if dish.get("summary"):
                            ad["summary_parts"].append(dish.get("summary"))
            
            except Exception as e:
                import traceback
                traceback.print_exc()

        # Process aggregated data and upsert to database
        print(f"[LOG] Total reviews processed: {len(review_results)}")
        
        all_valid_dishes = []
        for item_id, ad in aggregated_dishes.items():
            if ad["occurrence_count"] > 0:
                ad["sentiment_score"] = ad["sentiment_score"] / ad["occurrence_count"]
            
            total_mentions = ad["positive_mentions"] + ad["negative_mentions"] + ad["neutral_mentions"]
            ad["mention_count"] = total_mentions
            ad["popularity_score"] = min(total_mentions * 5.0, 100.0)
            if total_mentions > 0:
                ad["hype_score"] = ((ad["positive_mentions"] - ad["negative_mentions"]) / total_mentions) * 100.0
            else:
                ad["hype_score"] = 0.0
            ad["confidence_score"] = min(total_mentions * 10.0, 100.0)
            ad["final_summary"] = " ".join(ad["summary_parts"])[:500]
            
            ad["top_keywords"] = list(set(ad["top_keywords"]))[:10]
            ad["taste_descriptors"] = list(set(ad["taste_descriptors"]))[:10]
            
            all_valid_dishes.append(ad)

        if all_valid_dishes:
            with engine.begin() as connection:
                for ad in all_valid_dishes:
                    upsert_query = text("""
                        INSERT INTO dish_review_analytics (
                            menu_item_id, hotel_id, mention_count, positive_mentions, 
                            negative_mentions, neutral_mentions, sentiment_score, 
                            popularity_score, hype_score, confidence_score, 
                            llm_summary, top_keywords, taste_descriptors, processed_review_count,
                            last_processed_at
                        ) VALUES (
                            :menu_item_id, :hotel_id, :mention_count, :positive_mentions,
                            :negative_mentions, :neutral_mentions, :sentiment_score,
                            :popularity_score, :hype_score, :confidence_score,
                            :llm_summary, :top_keywords, :taste_descriptors, :processed_review_count,
                            CURRENT_TIMESTAMP
                        ) ON CONFLICT (menu_item_id) DO UPDATE SET
                            mention_count = EXCLUDED.mention_count,
                            positive_mentions = EXCLUDED.positive_mentions,
                            negative_mentions = EXCLUDED.negative_mentions,
                            neutral_mentions = EXCLUDED.neutral_mentions,
                            sentiment_score = EXCLUDED.sentiment_score,
                            popularity_score = EXCLUDED.popularity_score,
                            hype_score = EXCLUDED.hype_score,
                            confidence_score = EXCLUDED.confidence_score,
                            llm_summary = EXCLUDED.llm_summary,
                            top_keywords = EXCLUDED.top_keywords,
                            taste_descriptors = EXCLUDED.taste_descriptors,
                            processed_review_count = EXCLUDED.processed_review_count,
                            last_processed_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                    """)
                    connection.execute(upsert_query, {
                        "menu_item_id": ad["id"],
                        "hotel_id": hotel_id,
                        "mention_count": ad["mention_count"],
                        "positive_mentions": ad["positive_mentions"],
                        "negative_mentions": ad["negative_mentions"],
                        "neutral_mentions": ad["neutral_mentions"],
                        "sentiment_score": ad["sentiment_score"],
                        "popularity_score": ad["popularity_score"],
                        "hype_score": ad["hype_score"],
                        "confidence_score": ad["confidence_score"],
                        "llm_summary": ad["final_summary"],
                        "top_keywords": json.dumps(ad["top_keywords"]),
                        "taste_descriptors": json.dumps(ad["taste_descriptors"]),
                        "processed_review_count": len(review_results)
                    })

            # Sort combined dishes for recommendations stream (only positive ones)
            positive_dishes = [d for d in all_valid_dishes if d["sentiment_score"] > 0 and d["positive_mentions"] > d["negative_mentions"]]
            positive_dishes.sort(key=lambda x: (x.get("sentiment_score", 0), x.get("positive_mentions", 0)), reverse=True)
            limited_dishes = positive_dishes[:limit]
            
            formatted_chunk = [{"id": item["id"]} for item in limited_dishes]
            yield f"data: {json.dumps(formatted_chunk)}\n\n"
            
        processing_end_time = time.time()
        print(f"[TIME] Total Data Processing & LLM Time: {processing_end_time - processing_start_time:.2f} seconds")
        print(f"[TIME] Total Execution Time: {time.time() - total_start_time:.2f} seconds")

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"{json.dumps({'type': 'error', 'message': str(e)})}\n\n"
