import re
import json
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
    LLM-based review analysis.
    Flow:
    1. Load reviews from PostgreSQL
    2. LLM extracts dishes + sentiment
    3. Validate dish names against menu table
    4. Keep only positive dishes
    5. Sort by strongest positivity
    6. Return top N
    """

    try:
        # 1️⃣ Load reviews from PostgreSQL
        with engine.connect() as connection:
            review_query = text("""
                SELECT id, comment
                FROM reviews
                WHERE comment IS NOT NULL
            """)
            review_results = connection.execute(review_query).fetchall()

        if not review_results:
            return {"error": "No reviews found"}

        # Chunk reviews to avoid context limits (e.g., 20 at a time)
        chunk_size = 20
        extracted_dishes = {}

        for i in range(0, len(review_results), chunk_size):
            chunk = review_results[i:i + chunk_size]
            reviews_text = "\n\n".join([
                f"Review #{row[0]}:\n{row[1]}"
                for row in chunk
            ])
            print(f"DEBUG: Processing chunk {i//chunk_size + 1} ({len(chunk)} reviews)...")

            # 2️⃣ LLM Prompt
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

            final_prompt = system_prompt + "\n\nREVIEWS:\n\n" + reviews_text
            print(f"DEBUG: Connecting to LLM ({deployment_name}) for chunk {i//chunk_size + 1}...")
            response = llm.invoke(final_prompt)
            print(f"DEBUG: Raw LLM response received for chunk {i//chunk_size + 1}:")
            print(response.content)

            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if not json_match:
                print(f"DEBUG: Invalid LLM response for chunk {i//chunk_size + 1}. Skipping.")
                continue

            try:
                agent_analysis = json.loads(json_match.group())
                chunk_dishes = agent_analysis.get("dishes", [])
                
                # Aggregate results
                for dish in chunk_dishes:
                    name = dish.get("name", "").lower()
                    if not name:
                        continue
                    if name in extracted_dishes:
                        extracted_dishes[name]["positive_mentions"] += dish.get("positive_mentions", 0)
                        extracted_dishes[name]["negative_mentions"] += dish.get("negative_mentions", 0)
                        # Very simple weighted average for sentiment mapping across chunks
                        old_weight = extracted_dishes[name]["positive_mentions"] + extracted_dishes[name]["negative_mentions"]
                        new_weight = dish.get("positive_mentions", 0) + dish.get("negative_mentions", 0)
                        if old_weight + new_weight > 0:
                            extracted_dishes[name]["sentiment_score"] = (
                                (extracted_dishes[name]["sentiment_score"] * old_weight) + 
                                (dish.get("sentiment_score", 0) * new_weight)
                            ) / (old_weight + new_weight)
                    else:
                        extracted_dishes[name] = dish

                print(f"DEBUG: Extracted {len(chunk_dishes)} dishes from chunk {i//chunk_size + 1}.")
            except json.JSONDecodeError:
                print(f"DEBUG: JSON parsing error for chunk {i//chunk_size + 1}. Skipping.")
                continue

        extracted_dishes = list(extracted_dishes.values())
        print(f"DEBUG: Total extracted unique dishes across all chunks: {len(extracted_dishes)}")

        if not extracted_dishes:
            print("DEBUG: Final aggregated array is empty.")
            return {"error": "No dishes identified by LLM across all chunks"}

        # 3️⃣ Validate against real menu FIRST
        valid_dishes = []

        with engine.connect() as connection:
            for dish in extracted_dishes:
                # Pre-process the dish name for wider matching (e.g., Mutton Maasala -> %mutton%masala%)
                name_words = dish["name"].split()
                if len(name_words) == 1:
                    like_pattern = f"%{name_words[0]}%"
                else:
                    # e.g., "Mutton Maasala" -> "%mutton%masala%" but handle max 2 words for safe matching
                    like_pattern = f"%{name_words[0]}%{name_words[-1]}%"

                query = text("""
                    SELECT 
                        m.id,
                        m.name AS item_name,
                        c.name AS item_category,
                        m.price AS item_price,
                        NULL AS item_rating,
                        NULL AS item_taste,
                        NULL AS item_special
                    FROM menu_items m
                    LEFT JOIN menu_categories c ON m.category_id = c.id
                    WHERE LOWER(m.name) LIKE LOWER(:dish_name_pattern)
                    LIMIT 1
                """)

                result = connection.execute(
                    query,
                    {"dish_name_pattern": like_pattern}
                ).fetchone()

                if result:
                    print(f"DEBUG: Matched dish '{dish['name']}' in DB: {result}")
                    valid_dishes.append({
                        "dish_name": result[1],
                        "category": result[2],
                        "price": result[3],
                        "rating": result[4],
                        "taste": result[5],
                        "special": result[6],
                        "sentiment_score": dish.get("sentiment_score", 0),
                        "positive_mentions": dish.get("positive_mentions", 0),
                        "negative_mentions": dish.get("negative_mentions", 0)
                    })
                else:
                    print(f"DEBUG: Failed to match dish '{dish['name']}' to any menu item.")

        if not valid_dishes:
            return {"error": "No valid menu items matched"}

        # 4️⃣ Keep only positive dishes
        positive_dishes = [
            dish for dish in valid_dishes
            if dish["sentiment_score"] > 0
            and dish["positive_mentions"] > dish["negative_mentions"]
        ]

        if not positive_dishes:
            return {"error": "No positive dishes found"}

        # 5️⃣ Sort by strongest positivity
        positive_dishes = sorted(
            positive_dishes,
            key=lambda x: (x["sentiment_score"], x["positive_mentions"]),
            reverse=True
        )

        # 6️⃣ Return top N
        return {
            "total_reviews_analyzed": len(review_results),
            "top_positive_recommendations": positive_dishes[:limit]
        }

    except Exception as e:
        return {"error": str(e)}