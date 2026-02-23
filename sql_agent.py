import os
import json
from sqlalchemy import text
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from db import engine
from difflib import SequenceMatcher

load_dotenv()

# LLM
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "Hospyra")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
print(f"[LOG] Initializing Azure OpenAI LLM with deployment: {deployment_name}")
llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    api_version=api_version,
    temperature=0
)
print("[LOG] Azure OpenAI LLM initialized successfully")

def parse_sql_from_output(output: str):
    """
    Extract SQL query from LLM output
    """
    # Look for SQL between backticks if present
    if "```sql" in output:
        start_idx = output.find("```sql") + 6
        end_idx = output.find("```", start_idx)
        return output[start_idx:end_idx].strip()
    elif "```" in output:
        start_idx = output.find("```") + 3
        end_idx = output.find("```", start_idx)
        return output[start_idx:end_idx].strip()
    return output.strip()

def query_menu(user_prompt: str, hotel_id: int):
    """
    Converts user prompt into SQL query and executes it against PostgreSQL
    """
    print(f"\n[LOG] query_menu() called with prompt: {user_prompt}, hotel_id: {hotel_id}")
    
    system_prompt = f"""
                  You are a PostgreSQL Query Generator for a restaurant ordering system.

                  CRITICAL REQUIREMENT:
                  You MUST filter the queries for the current restaurant (ID: {hotel_id}). 
                  When querying `menu_items`, you MUST join with `menu_categories mc` and `menus mn` and include: `mn.hotel_id = {hotel_id}`
                  When querying `orders`, you MUST include: `orders.hotel_id = {hotel_id}`

                  DATABASE SCHEMA:
                  Table: menu_items
                  - id (INTEGER PRIMARY KEY)
                  - category_id (INTEGER) - Foreign Key to menu_categories.id
                  - name (VARCHAR) - Name of the food item
                  - description (TEXT)
                  - price (NUMERIC)
                  - is_available (BOOLEAN) - True if item can be ordered
                  - is_vegetarian (BOOLEAN)
                  - is_vegan (BOOLEAN)
                  - is_jain (BOOLEAN)
                  - spice_level (INTEGER)
                  - allergens (JSONB)

                  Table: menu_categories
                  - id (INTEGER PRIMARY KEY)
                  - menu_id (INTEGER) - Foreign Key to menus.id
                  - name (VARCHAR) - Category name (e.g., 'Starters', 'Main Course')
                  - priority (INTEGER)
                  - is_active (BOOLEAN)
                  - created_at (TIMESTAMP)
                  - updated_at (TIMESTAMP)

                  Table: menus
                  - id (INTEGER PRIMARY KEY)
                  - hotel_id (INTEGER) - Must exactly match {hotel_id}
                  - name (VARCHAR)

                  Table: orders
                  - id (INTEGER PRIMARY KEY)
                  - user_id (INTEGER) - Foreign Key to users.id. This identifies the customer.
                  - status (VARCHAR) - (e.g., 'PENDING', 'COMPLETED', 'CANCELLED')

                  Table: order_items
                  - id (INTEGER PRIMARY KEY)
                  - order_id (INTEGER) - Foreign Key to orders.id
                  - menu_item_id (INTEGER) - Foreign Key to menu_items.id
                  - price (NUMERIC)

                  Table: users
                  - id (INTEGER PRIMARY KEY)
                  - name (VARCHAR)

                  KEYWORD & SEMANTIC MAPPING:
                  - "veg", "vegetarian" -> Filter by menu_items.is_vegetarian = true
                  - "non-veg" -> Filter by menu_items.is_vegetarian = false
                  - "vegan" -> Filter by menu_items.is_vegan = true
                  - "jain" -> Filter by menu_items.is_jain = true
                  - Search by name: Use ILIKE on menu_items.name
                  - Search by category: Join with menu_categories and filter by name

                  CUSTOMER RECOMMENDATION SUPPORT:
                  If the user input is a USER ID (value in 'orders.user_id') or recommendations are requested for a user:
                  1. Identify Previously Ordered Items: Find the `id`, `name`, and `is_vegetarian` status of items the user has ordered before.
                  2. RANKING LAYERS:
                     - **LAYER 1 (TOP)**: Items the user HAS ordered before.
                     - **LAYER 2 (LAST LAYER)**: Items the user HAS NOT ordered before, but share the SAME `is_vegetarian` status as previous orders AND contain words from those previous item names in their own names.
                     - SEARCH SCOPE: This search is GLOBAL (across all categories).
                  
                  - SQL Strategy:
                    - Use a CTE (`user_history`) for previous orders.
                    - Join with `menu_items` and filter by matching `is_vegetarian`.
                    - **NEVER use SELECT DISTINCT** (it conflicts with ranking). Use **GROUP BY** instead.

                  - Example SQL for user 21 (Global keyword search with Ranking Layers):
                    ```sql
                    WITH user_history AS (
                        SELECT mi.id, mi.name, mi.is_vegetarian
                        FROM order_items oi
                        JOIN orders o ON oi.order_id = o.id
                        JOIN menu_items mi ON oi.menu_item_id = mi.id
                        WHERE o.user_id = 21 AND o.hotel_id = {hotel_id}
                    )
                    SELECT mi.name, mc.name as category, mi.price
                    FROM menu_items mi
                    JOIN menu_categories mc ON mi.category_id = mc.id
                    JOIN menus mn ON mc.menu_id = mn.id
                    JOIN user_history uh ON mi.is_vegetarian = uh.is_vegetarian
                    WHERE mi.is_available = true AND mn.hotel_id = {hotel_id}
                    GROUP BY mi.name, mc.name, mi.price, mi.id
                    ORDER BY 
                      MAX(CASE WHEN mi.id IN (SELECT id FROM user_history) THEN 1 ELSE 0 END) DESC, -- LAYER 1
                      MAX(CASE WHEN EXISTS (
                          SELECT 1 FROM user_history uh2 
                          WHERE mi.name ILIKE '%' || uh2.name || '%' 
                          OR uh2.name ILIKE '%' || mi.name || '%'
                      ) THEN 1 ELSE 0 END) DESC, -- LAYER 2 (LAST LAYER)
                      mi.name
                    LIMIT 100;
                    ```

                  - Query for Global Name Matches (Layer 2 focus):
                    ```sql
                    WITH history AS (SELECT DISTINCT name, is_vegetarian FROM menu_items WHERE id IN (SELECT menu_item_id FROM order_items oi JOIN orders o ON oi.order_id = o.id WHERE o.user_id = 7 AND o.hotel_id = {hotel_id}))
                    SELECT mi.name, mc.name as category, mi.price
                    FROM menu_items mi
                    JOIN menu_categories mc ON mi.category_id = mc.id
                    JOIN menus mn ON mc.menu_id = mn.id
                    JOIN history h ON mi.is_vegetarian = h.is_vegetarian
                    WHERE (mi.name ILIKE '%' || h.name || '%' OR h.name ILIKE '%' || mi.name || '%') AND mn.hotel_id = {hotel_id}
                    GROUP BY mi.name, mc.name, mi.price, mi.id
                    ORDER BY mi.name
                    LIMIT 100;
                    ```

                  CRITICAL RULES:
                  1. Return ONLY the SQL query.
                  2. ALWAYS use JOINs to connect orders, order_items, and menu_items.
                  3. ALWAYS use ILIKE for search.
                  4. Limit to 100.
                  5. No explanations outside of the SQL code block.
                  6. **NEVER USE SELECT DISTINCT** in recommendation queries (it conflicts with ranking). Use **GROUP BY** instead.
                  7. When ranking by non-selected expressions, use aggregate functions like `MAX()` in the ORDER BY clause.
                  8. Strictly follow the dietary preference (`is_vegetarian`) of the user's previous orders.


                  Example search: "cheese pizza"
                  ```sql
                  SELECT mi.name, mc.name as category, mi.price 
                  FROM menu_items mi 
                  JOIN menu_categories mc ON mi.category_id = mc.id
                  JOIN menus mn ON mc.menu_id = mn.id
                  WHERE mi.name ILIKE '%cheese%' AND mi.name ILIKE '%pizza%' AND mn.hotel_id = {hotel_id}
                  LIMIT 30;
                  ```"""
                
    final_prompt = system_prompt + "\n\nUser request: " + user_prompt
    print(f"[LOG] Invoking LLM for SQL Query generation...")
    
    try:
        response = llm.invoke(final_prompt)
        print(f"[LOG] LLM Response: {response.content}")
        
        sql_query = parse_sql_from_output(response.content)
        
        # Remove trailing semicolon if present as SQLAlchemy text() can be picky
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
            
        print(f"[LOG] Executing SQL Query: {sql_query}")
        
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            results = [dict(row) for row in result.mappings()]
            
        print(f"[LOG] Found {len(results)} results")
        return results

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return {"error": "Internal Server Error", "details": str(e)}


def get_user_order_history(user_id: str, hotel_id: int):
    """
    Retrieves all items previously ordered by a specific user from the database.
    
    Args:
        user_id: The user ID to retrieve order history for
        hotel_id: The restaurant ID to filter on
    
    Returns:
        List of item names the user has previously ordered
    """
    print(f"[LOG] Retrieving order history for user: {user_id} and hotel: {hotel_id}")
    
    try:
        query = """
        SELECT DISTINCT mi.name
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN menu_items mi ON oi.menu_item_id = mi.id
        WHERE o.user_id = :user_id AND o.hotel_id = :hotel_id
        ORDER BY mi.name
        """
        
        with engine.connect() as connection:
            result = connection.execute(text(query), {"user_id": int(user_id), "hotel_id": hotel_id})
            order_history = [row[0].strip() for row in result.fetchall()]
        
        print(f"[LOG] Found {len(order_history)} distinct items in user's order history")
        for item in order_history:
            print(f"[LOG]   - {item}")
        
        return order_history
    
    except Exception as e:
        print(f"[ERROR] Error retrieving order history: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return []


def semantic_search_similar_items(user_prompt: str, hotel_id: int, similarity_threshold: float = 0.4):
    """
    Performs semantic search on query_menu results against user's order history.
    Retrieves all items the user has previously ordered and compares them with query_menu results.
    Returns items that match any of the user's previous orders based on semantic similarity.
    
    Args:
        user_prompt: The user ID or prompt to pass to query_menu
        similarity_threshold: Minimum similarity score (0.0 to 1.0) to include results
    
    Returns:
        List of items with similar names to any of the user's previous orders, sorted by similarity
    """
    print(f"\n[LOG] semantic_search_similar_items() called with prompt: {user_prompt}")
    
    try:
        # Get user's order history (all items they've previously ordered)
        user_order_history = get_user_order_history(user_prompt, hotel_id)
        
        if not user_order_history:
            print("[LOG] No order history found for this user")
            return []
        
        # Get recommendations from query_menu
        results = query_menu(user_prompt, hotel_id)
        
        if not results or isinstance(results, dict) and "error" in results:
            print("[LOG] No results from query_menu or error occurred")
            return results
        
        if len(results) == 0:
            print("[LOG] No results found from query_menu")
            return []
        
        print(f"[LOG] Comparing {len(results)} menu items against {len(user_order_history)} user's previous orders")
        
        # Perform semantic search: compare each menu item against ALL user's previous orders
        similar_items = []
        
        for menu_item in results:
            item_name = menu_item.get("name", "").strip()
            
            # Find the maximum similarity score against any of the user's previous orders
            max_similarity = 0.0
            matched_history_item = None
            
            for history_item in user_order_history:
                similarity_ratio = SequenceMatcher(None, history_item.lower(), item_name.lower()).ratio()
                
                if similarity_ratio > max_similarity:
                    max_similarity = similarity_ratio
                    matched_history_item = history_item
            
            # Include items that meet the similarity threshold
            if max_similarity >= similarity_threshold:
                item_with_score = menu_item.copy()
                item_with_score["similarity_score"] = round(max_similarity, 2)
                item_with_score["matched_with"] = matched_history_item
                similar_items.append(item_with_score)
                print(f"[LOG] ✓ '{item_name}' matches '{matched_history_item}' with score: {max_similarity:.2f}")
        
        print(f"[LOG] Found {len(similar_items)} semantically similar items")
        
        # Sort by similarity score in descending order
        similar_items.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_items
    
    except Exception as e:
        print(f"[ERROR] An error occurred in semantic_search_similar_items: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return {"error": "Internal Server Error", "details": str(e)}

