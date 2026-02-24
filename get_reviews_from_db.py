import json
from sqlalchemy import text
from db import engine

async def get_db_recommendations(hotel_id: int, limit: int = 20):
    """
    Retrieves dish recommendations from the dish_review_analytics table, 
    prioritizing recently reviewed items by joining with the reviews table.
    Matches the SSE format of get_agent_recommendations.
    """
    try:
        query = text("""
            WITH recent_reviews AS (
                SELECT oi.menu_item_id, MAX(r.review_date) as max_review_date
                FROM reviews r
                JOIN order_items oi ON r.id = oi.order_id -- assuming we eventually link review to dish directly. if not we sort by hotel level review date
                WHERE r.hotel_id = :hotel_id
                GROUP BY oi.menu_item_id
            )
            SELECT 
                dra.menu_item_id as id,
                dra.hotel_id,
                dra.sentiment_score
            FROM 
                dish_review_analytics dra
            LEFT JOIN 
                recent_reviews rr ON dra.menu_item_id = rr.menu_item_id
            WHERE 
                dra.hotel_id = :hotel_id 
                AND dra.sentiment_score > 0
                AND dra.positive_mentions > dra.negative_mentions
            ORDER BY 
                rr.max_review_date DESC NULLS LAST,
                dra.sentiment_score DESC,
                dra.popularity_score DESC
            LIMIT :limit
        """)
        
        with engine.connect() as connection:
            results = connection.execute(query, {"hotel_id": hotel_id, "limit": limit}).fetchall()
            
        # Format like the original SSE stream: [{"id": 123}, {"id": 456}]
        formatted_chunk = [
            {
                "id": row.id,
                "hotel_id": row.hotel_id,
                "sentiment_score": row.sentiment_score
            } 
            for row in results
        ]
        
        yield f"data: {json.dumps(formatted_chunk)}\n\n"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
