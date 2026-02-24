from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from sql_agent import query_menu, semantic_search_similar_items
from review_analyzer import analyze_reviews_with_agent
from get_reviews_from_db import get_db_recommendations
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(name="hospyra", title="ai_menu_recommender")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8082",
        "http://127.0.0.1:8082",
        "http://192.168.4.112:8082",
        "http://localhost:3000",        # if using React default
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/user-preferences",tags=["ai-menu"])
async def run_query(user_id: int, hotel_id: int):
    return {
        "user_id": user_id,
        "hotel_id": hotel_id,
        "result": semantic_search_similar_items(str(user_id), hotel_id)
    }
    
@app.get("/reviews-recommendations", tags=["ai-menu"])
async def get_agent_recommendations(hotel_id: int, limit: int = 20):
    """
    LLM-Based Positive Dish Recommendation (Streaming)
    """
    # Return a StreamingResponse using Server-Sent Events (SSE)
    return StreamingResponse(
        analyze_reviews_with_agent(hotel_id=hotel_id, limit=limit),
        media_type="text/event-stream"
    )

@app.get("/reviews-recommendations-db", tags=["ai-menu"])
async def get_db_recommendations_endpoint(hotel_id: int, limit: int = 20):
    """
    Retrieves Top Dishes directly from dish_review_analytics database table.
    Prioritizes items by recent reviews and then by sentiment score.
    Returns as Streaming Responses (SSE).
    """
    return StreamingResponse(
        get_db_recommendations(hotel_id=hotel_id, limit=limit),
        media_type="text/event-stream"
    )


@app.get("/health",tags=["ai-menu"])
async def health():
    return {"status": "API is running"}
