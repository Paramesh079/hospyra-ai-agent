from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from sql_agent import query_menu, semantic_search_similar_items
from review_analyzer import analyze_reviews_with_agent

app = FastAPI(name="hospyra", version="1.0.0", title="ai_menu_recommender", description="AI Menu Recommender for Hospyra")

@app.get("/user-preferences",tags=["ai-menu"])
async def run_query(prompt: str):
    return {
        "prompt": prompt,
        "result": semantic_search_similar_items(prompt)
    }
    
@app.get("/reviews-recommendations", tags=["ai-menu"])
async def get_agent_recommendations(limit: int = 20):
    """
    LLM-Based Positive Dish Recommendation (Streaming)
    """
    # Return a StreamingResponse using Server-Sent Events (SSE)
    return StreamingResponse(
        analyze_reviews_with_agent(limit=limit),
        media_type="text/event-stream"
    )


@app.get("/health",tags=["ai-menu"])
async def health():
    return {"status": "API is running"}
