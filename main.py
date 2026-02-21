from fastapi import FastAPI, HTTPException
from sql_agent import query_menu, semantic_search_similar_items
from review_analyzer import analyze_reviews_with_agent

app = FastAPI(name="hospyra", version="1.0.0", title="ai_menu_recommender", description="AI Menu Recommender for Hospyra")

@app.get("/user-preferences",tags=["ai-menu"])
async def run_query(prompt: str):
    return {
        "prompt": prompt,
        "result": semantic_search_similar_items(prompt)
    }
    
@app.get("/reviews-recommendations",tags=["ai-menu"])
async def get_agent_recommendations(limit: int = 20):
    """
    LLM-Based Positive Dish Recommendation

    Flow:
    - Reviews from PostgreSQL
    - LLM extracts dishes + sentiment
    - Validate against real menu
    - Keep only positive dishes
    - Sort by strongest positivity
    - Return top N
    """

    result = await analyze_reviews_with_agent(limit=limit)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "analysis_type": "LLM + DB Validated Positive Ranking",
        "total_reviews_analyzed": result["total_reviews_analyzed"],
        "recommendations_count": len(result["top_positive_recommendations"]),
        "recommendations": result["top_positive_recommendations"]
    }


@app.get("/health",tags=["ai-menu"])
async def health():
    return {"status": "API is running"}
