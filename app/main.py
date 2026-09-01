from fastapi import FastAPI

from app.routes.contour import router as contour_router

app = FastAPI(
    title="Pond Planning API",
    description="API for contour and catchment analysis",
    version="1.0.0",
)

app.include_router(contour_router)

@app.get("/")
def root():
    """
    Sample
    """
    return {
        "message": "Pond Planning API is running"
    }
