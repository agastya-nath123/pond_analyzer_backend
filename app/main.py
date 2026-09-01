from fastapi import FastAPI

from app.routes.contour import router as contour_router

app = FastAPI(
    title="Pond Planning API",
    description="""
    Backend API for automated pond and catchment analysis.

    The backend accepts KML contour data, generates a Digital Elevation Model
    (DEM), performs hydrological analysis, identifies potential pond locations,
    and calculates the catchment area and related characteristics for selected
    ponds.
    """,
    version="1.0.0",
)

app.include_router(contour_router)

@app.get("/")
def root():
    """
    Return basic information about the Pond Analysis API.

    This endpoint serves as the root entry point of the backend API.
    It can be used to verify that the FastAPI application is running
    and accessible.

    Unlike the analysis endpoints, this endpoint does not perform any
    terrain, DEM, hydrological, or pond-related calculations.

    Returns:
        200:
            A JSON object containing a simple status message indicating
            that the API is operational.

    Example response:

        {
            "message": "Pond Analysis API is running"
        }
    """

    return {
        "message": "Pond Planning API is running"
    }
