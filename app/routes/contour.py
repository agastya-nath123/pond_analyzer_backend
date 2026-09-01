from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/analyzeContour")
async def analyze_contour(
    contour_file: UploadFile = File(...)
):
    """
    Upload KML file
    """
    return {
        "filename": contour_file.filename,
        "message": "Contour file received"
    }
