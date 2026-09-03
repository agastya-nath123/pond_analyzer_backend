from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)
from pydantic import BaseModel

from app.services.contour_analyzer import (
    analyze_ponds,
    find_catchment
)
from pathlib import Path

router = APIRouter()


class Pond(BaseModel):
    pond_id: int
    pond_area_ha: float
    max_depth_m: float
    volume_m3: float


class AnalyzeContourResponse(BaseModel):
    ponds: list[Pond]


class SpillPoint(BaseModel):
    easting: float
    northing: float
    elevation_m: float


class CatchmentResponse(BaseModel):
    pond_id: int
    spill: SpillPoint
    flow_accumulation_cells: int
    catchment_area_m2: float
    catchment_area_ha: float
    catchment_pond_ratio: float

# -------------------------
# /analyzeContour
# -------------------------

@router.post("/analyzeContour",
             response_model=AnalyzeContourResponse
)
async def analyze_contour(
    contour_map: UploadFile = File(...)
):
    """
    Analyze a KML file containing topographic contour lines and identify
    potential pond locations based on terrain depressions derived from
    the contour data.

    The uploaded KML file is expected to contain contour lines represented
    as KML LineString geometries. Each contour's elevation is read from the
    corresponding Placemark <name> element. The coordinates are interpreted
    as WGS84 geographic coordinates (EPSG:4326) and are subsequently
    transformed into UTM Zone 44N (EPSG:32644) so that all terrain
    calculations can be performed using metric distances and areas.

    Processing performed by this endpoint:

    1. The KML file is parsed and all valid contour lines and their
       elevations are extracted.

    2. The contour coordinates are projected from geographic coordinates
       (longitude/latitude) into a metric projected coordinate system.

    3. Points are sampled along the contour lines at approximately
       5-metre intervals.

    4. A Digital Elevation Model (DEM) is generated from the sampled
       contour points using linear interpolation at the configured
       raster resolution.

    5. Depressions in the DEM are filled using a priority-flood
       depression-filling algorithm. This produces a filled terrain
       surface representing the elevation water would reach before
       spilling out of a depression.

    6. Flat regions created by depression filling are resolved by applying
       a very small deterministic elevation gradient. This allows the
       subsequent flow-direction calculation to determine a consistent
       drainage direction across otherwise flat terrain.

    7. D8 flow direction is calculated from the processed DEM.

    8. Flow accumulation is calculated to determine the number of
       upstream raster cells contributing flow to each cell.

    9. The difference between the filled DEM and the original DEM is
       calculated. This represents the depth to which each cell would
       need to be filled before water could escape the depression.

    10. Cells having a fill depth of at least 1 metre are classified as
        potential depression cells.

    11. Connected depression cells are grouped into individual regions.
        Depressions touching the boundary of the DEM are discarded because
        they may represent terrain extending outside the available
        contour-data coverage rather than enclosed pond locations.

    12. For each remaining depression, its area, maximum depth, spill point,
        catchment area, and approximate water-storage volume are calculated.

    13. The resulting depression information is converted into a list of
        potential pond candidates.

    Response:

        The endpoint returns a JSON object containing a list of detected
        pond candidates. Each candidate contains:

        - pond_id:
            The integer identifier assigned to the connected depression
            region by scipy.ndimage.label(). This identifier is used by
            the /findCatchment endpoint to select a particular pond.

        - pond_area_ha:
            The estimated horizontal area of the detected depression in
            hectares. The area is calculated from the number of raster
            cells belonging to the depression and the configured DEM
            resolution.

        - max_depth_m:
            The maximum calculated depression depth in metres. This is
            obtained from the difference between the filled DEM and the
            original DEM.

        - volume_m3:
            The estimated water-storage volume of the depression in cubic
            metres. It is calculated by summing the depression depth of
            each valid depression cell and multiplying by the area of a
            DEM cell.

    Request:
        Content-Type: multipart/form-data

        file:
            A KML file containing contour lines and their elevations.

    Returns:
        200:
            JSON object containing the detected pond candidates.

        400:
            The uploaded KML data is invalid or cannot be processed.

        500:
            An unexpected error occurs during DEM generation,
            hydrological analysis, or pond detection.

    Example response:

        {
            "ponds": [
                {
                    "pond_id": 67,
                    "pond_area_ha": 2.1925,
                    "max_depth_m": 2.9325,
                    "volume_m3": 46492.68
                }
            ]
        }

    Notes:
        The pond_id values are labels generated from the connected
        depression regions. They are not permanent geographic identifiers.
        They should therefore be treated as identifiers for the specific
        KML/DEM analysis rather than globally unique pond IDs.

        The endpoint currently performs the complete terrain and
        hydrological processing for each request. Intermediate DEM,
        flow-direction, flow-accumulation, and depression-label data are
        not persisted between requests.
    """

    file_path = Path("uploads") / contour_map.filename

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        results = analyze_ponds(
            file_path,
            resolution=5
        )

        return {
            "ponds": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# -------------------------
# /findCatchment
# -------------------------

@router.post("/findCatchment",
             response_model=CatchmentResponse
)
async def find_catchment_endpoint(
    contour_map: UploadFile = File(...),
    pond_id: int = Form(...)
):
    """
    Calculate the contributing catchment area for a previously detected
    pond using the pond's identifier and the same KML contour data used
    for pond detection.

    This endpoint is intended to be used after /analyzeContour has been
    executed. The client selects one of the pond_id values returned by
    /analyzeContour and submits that identifier together with the original
    KML file.

    The endpoint reconstructs the terrain and hydrological model from the
    supplied KML file so that the same depression-labeling process can be
    used to locate the requested pond.

    Processing performed by this endpoint:

    1. The uploaded KML file is parsed to extract contour lines and their
       associated elevations.

    2. The contour coordinates are transformed from WGS84 geographic
       coordinates (EPSG:4326) into UTM Zone 44N (EPSG:32644).

    3. Points are sampled along the contour lines at approximately
       5-metre intervals.

    4. A Digital Elevation Model (DEM) is generated from the sampled
       contour points using linear interpolation.

    5. The DEM is processed using depression filling to produce a filled
       terrain surface.

    6. Flat regions in the filled terrain are resolved so that flow
       directions can be determined consistently.

    7. D8 flow direction is calculated for the processed terrain.

    8. Flow accumulation is calculated for every raster cell. The
       accumulation value represents the number of upstream raster cells
       contributing flow to that cell.

    9. Depression depth is calculated as the difference between the filled
       DEM and the original DEM.

    10. Cells with a fill depth of at least 1 metre are classified as
        depression cells.

    11. Connected depression cells are labeled using
        scipy.ndimage.label(). The supplied pond_id is used to select
        the corresponding connected depression region.

    12. The area of the selected depression is calculated from its number
        of raster cells and the configured raster resolution.

    13. A candidate spill point is identified along the boundary of the
        selected depression. The spill elevation is estimated using the
        depression-filled terrain, and the boundary cell whose elevation
        is closest to the calculated spill elevation is selected as the
        spill point.

    14. The geographic position of the spill point is converted to the
        corresponding raster cell.

    15. The flow-accumulation value at the spill cell is obtained. This
        value represents the number of upstream cells contributing runoff
        to the selected spill location.

    16. The accumulated cell count is converted into an area using the
        DEM cell area.

    17. The catchment area is reported both in square metres and hectares.

    18. A catchment-to-pond-area ratio is calculated by dividing the
        calculated catchment area by the area of the selected pond
        depression.

    Request:

        Content-Type: multipart/form-data

        file:
            The KML file containing the contour data. It should be the
            same contour dataset from which the requested pond_id was
            obtained.

        pond_id:
            Integer identifier of the pond returned by /analyzeContour.
            This identifies the connected depression whose catchment is
            being calculated.

    Returns:
        200:
            JSON object containing the selected pond's spill point,
            flow accumulation, catchment area, and catchment-to-pond
            ratio.

        404:
            The supplied pond_id does not correspond to a depression
            identified in the supplied KML data.

        400:
            The uploaded KML data or supplied parameters are invalid.

        500:
            An unexpected error occurs during terrain reconstruction
            or catchment analysis.

    Example request:

        multipart/form-data

        file = contours.kml
        pond_id = 67

    Example response:

        {
            "pond_id": 67,
            "spill": {
                "easting": 382145.0,
                "northing": 2345678.0,
                "elevation_m": 142.35
            },
            "flow_accumulation_cells": 53788,
            "catchment_area_m2": 1344700.0,
            "catchment_area_ha": 134.47,
            "catchment_pond_ratio": 61.33
        }

    Notes:
        The pond_id is derived from the connected-component labeling of
        the depression mask. It is therefore meaningful only relative to
        the same KML dataset and the same DEM-processing configuration.

        The catchment area is derived from raster flow accumulation.
        Consequently, its accuracy depends on the quality and resolution
        of the generated DEM, the contour spacing, interpolation quality,
        and the correctness of the input contour elevations.

        The spill point returned by this endpoint is a candidate spill
        location derived from the raster depression boundary. It should
        therefore be treated as an estimated hydrological spill point
        rather than a surveyed field location.

        The endpoint currently reconstructs the DEM and hydrological
        model for each request rather than reusing the intermediate
        results produced by /analyzeContour.
    """

    file_path = Path("uploads") / contour_map.filename

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        result = find_catchment(
            file_path,
            pond_id,
            resolution=5
        )

        return {
            "pond_id": pond_id,
            **result
        }

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
