from pathlib import Path
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pyproj import Transformer
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.interpolate import griddata
import heapq
from scipy.ndimage import label
from scipy.ndimage import distance_transform_edt
from collections import deque
import rasterio
from rasterio.transform import from_origin
if not hasattr(np, "in1d"):
    np.in1d = np.isin
from pysheds.grid import Grid

KML_NAMESPACE = {
    "kml": "http://www.opengis.net/kml/2.2"
}


def parse_kml(file_path: Path):
    """
    Parse a KML file and extract contour lines and elevations.
    """

    tree = ET.parse(file_path)
    root = tree.getroot()

    contours = []

    placemarks = root.findall(
        ".//kml:Placemark",
        KML_NAMESPACE
    )

    for placemark in placemarks:

        # Get elevation from <name>
        name = placemark.find(
            "kml:name",
            KML_NAMESPACE
        )

        if name is None or name.text is None:
            continue

        try:
            elevation = float(name.text.strip())
        except ValueError:
            continue

        # Get LineString
        coordinates = placemark.find(
            ".//kml:LineString/kml:coordinates",
            KML_NAMESPACE
        )

        if coordinates is None or coordinates.text is None:
            continue

        points = []

        for coordinate in coordinates.text.split():

            values = coordinate.split(",")

            if len(values) < 2:
                continue

            longitude = float(values[0])
            latitude = float(values[1])

            points.append(
                (longitude, latitude)
            )

        if not points:
            continue

        contours.append({
            "elevation": elevation,
            "coordinates": points
        })

    return contours

def project_contours(contours):
    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32644",
        always_xy=True
    )

    projected_contours = []

    for contour in contours:

        projected_coordinates = []

        for longitude, latitude in contour["coordinates"]:

            x, y = transformer.transform(
                longitude,
                latitude
            )

            projected_coordinates.append(
                (x, y)
            )

        projected_contours.append({
            "elevation": contour["elevation"],
            "coordinates": projected_coordinates
        })

    return projected_contours

def get_extent(contours):

    all_points = [
        point
        for contour in contours
        for point in contour["coordinates"]
    ]

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "width_m": max(xs) - min(xs),
        "height_m": max(ys) - min(ys)
    }

def contour_points(contours, spacing=5):
    """
    Sample points along contour lines every `spacing` metres.
    """

    points = []
    elevations = []

    for contour in contours:

        line = LineString(
            contour["coordinates"]
        )

        elevation = contour["elevation"]

        distance = 0

        while distance <= line.length:

            point = line.interpolate(distance)

            points.append(
                (point.x, point.y)
            )

            elevations.append(elevation)

            distance += spacing

    return points, elevations

def create_dem(points, elevations, resolution=5):

    points = np.asarray(points)
    elevations = np.asarray(elevations)

    min_x = points[:, 0].min()
    max_x = points[:, 0].max()

    min_y = points[:, 1].min()
    max_y = points[:, 1].max()

    x = np.arange(
        min_x,
        max_x,
        resolution
    )

    y = np.arange(
        max_y,
        min_y,
        -resolution
    )

    grid_x, grid_y = np.meshgrid(
        x,
        y
    )

    dem = griddata(
        points,
        elevations,
        (grid_x, grid_y),
        method="linear"
    )

    return grid_x, grid_y, dem

def calculate_flow_direction(dem, resolution):
    """
    Calculate D8 flow direction.

    Each cell flows toward the neighboring cell
    with the steepest downward slope.

    Returns:
        flow_direction: array with values 0-7
        -1 means no valid downhill neighbor
    """

    rows, cols = dem.shape

    flow_direction = np.full(
        dem.shape,
        -1,
        dtype=np.int8
    )

    #              row   col
    neighbors = [
        ( 1, -1),  # NW
        ( 1,  0),  # N
        ( 1,  1),  # NE
        ( 0, -1),  # W
        ( 0,  1),  # E
        (-1, -1),  # SW
        (-1,  0),  # S
        (-1,  1),  # SE
    ]

    for row in range(1, rows - 1):

        for col in range(1, cols - 1):

            current = dem[row, col]

            if np.isnan(current):
                continue

            best_direction = -1
            best_slope = 0

            for direction, (dr, dc) in enumerate(neighbors):

                neighbor = dem[
                    row + dr,
                    col + dc
                ]

                if np.isnan(neighbor):
                    continue

                distance = resolution

                if dr != 0 and dc != 0:
                    distance = resolution * np.sqrt(2)

                slope = (
                    current - neighbor
                ) / distance

                if slope > best_slope:
                    best_slope = slope
                    best_direction = direction

            flow_direction[row, col] = best_direction

    return flow_direction

def find_depressions(dem):
    rows, cols = dem.shape

    depressions = []

    for row in range(1, rows - 1):
        for col in range(1, cols - 1):

            current = dem[row, col]

            if np.isnan(current):
                continue

            neighbors = dem[
                row - 1:row + 2,
                col - 1:col + 2
            ]

            valid_neighbors = neighbors[
                ~np.isnan(neighbors)
            ]

            # Remove the current cell itself
            valid_neighbors = valid_neighbors[
                valid_neighbors != current
            ]

            if len(valid_neighbors) == 0:
                continue

            if current < np.min(valid_neighbors):
                depressions.append(
                    (row, col, current)
                )

    return depressions

def fill_depressions(dem):
    """
    Fill enclosed depressions in a DEM using a priority-flood algorithm.

    NaN cells are treated as outside the terrain domain.

    The original DEM is not modified.
    """

    filled = dem.copy()

    rows, cols = dem.shape

    visited = np.zeros(
        dem.shape,
        dtype=bool
    )

    priority_queue = []

    # Add valid boundary cells to the queue.
    for row in range(rows):
        for col in range(cols):

            if (
                row == 0
                or row == rows - 1
                or col == 0
                or col == cols - 1
            ):

                if not np.isnan(dem[row, col]):

                    heapq.heappush(
                        priority_queue,
                        (
                            dem[row, col],
                            row,
                            col
                        )
                    )

                    visited[row, col] = True

    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1)
    ]

    while priority_queue:

        elevation, row, col = heapq.heappop(
            priority_queue
        )

        for dr, dc in neighbors:

            nr = row + dr
            nc = col + dc

            if not (
                0 <= nr < rows
                and 0 <= nc < cols
            ):
                continue

            if visited[nr, nc]:
                continue

            if np.isnan(dem[nr, nc]):
                continue

            visited[nr, nc] = True

            neighbor_elevation = dem[nr, nc]

            # Raise the neighbor if it is below
            # the current spill elevation.
            filled[nr, nc] = max(
                neighbor_elevation,
                elevation
            )

            heapq.heappush(
                priority_queue,
                (
                    filled[nr, nc],
                    nr,
                    nc
                )
            )

    return filled

def resolve_flats(dem, filled_dem):
    """
    Resolve flat regions in a depression-filled DEM.

    For each connected region of approximately equal elevation:
      1. Find cells on the flat that touch lower terrain.
      2. Compute distance from every flat cell to those outlet cells.
      3. Add a tiny deterministic elevation gradient that decreases
         toward the outlet.

    The perturbation is computational only.
    """

    resolved = filled_dem.copy()

    valid = ~np.isnan(filled_dem)

    rows, cols = filled_dem.shape

    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]

    # ---------------------------------------------------------
    # Find cells that belong to a flat.
    #
    # A cell is flat if it has at least one neighboring cell
    # at essentially the same elevation.
    # ---------------------------------------------------------

    flat = np.zeros_like(
        filled_dem,
        dtype=bool
    )

    for row in range(rows):
        for col in range(cols):

            if not valid[row, col]:
                continue

            current = filled_dem[row, col]

            for dr, dc in neighbors:

                nr = row + dr
                nc = col + dc

                if not (
                    0 <= nr < rows
                    and 0 <= nc < cols
                ):
                    continue

                if np.isnan(filled_dem[nr, nc]):
                    continue

                if np.isclose(
                    filled_dem[nr, nc],
                    current,
                    atol=1e-8
                ):
                    flat[row, col] = True
                    break

    # ---------------------------------------------------------
    # Group connected flat cells
    # ---------------------------------------------------------

    flat_labels, num_flats = label(
        flat,
        structure=np.ones((3, 3), dtype=int)
    )

    print(
        "Number of flat regions:",
        num_flats
    )

    epsilon = 1e-6

    # ---------------------------------------------------------
    # Process every flat independently
    # ---------------------------------------------------------

    for flat_id in range(1, num_flats + 1):

        flat_region = (
            flat_labels == flat_id
        )

        flat_rows, flat_cols = np.where(
            flat_region
        )

        if len(flat_rows) == 0:
            continue

        # -----------------------------------------------------
        # Find cells on this flat that touch LOWER terrain.
        # These are potential outlets.
        # -----------------------------------------------------

        outlet_cells = []

        for row, col in zip(
            flat_rows,
            flat_cols
        ):

            current = filled_dem[row, col]

            found_lower = False

            for dr, dc in neighbors:

                nr = row + dr
                nc = col + dc

                if not (
                    0 <= nr < rows
                    and 0 <= nc < cols
                ):
                    continue

                if np.isnan(filled_dem[nr, nc]):
                    continue

                if (
                    filled_dem[nr, nc]
                    < current - 1e-8
                ):
                    found_lower = True
                    break

            if found_lower:
                outlet_cells.append(
                    (row, col)
                )

        # No lower outlet means this flat cannot
        # be resolved by this method.
        if not outlet_cells:
            continue

        # -----------------------------------------------------
        # BFS distance from outlets through the flat
        # -----------------------------------------------------

        distance = np.full(
            (rows, cols),
            np.inf
        )

        queue = deque()

        for row, col in outlet_cells:

            distance[row, col] = 0

            queue.append(
                (row, col)
            )

        while queue:

            row, col = queue.popleft()

            current_distance = distance[
                row, col
            ]

            for dr, dc in neighbors:

                nr = row + dr
                nc = col + dc

                if not (
                    0 <= nr < rows
                    and 0 <= nc < cols
                ):
                    continue

                if not flat_region[nr, nc]:
                    continue

                new_distance = (
                    current_distance + 1
                )

                if (
                    new_distance
                    < distance[nr, nc]
                ):

                    distance[nr, nc] = (
                        new_distance
                    )

                    queue.append(
                        (nr, nc)
                    )

        # -----------------------------------------------------
        # Add gradient.
        #
        # Farther from outlet = slightly higher.
        # Therefore D8 flows toward the outlet.
        # -----------------------------------------------------

        for row, col in zip(
            flat_rows,
            flat_cols
        ):

            d = distance[row, col]

            if np.isfinite(d):

                resolved[row, col] += (
                    epsilon * d
                )

    return resolved

def find_spill_elevation(depression_mask, filled_dem):
    """
    Estimate the spill elevation of a depression
    from the priority-flood filled DEM.
    """

    mask = depression_mask

    valid_filled = filled_dem[mask]
    valid_filled = valid_filled[~np.isnan(valid_filled)]

    if len(valid_filled) == 0:
        return None

    return float(np.max(valid_filled))

def find_spill_point(
    depression_mask,
    dem,
    filled_dem,
    grid_x,
    grid_y
):
    """
    Find a candidate spill point for a depression.

    The spill elevation comes from the priority-flood
    surface. The spill point is searched along the
    boundary of the depression.
    """

    mask = depression_mask

    rows, cols = np.where(mask)

    boundary = []

    for row, col in zip(rows, cols):

        for dr, dc in [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]:

            nr = row + dr
            nc = col + dc

            if not (
                0 <= nr < mask.shape[0]
                and 0 <= nc < mask.shape[1]
            ):
                continue

            if not mask[nr, nc]:
                boundary.append((row, col))
                break

    if not boundary:
        return None

    boundary = np.array(
        boundary,
        dtype=int
    )

    boundary_rows = boundary[:, 0]
    boundary_cols = boundary[:, 1]

    boundary_elevations = dem[
        boundary_rows,
        boundary_cols
    ]

    valid = ~np.isnan(boundary_elevations)

    boundary_rows = boundary_rows[valid]
    boundary_cols = boundary_cols[valid]
    boundary_elevations = boundary_elevations[valid]

    if len(boundary_elevations) == 0:
        return None

    spill_elevation = find_spill_elevation(
        depression_mask,
        filled_dem
    )

    # Find the boundary point closest to the
    # spill elevation.
    index = np.argmin(
        np.abs(
            boundary_elevations -
            spill_elevation
        )
    )

    row = boundary_rows[index]
    col = boundary_cols[index]

    return {
        "easting": float(grid_x[row, col]),
        "northing": float(grid_y[row, col]),
        "elevation_m": float(dem[row, col]),
        "spill_elevation_m": float(spill_elevation)
    }

def touches_boundary(mask):
    return (
        np.any(mask[0, :]) or
        np.any(mask[-1, :]) or
        np.any(mask[:, 0]) or
        np.any(mask[:, -1])
    )

def analyze_ponds(kml_file, resolution=5):

    contours = parse_kml(
        Path(kml_file)
    )

    projected = project_contours(contours)

    extent = get_extent(projected)

    points, elevations = contour_points(
        projected,
        spacing=5
    )


    grid_x, grid_y, dem = create_dem(
        points,
        elevations,
        resolution
    )
    dem_height, dem_width = dem.shape

    dem_area_ha = (
        dem_height *
        dem_width *
        resolution**2
        / 10000
    )

    x_min = grid_x.min()
    y_max = grid_y.max()

    transform = from_origin(
        x_min,
        y_max,
        resolution,
        resolution
    )

    with rasterio.open(
        "dem.tif",
        "w",
        driver="GTiff",
        height=dem.shape[0],
        width=dem.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32644",  # CHANGE if your actual CRS differs
        transform=transform,
        nodata=-9999
    ) as dst:

        dem_to_write = np.where(
            np.isnan(dem),
            -9999,
            dem
        ).astype("float32")

        dst.write(
            dem_to_write,
            1
        )

    grid = Grid.from_raster(
        "dem.tif"
    )

    dem_raster = grid.read_raster(
        "dem.tif"
    )

    filled_dem = grid.fill_depressions(
        dem_raster
    )
    inflated_dem = grid.resolve_flats(
        filled_dem
    )

    fdir = grid.flowdir(
        inflated_dem
    )

    acc = grid.accumulation(
        fdir
    )


    fill_depth = filled_dem - dem

    depression_mask = fill_depth >= 1.0

    labeled, num_depressions = label(
        depression_mask
    )

    depression_info = []

    for region_id in range(1, num_depressions + 1):

        region = labeled == region_id

        if not np.any(region):
            continue


        if touches_boundary(region):
            continue

        cell_count = np.count_nonzero(region)

        area_m2 = cell_count * resolution ** 2

        depths = fill_depth[region]
        depths = depths[np.isfinite(depths)]
        depths = depths[depths > 0]

        if len(depths) == 0:
            continue

        max_depth = np.max(depths)
        depression_info.append({
            "id": region_id,
            "mask": region,
            "cells": cell_count,
            "area_m2": area_m2,
            "area_ha": area_m2 / 10_000,
            "max_depth": max_depth
        })

    depression_info.sort(
        key=lambda x: x["max_depth"],
        reverse=True
    )

    
    for depression in depression_info:

        spill = find_spill_point(
            depression["mask"],
            dem,
            filled_dem,
            grid_x,
            grid_y
        )

        depression["spill_point"] = spill

    
    for depression in depression_info:

        spill = depression["spill_point"]

        if spill is None:
            depression["catchment_area_m2"] = 0
            depression["catchment_area_ha"] = 0
            depression["volume_m3"] = 0
            depression["catchment_pond_ratio"] = 0
            continue

        spill_x = spill["easting"]
        spill_y = spill["northing"]

        distance = (
            (grid_x - spill_x)**2 +
            (grid_y - spill_y)**2
        )

        spill_row, spill_col = np.unravel_index(
            np.nanargmin(distance),
            distance.shape
        )

        accumulation_cells = acc[spill_row, spill_col]

        accumulation_area_m2 = accumulation_cells * resolution ** 2

        accumulation_area_ha = (
            accumulation_area_m2 / 10000
        )

        depression["catchment_area_m2"] = accumulation_area_m2
        depression["catchment_area_ha"] = accumulation_area_ha

        depression_mask = depression["mask"]

        depths = fill_depth[depression_mask]

        depths = depths[np.isfinite(depths)]
        depths = depths[depths > 0]

        volume_m3 = np.sum(depths) * (resolution ** 2)

        depression["volume_m3"] = volume_m3

    #MIN_POND_AREA_HA = 0.10
    #MIN_CATCHMENT_HA = 0.50

    pond_candidates = []

    for d in depression_info:

        #if d["area_ha"] < MIN_POND_AREA_HA:
        #    continue

        #if d["catchment_area_ha"] < MIN_CATCHMENT_HA:
        #    continue

        if touches_boundary(d["mask"]):
            continue

        pond_candidates.append(d)

    results = []

    for d in pond_candidates:

        results.append({
            "pond_id": int(d["id"]),
            "pond_area_ha": float(d["area_ha"]),
            "max_depth_m": float(d["max_depth"]),
            "volume_m3": float(d["volume_m3"]),
        })

    return results

def find_catchment(
    kml_file,
    pond_id,
    resolution=5
):

    contours = parse_kml(
        Path(kml_file)
    )

    projected = project_contours(contours)

    extent = get_extent(projected)

    points, elevations = contour_points(
        projected,
        spacing=5
    )

    grid_x, grid_y, dem = create_dem(
        points,
        elevations,
        resolution
    )

    x_min = grid_x.min()
    y_max = grid_y.max()

    transform = from_origin(
        x_min,
        y_max,
        resolution,
        resolution
    )

    with rasterio.open(
        "dem.tif",
        "w",
        driver="GTiff",
        height=dem.shape[0],
        width=dem.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32644",
        transform=transform,
        nodata=-9999
    ) as dst:

        dem_to_write = np.where(
            np.isnan(dem),
            -9999,
            dem
        ).astype("float32")

        dst.write(
            dem_to_write,
            1
        )

    grid = Grid.from_raster(
        "dem.tif"
    )

    dem_raster = grid.read_raster(
        "dem.tif"
    )

    filled_dem = grid.fill_depressions(
        dem_raster
    )

    inflated_dem = grid.resolve_flats(
        filled_dem
    )

    fdir = grid.flowdir(
        inflated_dem
    )

    acc = grid.accumulation(
        fdir
    )

    fill_depth = filled_dem - dem

    depression_mask = fill_depth >= 1.0

    labeled, num_depressions = label(
        depression_mask
    )

    if pond_id < 1 or pond_id > num_depressions:
        raise ValueError(
            f"Pond {pond_id} not found"
        )

    region = labeled == pond_id

    cell_count = np.count_nonzero(region)

    area_m2 = cell_count * resolution ** 2
    area_ha = area_m2 / 10000

    spill = find_spill_point(
            region,
            dem,
            filled_dem,
            grid_x,
            grid_y
        )

    if spill is None:
        raise ValueError(
            f"Could not find spill point for pond {pond_id}"
        )

    # Find raster cell nearest to
    # supplied spill coordinate

    spill_easting = spill["easting"]
    spill_northing = spill["northing"]

    distance = (
        (grid_x - spill_easting) ** 2
        +
        (grid_y - spill_northing) ** 2
    )

    spill_row, spill_col = (
        np.unravel_index(
            np.nanargmin(distance),
            distance.shape
        )
    )

    accumulation_cells = acc[
        spill_row,
        spill_col
    ]

    catchment_area_m2 = (
        accumulation_cells *
        resolution ** 2
    )

    catchment_area_ha = (
        catchment_area_m2 /
        10000
    )

    spill_elevation = dem[
        spill_row,
        spill_col
    ]

    if area_ha > 0:
        catchment_pond_ratio = (
            catchment_area_ha
            / area_ha
        )
    else:
        catchment_pond_ratio = 0            

    return {
        "pond_id": pond_id,
        "spill": {
            "easting": float(
                grid_x[spill_row, spill_col]
            ),
            "northing": float(
                grid_y[spill_row, spill_col]
            ),
            "elevation_m": float(
                spill_elevation
            )
        },

        "flow_accumulation_cells": int(
            accumulation_cells
        ),

        "catchment_area_m2": float(
            catchment_area_m2
        ),

        "catchment_area_ha": float(
            catchment_area_ha
        ),
        "catchment_pond_ratio": float(
            catchment_pond_ratio
        ),
    }


        
