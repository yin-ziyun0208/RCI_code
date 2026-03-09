"""
River Network Centroid Analysis from Gridded Data

This module calculates the mass centroid of a river basin using gridded data
(e.g., precipitation, runoff, temperature) by:
1. Creating incremental upstream areas ("slices") along the main stem
2. Extracting mass values for each slice from gridded data
3. Calculating the mass-weighted centroid along the main stem

Input:
    - River network shapefile with topology
    - Catchment shapefile with COMID geometries
    - Gridded data file (GeoTIFF or NetCDF format)

Output:
    - CSV file with basin name, centroid distance (km), centroid COMID, and RCI

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import subprocess
import tempfile
from typing import List, Tuple, Dict, Optional
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
import xarray as xr
import shutil


# ============================================================================
# 1. MAIN STEM TRACING FUNCTIONS
# ============================================================================

def trace_main_stem_from_outlet(
    gdf: gpd.GeoDataFrame, 
    outlet_comid: int,
    uparea_col: str = 'uparea',
    up_cols: List[str] = None
) -> List[int]:
    """
    Trace the main stem upstream from a given outlet COMID.
    
    This function follows the same logic as trace_main_stem_from_max_uparea:
    starting from the outlet, it traces upstream by always selecting the tributary
    with the largest upstream area.
    
    Args:
        gdf: GeoDataFrame containing river network data
        outlet_comid: COMID of the outlet segment
        uparea_col: Name of the column containing upstream area values
        up_cols: List of column names for upstream COMIDs
    
    Returns:
        List of COMIDs along the main stem from upstream to downstream
    """
    if up_cols is None:
        up_cols = ['up1', 'up2', 'up3', 'up4']
    
    # Start from the outlet
    main_stem_comids_down_to_up = [outlet_comid]
    current_comid = outlet_comid
    
    # Trace upstream
    while True:
        # Get current segment data
        try:
            current_row = gdf[gdf['COMID'] == current_comid].iloc[0]
        except IndexError:
            # If current COMID not found, terminate
            print(f"Warning: COMID {current_comid} not found in data, tracing terminated.")
            break

        # Get all upstream COMIDs
        up_comids = [current_row[up_col] for up_col in up_cols if current_row[up_col] != 0]
        
        # If no upstream segments, reached the source
        if not up_comids:
            break

        # Find the upstream segment with maximum upstream area
        max_uparea = -1
        next_comid = None
        for up_id in up_comids:
            # Ensure upstream COMID exists in dataset
            if not gdf['COMID'].isin([up_id]).any():
                continue
            
            up_row = gdf[gdf['COMID'] == up_id].iloc[0]
            if up_row[uparea_col] > max_uparea:
                max_uparea = up_row[uparea_col]
                next_comid = up_id
        
        # Add next main stem segment to list and update current_comid
        if next_comid is not None:
            main_stem_comids_down_to_up.append(next_comid)
            current_comid = next_comid
        else:
            # If no valid upstream COMIDs found, terminate
            break
    
    # Reverse to get upstream to downstream order
    main_stem_comids_up_to_down = main_stem_comids_down_to_up[::-1]
    return main_stem_comids_up_to_down

def find_basin_outlet(
    gdf: gpd.GeoDataFrame,
    uparea_col: str = 'uparea'
) -> int:
    """
    Find the outlet segment of the entire basin.
    
    The outlet is defined as the segment with the maximum upstream area.
    
    Args:
        gdf: GeoDataFrame with river network data
        uparea_col: Column name for upstream area
    
    Returns:
        COMID of the basin outlet
    """
    outlet_idx = gdf[uparea_col].idxmax()
    return gdf.loc[outlet_idx, 'COMID']


# ============================================================================
# 2. UPSTREAM MAPPING FUNCTIONS
# ============================================================================

def create_upstream_mapping(gdf: gpd.GeoDataFrame) -> Dict[int, List[int]]:
    """
    Create a mapping from each river segment to its upstream segments.
    
    Args:
        gdf: GeoDataFrame containing river network data
    
    Returns:
        Dictionary mapping COMID to list of upstream COMIDs
    """
    upstream_map = {}
    
    for _, row in gdf.iterrows():
        comid = row['COMID']
        upstreams = []
        
        # Collect all upstream segments
        for up_col in ['up1', 'up2', 'up3', 'up4']:
            up_id = row[up_col]
            if up_id != 0 and up_id != -1 and not pd.isna(up_id):
                upstreams.append(int(up_id))
        
        upstream_map[comid] = upstreams

    return upstream_map


def find_all_upstreams(
    comid: int, 
    upstream_map: Dict[int, List[int]], 
    cache: Dict[int, List[int]],
    visited: set = None
) -> List[int]:
    """
    Recursively find all upstream segments.
    
    Args:
        comid: Starting COMID
        upstream_map: Global upstream mapping
        cache: Cache dictionary for storing results
        visited: Set of visited COMIDs to prevent cycles
    
    Returns:
        List of all upstream COMIDs (including indirect upstreams)
    """
    if comid in cache:
        return cache[comid]
    
    if visited is None:
        visited = set()
    
    if comid in visited:
        return []
    
    visited.add(comid)
    upstreams = upstream_map.get(comid, [])
    
    all_upstreams = upstreams.copy()
    for up_comid in upstreams:
        if up_comid not in visited:
            all_upstreams.extend(
                find_all_upstreams(up_comid, upstream_map, cache, visited)
            )
    
    # Remove duplicates and sort
    result = sorted(list(set(all_upstreams)))
    cache[comid] = result
    return result


def get_slice_comids(
    current_list: List[int], 
    previous_list: Optional[List[int]]
) -> List[int]:
    """
    Calculate the difference between current and previous COMID lists.
    
    This creates incremental slices along the main stem.
    
    Args:
        current_list: Current segment's upstream COMID list
        previous_list: Previous segment's upstream COMID list
    
    Returns:
        List of COMIDs present in current but not in previous
    """
    if previous_list is None:
        return current_list
    else:
        current_set = set(current_list)
        previous_set = set(previous_list)
        return list(current_set - previous_set)


# ============================================================================
# 3. GEOMETRY PROCESSING FUNCTIONS
# ============================================================================

def dissolve_comids(
    comids: List[int], 
    comid_to_geometry: Dict[int, object]
) -> object:
    """
    Dissolve multiple COMID geometries into a single geometry.
    
    Args:
        comids: List of COMIDs to dissolve
        comid_to_geometry: Mapping from COMID to geometry
    
    Returns:
        Dissolved geometry
    """
    polygons = []
    for comid in comids:
        if comid in comid_to_geometry:
            polygons.append(comid_to_geometry[comid])
    
    if not polygons:
        return None
    
    gdf_temp = gpd.GeoDataFrame(geometry=polygons)
    dissolved = gdf_temp.dissolve()
    return dissolved.geometry.iloc[0] if len(dissolved) > 0 else None


def create_basin_slices(
    gdf_river: gpd.GeoDataFrame,
    gdf_catchment: gpd.GeoDataFrame,
    outlet_comid: int
) -> gpd.GeoDataFrame:
    """
    Create incremental basin slices along the main stem.
    
    Args:
        gdf_river: GeoDataFrame with river network data
        gdf_catchment: GeoDataFrame with catchment polygons (COMID geometry)
        outlet_comid: COMID of the basin outlet
    
    Returns:
        GeoDataFrame with basin slices
    """
    print("Creating upstream mapping...")
    upstream_map = create_upstream_mapping(gdf_river)
    
    # Trace main stem
    print("Tracing main stem...")
    mainstem_comids = trace_main_stem_from_outlet(gdf_river, outlet_comid)
    
    print(f"Main stem has {len(mainstem_comids)} segments")
    
    # Create COMID to geometry mapping
    comid_to_geometry = dict(zip(gdf_catchment['COMID'], gdf_catchment['geometry']))
    
    # Find all upstreams for each main stem segment
    print("Finding upstream areas for each segment...")
    cache = {}
    upstream_lists = []
    
    for comid in mainstem_comids:
        upstreams = find_all_upstreams(comid, upstream_map, cache)
        upstream_lists.append(sorted(upstreams + [comid]))
    
    # Create slices (differences between consecutive upstream lists)
    print("Creating incremental slices...")
    slices = []
    previous_list = None
    
    for i, (comid, current_list) in enumerate(zip(mainstem_comids, upstream_lists)):
        slice_comids = get_slice_comids(current_list, previous_list)
        
        if slice_comids:
            slice_geom = dissolve_comids(slice_comids, comid_to_geometry)
            
            if slice_geom is not None and not slice_geom.is_empty:
                slices.append({
                    'segment_CO': comid,
                    'outlet_COMID': outlet_comid,
                    'slice_geometry': slice_geom,
                    'area_km2': slice_geom.area / 1e6,
                    'slice_comid_count': len(slice_comids),
                    'segment_order': i,
                    'cumulative_comids': len(current_list)
                })
        
        previous_list = current_list
    
    # Create GeoDataFrame
    slices_gdf = gpd.GeoDataFrame(slices, geometry='slice_geometry', crs=gdf_catchment.crs)
    
    print(f"Created {len(slices_gdf)} slices")
    
    return slices_gdf


# ============================================================================
# 4. RASTER PROCESSING FUNCTIONS
# ============================================================================

def rasterize_slice(
    slice_gdf: gpd.GeoDataFrame,
    output_tif: str,
    attribute: str = 'segment_CO',
    resolution: Tuple[int, int] = (43200, 17400),
    extent: Tuple[float, float, float, float] = (-180, -60, 180, 85),
    keep_temp_files: bool = False
) -> str:
    """
    Rasterize a basin slice shapefile using gdal_rasterize.
    
    Args:
        slice_gdf: GeoDataFrame with a single slice
        output_tif: Path for output GeoTIFF
        attribute: Attribute field to burn into raster
        resolution: Output raster resolution (columns, rows)
        extent: Output extent (xmin, ymin, xmax, ymax)
        keep_temp_files: Whether to keep intermediate shapefiles
    
    Returns:
        Path to output GeoTIFF
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_tif)) or '.', exist_ok=True)
    
    # Save slice to shapefile (use same name as output_tif but with .shp extension)
    shp_path = output_tif.replace('.tif', '.shp')
    slice_gdf.to_file(shp_path)
    print(f"  Saved slice shapefile: {shp_path}")
    
    # Get layer name (without extension)
    layer_name = os.path.splitext(os.path.basename(shp_path))[0]
    
    # Build gdal_rasterize command
    cmd = [
        'gdal_rasterize',
        '-a', attribute,
        '-ts', str(resolution[0]), str(resolution[1]),
        '-te', str(extent[0]), str(extent[1]), str(extent[2]), str(extent[3]),
        '-a_nodata', '0',
        '-l', layer_name,
        shp_path,
        output_tif
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    
    # Run command
    result = subprocess.run(' '.join(cmd), shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  GDAL error: {result.stderr}")
        raise RuntimeError(f"gdal_rasterize failed with error code {result.returncode}")
    
    print(f"  Created raster: {output_tif}")
    
    # Clean up temporary shapefile if not keeping
    if not keep_temp_files:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            temp_file = output_tif.replace('.tif', ext)
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print(f"  Cleaned up temporary shapefiles")
    else:
        print(f"  Kept temporary shapefiles")
    
    return output_tif


def extract_slice_mass_from_raster(
    slice_tif: str,
    data_tif: str,
    method: str = 'sum'
) -> float:
    """
    Extract mass value from gridded data for a rasterized slice.
    
    Args:
        slice_tif: Path to slice raster (with segment COMID values)
        data_tif: Path to data raster (precipitation, runoff, etc.)
        method: Extraction method ('sum' or 'mean')
    
    Returns:
        Mass value for the slice
    """
    with rasterio.open(data_tif) as src_data:
        data_array = src_data.read(1)
        data_transform = src_data.transform
        data_nodata = src_data.nodata
        
        with rasterio.open(slice_tif) as src_slice:
            slice_array = src_slice.read(1)
            
            # Create mask for non-zero slice pixels
            mask = slice_array > 0
            
            if not np.any(mask):
                return 0.0
            
            # Extract data values where slice mask is True
            values = data_array[mask]
            
            # Remove nodata values
            if data_nodata is not None:
                values = values[values != data_nodata]
            
            if len(values) == 0:
                return 0.0
            
            if method == 'sum':
                return float(np.sum(values))
            elif method == 'mean':
                return float(np.mean(values))
            else:
                raise ValueError(f"Unknown method: {method}")


def extract_slice_mass_from_netcdf(
    slice_tif: str,
    data_nc: str,
    variable: str = 'precipitation',
    method: str = 'sum'
) -> float:
    """
    Extract mass value from NetCDF data for a rasterized slice.
    
    Args:
        slice_tif: Path to slice raster
        data_nc: Path to NetCDF file
        variable: Variable name in NetCDF
        method: Extraction method ('sum' or 'mean')
    
    Returns:
        Mass value for the slice
    """
    # Open NetCDF with xarray
    ds = xr.open_dataset(data_nc)
    
    if variable not in ds.variables:
        raise ValueError(f"Variable '{variable}' not found in NetCDF. Available: {list(ds.variables.keys())}")
    
    data_var = ds[variable]
    
    # Get raster info for masking
    with rasterio.open(slice_tif) as src:
        slice_array = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    # Create mask for non-zero slice pixels
    mask = slice_array > 0
    
    if not np.any(mask):
        return 0.0
    
    # Get coordinates of masked pixels
    rows, cols = np.where(mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    
    # Extract values from NetCDF (simplified - you may need to handle coordinate mapping)
    # This assumes NetCDF has lat/lon coordinates
    values = []
    for x, y in zip(xs, ys):
        try:
            val = data_var.sel(lon=x, lat=y, method='nearest').values
            if not np.isnan(val):
                values.append(val)
        except:
            continue
    
    if len(values) == 0:
        return 0.0
    
    if method == 'sum':
        return float(np.sum(values))
    elif method == 'mean':
        return float(np.mean(values))
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_slice_masses(
    slices_gdf: gpd.GeoDataFrame,
    gridded_data_path: str,
    temp_dir: str = './temp_raster',
    data_type: str = 'tif',
    variable: str = None,
    resolution: Tuple[int, int] = (43200, 17400),
    extent: Tuple[float, float, float, float] = (-180, -60, 180, 85),
    keep_temp_files: bool = False
) -> List[float]:
    """
    Calculate mass values for all slices from gridded data.
    
    Args:
        slices_gdf: GeoDataFrame with basin slices
        gridded_data_path: Path to gridded data file (GeoTIFF or NetCDF)
        temp_dir: Directory for temporary raster files
        data_type: Type of gridded data ('tif' or 'nc')
        variable: Variable name (required for NetCDF)
        resolution: Raster resolution for slice rasters
        extent: Spatial extent for slice rasters
        keep_temp_files: Whether to keep intermediate files
    
    Returns:
        List of mass values for each slice (in order)
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    masses = []
    
    for idx, row in slices_gdf.iterrows():
        print(f"  Processing slice {idx+1}/{len(slices_gdf)}...")
        
        # Create a GeoDataFrame for this single slice
        slice_gdf = gpd.GeoDataFrame(
            [row], 
            geometry='slice_geometry', 
            crs=slices_gdf.crs
        )
        
        # Rasterize this slice
        slice_tif = os.path.join(temp_dir, f"slice_{idx:04d}.tif")
        rasterize_slice(
            slice_gdf, 
            slice_tif,
            attribute='segment_CO',
            resolution=resolution,
            extent=extent,
            keep_temp_files=keep_temp_files
        )
        
        # Extract mass from gridded data
        if data_type.lower() == 'tif':
            mass = extract_slice_mass_from_raster(slice_tif, gridded_data_path)
        elif data_type.lower() == 'nc':
            if variable is None:
                raise ValueError("Variable name required for NetCDF files")
            mass = extract_slice_mass_from_netcdf(slice_tif, gridded_data_path, variable)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        masses.append(mass)
        
        # Clean up temporary raster if not keeping
        if not keep_temp_files and os.path.exists(slice_tif):
            os.remove(slice_tif)
            print(f"  Removed temporary raster: {slice_tif}")
    
    return masses


# ============================================================================
# 5. CENTROID CALCULATION FUNCTIONS
# ============================================================================

def calculate_centroid_from_slices(
    slices_gdf: gpd.GeoDataFrame,
    slice_masses: List[float]
) -> Tuple[float, float, int, float]:
    """
    Calculate centroid from basin slices and their masses.
    
    Args:
        slices_gdf: GeoDataFrame with basin slices (in order from source to outlet)
        slice_masses: Mass values for each slice (same order as slices_gdf)
    
    Returns:
        Tuple containing:
            - max_length: Total length of main stem (km)
            - centroid: Distance from outlet to centroid (km)
            - centroid_comid: COMID of segment containing centroid
            - rci: Relative Centroid Index
    """
    # Sort slices by segment_order (from source to outlet)
    slices_sorted = slices_gdf.sort_values('segment_order')
    
    # Get main stem COMIDs in order
    mainstem_comids = slices_sorted['segment_CO'].tolist()
    
    # Get river network to extract segment lengths
    # For now, assume we need to pass this separately
    # This will be handled in the main function
    
    return mainstem_comids, slices_sorted, slice_masses


# ============================================================================
# 6. MAIN WORKFLOW FUNCTION
# ============================================================================

def calculate_basin_centroid_from_gridded(
    river_network_path: str,
    catchment_path: str,
    gridded_data_path: str,
    output_path: str,
    basin_name: str,
    data_type: str = 'tif',
    variable: str = None,
    comid_col: str = 'COMID',
    length_col: str = 'lengthkm',
    uparea_col: str = 'uparea',
    up_cols: List[str] = None,
    temp_dir: str = './temp_raster',
    raster_resolution: Tuple[int, int] = (43200, 17400),
    raster_extent: Tuple[float, float, float, float] = (-180, -60, 180, 85),
    min_segments: int = 3,
    keep_temp_files: bool = False
) -> pd.DataFrame:
    """
    Calculate basin centroid using gridded data.
    
    This function:
    1. Loads river network and catchment data
    2. Identifies the basin outlet
    3. Creates incremental basin slices along the main stem
    4. Extracts mass values for each slice from gridded data
    5. Calculates the mass-weighted centroid
    
    Args:
        river_network_path: Path to river network shapefile
        catchment_path: Path to catchment shapefile (with COMID geometries)
        gridded_data_path: Path to gridded data (GeoTIFF or NetCDF)
        output_path: Path for output CSV file
        basin_name: Name of the basin
        data_type: Type of gridded data ('tif' or 'nc')
        variable: Variable name (required for NetCDF)
        comid_col: Name of COMID column
        length_col: Name of length column in river network
        uparea_col: Name of upstream area column
        up_cols: List of upstream column names
        temp_dir: Directory for temporary raster files
        raster_resolution: Resolution for slice rasters
        raster_extent: Extent for slice rasters
        min_segments: Minimum number of segments required
        keep_temp_files: Whether to keep intermediate files
    
    Returns:
        DataFrame with centroid results
    """
    if up_cols is None:
        up_cols = ['up1', 'up2', 'up3', 'up4']
    
    print("=" * 60)
    print(f"Calculating centroid from gridded data for basin: {basin_name}")
    print("=" * 60)
    
    # ------------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------------
    print("\n[Step 1] Loading data...")
    
    # Load river network
    print(f"  Loading river network from: {river_network_path}")
    gdf_river = gpd.read_file(river_network_path)
    gdf_river[comid_col] = gdf_river[comid_col].astype(int)
    gdf_river[length_col] = pd.to_numeric(gdf_river[length_col], errors='coerce').fillna(0)
    print(f"  Loaded {len(gdf_river)} river segments")
    
    # Load catchment
    print(f"  Loading catchment from: {catchment_path}")
    gdf_catchment = gpd.read_file(catchment_path)
    gdf_catchment[comid_col] = gdf_catchment[comid_col].astype(int)
    print(f"  Loaded {len(gdf_catchment)} catchment polygons")
    
    # Check CRS match
    if gdf_river.crs != gdf_catchment.crs:
        print(f"  Warning: CRS mismatch. Converting catchment to river CRS")
        gdf_catchment = gdf_catchment.to_crs(gdf_river.crs)
    
    # ------------------------------------------------------------------------
    # Step 2: Find basin outlet
    # ------------------------------------------------------------------------
    print("\n[Step 2] Identifying basin outlet...")
    
    outlet_comid = find_basin_outlet(gdf_river, uparea_col)
    outlet_area = gdf_river[gdf_river[comid_col] == outlet_comid][uparea_col].values[0]
    
    print(f"  Outlet COMID: {outlet_comid}")
    print(f"  Upstream area: {outlet_area:.2f} km²")
    
    # ------------------------------------------------------------------------
    # Step 3: Create basin slices
    # ------------------------------------------------------------------------
    print("\n[Step 3] Creating basin slices...")
    
    slices_gdf = create_basin_slices(gdf_river, gdf_catchment, outlet_comid)
    
    if len(slices_gdf) < min_segments:
        raise ValueError(f"Only {len(slices_gdf)} slices created (minimum {min_segments} required)")
    
    # ------------------------------------------------------------------------
    # Step 4: Extract segment lengths
    # ------------------------------------------------------------------------
    print("\n[Step 4] Extracting segment lengths...")
    
    # Get main stem COMIDs in order
    slices_sorted = slices_gdf.sort_values('segment_order')
    mainstem_comids = slices_sorted['segment_CO'].tolist()
    
    # Get lengths for each segment
    comid_to_length = dict(zip(gdf_river[comid_col], gdf_river[length_col]))
    segment_lengths = [comid_to_length.get(comid, 0) for comid in mainstem_comids]
    
    # Calculate cumulative lengths from source
    cum_lengths = []
    cum = 0
    for length in segment_lengths:
        cum += length
        cum_lengths.append(cum)
    
    print(f"  Main stem total length: {cum_lengths[-1]:.2f} km")
    
    # ------------------------------------------------------------------------
    # Step 5: Extract mass values from gridded data
    # ------------------------------------------------------------------------
    print("\n[Step 5] Extracting mass values from gridded data...")
    
    slice_masses = calculate_slice_masses(
        slices_gdf=slices_gdf,
        gridded_data_path=gridded_data_path,
        temp_dir=temp_dir,
        data_type=data_type,
        variable=variable,
        resolution=raster_resolution,
        extent=raster_extent,
        keep_temp_files=keep_temp_files
    )
    
    print(f"  Total mass: {sum(slice_masses):.2f}")
    
    # ------------------------------------------------------------------------
    # Step 6: Calculate centroid
    # ------------------------------------------------------------------------
    print("\n[Step 6] Calculating centroid...")
    
    # Calculate centroid using the same formula as before
    cum_array = np.array(cum_lengths)
    mass_array = np.array(slice_masses)
    total_mass = mass_array.sum()
    
    if total_mass == 0:
        raise ValueError("Total mass is zero - cannot calculate centroid")
    
    max_length = cum_array[-1]
    weighted_sum = np.sum(cum_array * mass_array)
    centroid_from_source = weighted_sum / total_mass
    centroid_from_outlet = max_length - centroid_from_source
    
    # Find nearest COMID
    distances_from_outlet = [max_length - cum for cum in cum_lengths]
    nearest_idx = np.argmin(np.abs(np.array(distances_from_outlet) - centroid_from_outlet))
    centroid_comid = mainstem_comids[nearest_idx]
    
    # Calculate RCI
    rci = centroid_from_outlet / max_length
    
    print(f"  Centroid distance from outlet: {centroid_from_outlet:.2f} km")
    print(f"  Centroid COMID: {centroid_comid}")
    print(f"  RCI: {rci:.4f}")
    
    # ------------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------------
    print("\n[Step 7] Saving results...")
    
    results = pd.DataFrame([{
        'basin_name': basin_name,
        'outlet_COMID': outlet_comid,
        'centroid_COMID': centroid_comid,
        'centroid_distance_km': centroid_from_outlet,
        'mainstem_length_km': max_length,
        'rci': rci,
        'num_slices': len(slices_gdf),
        'total_mass': total_mass,
        'outlet_uparea_km2': outlet_area
    }])
    
    # Also save slice details for reference
    slices_output = output_path.replace('.csv', '_slices.csv')
    slices_df = pd.DataFrame({
        'segment_CO': mainstem_comids,
        'cumulative_length_km': cum_lengths,
        'slice_mass': slice_masses,
        'segment_order': range(len(mainstem_comids))
    })
    slices_df.to_csv(slices_output, index=False)
    print(f"  Slice details saved to: {slices_output}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    
    # Save main results
    results.to_csv(output_path, index=False)
    
    print(f"  Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Calculation complete!")
    print("=" * 60)
    
    # Clean up temp directory if not keeping files
    if not keep_temp_files and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"  Cleaned up temporary directory: {temp_dir}")
    
    return results


# ============================================================================
# 7. MAIN FUNCTION - EDIT WITH YOUR DATA PATHS
# ============================================================================

def main():
    """
    Main function - EDIT THIS SECTION WITH YOUR DATA PATHS
    """
    # =========================================================================
    # USER INPUT SECTION - PLEASE MODIFY THESE PATHS
    # =========================================================================
    
    # Path to your river network shapefile
    # Must contain columns: COMID, lengthkm, uparea, up1, up2, up3, up4
    RIVER_NETWORK_PATH = "data/Ganjiang_reach.shp"
    
    # Path to your catchment shapefile (with COMID geometries)
    # Must contain COMID and geometry columns
    CATCHMENT_PATH = "data/Ganjiang_cat.shp"
    
    # Path to your gridded data file
    # Can be GeoTIFF (.tif) or NetCDF (.nc)
    GRIDDED_DATA_PATH = "data/pcp_as_basins_grid.nc"
    
    # Type of gridded data: 'tif' or 'nc'
    DATA_TYPE = "nc"
    
    # If using NetCDF, specify the variable name
    VARIABLE = "precipitation"  # e.g., "precipitation" or "runoff"
    
    # Output path for results
    OUTPUT_PATH = "centroid_results.csv"
    
    # Name of your basin
    BASIN_NAME = "poyang"
    
    # Column names in your shapefiles (modify if different)
    COMID_COL = "COMID"
    LENGTH_COL = "lengthkm"
    UPAREA_COL = "uparea"
    UP_COLS = ["up1", "up2", "up3", "up4"]
    
    # Raster processing parameters
    TEMP_DIR = "./temp_raster"
    RASTER_RESOLUTION = (43200, 17400)  # columns, rows
    RASTER_EXTENT = (-180, -60, 180, 85)  # xmin, ymin, xmax, ymax
    
    # Minimum number of slices required
    MIN_SEGMENTS = 3
    
    # Whether to keep intermediate files (shapefiles, rasters)
    KEEP_TEMP_FILES = True  # Set to True to examine intermediate files
    
    # =========================================================================
    # END OF USER INPUT SECTION
    # =========================================================================
    
    # Validate input files exist
    if not os.path.exists(RIVER_NETWORK_PATH):
        raise FileNotFoundError(f"River network file not found: {RIVER_NETWORK_PATH}")
    
    if not os.path.exists(CATCHMENT_PATH):
        raise FileNotFoundError(f"Catchment file not found: {CATCHMENT_PATH}")
    
    if not os.path.exists(GRIDDED_DATA_PATH):
        raise FileNotFoundError(f"Gridded data file not found: {GRIDDED_DATA_PATH}")
    
    # Run the centroid calculation
    results = calculate_basin_centroid_from_gridded(
        river_network_path=RIVER_NETWORK_PATH,
        catchment_path=CATCHMENT_PATH,
        gridded_data_path=GRIDDED_DATA_PATH,
        output_path=OUTPUT_PATH,
        basin_name=BASIN_NAME,
        data_type=DATA_TYPE,
        variable=VARIABLE,
        comid_col=COMID_COL,
        length_col=LENGTH_COL,
        uparea_col=UPAREA_COL,
        up_cols=UP_COLS,
        temp_dir=TEMP_DIR,
        raster_resolution=RASTER_RESOLUTION,
        raster_extent=RASTER_EXTENT,
        min_segments=MIN_SEGMENTS,
        keep_temp_files=KEEP_TEMP_FILES
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Basin: {BASIN_NAME}")
    print(f"Outlet COMID: {results['outlet_COMID'].values[0]}")
    print(f"Centroid COMID: {results['centroid_COMID'].values[0]}")
    print(f"Centroid distance from outlet: {results['centroid_distance_km'].values[0]:.2f} km")
    print(f"Main stem length: {results['mainstem_length_km'].values[0]:.2f} km")
    print(f"RCI: {results['rci'].values[0]:.4f}")
    print(f"Number of slices: {results['num_slices'].values[0]}")
    print(f"Total mass: {results['total_mass'].values[0]:.2f}")
    print(f"Outlet upstream area: {results['outlet_uparea_km2'].values[0]:.2f} km²")
    print("=" * 60)


if __name__ == "__main__":
    main()