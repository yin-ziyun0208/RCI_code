# this code claculate
# input: river network, discharge data
# output: csv (basinname, level, subbasin_code, centroid_loc, centroid_COMID, RCI)
# name: time:

# preparation: mainstem, discharge list
# calculate centroid
# return: distance, COMID, RCI 

"""
River Network Centroid Analysis from Discharge Data

This module calculates flow centroids for river basins at multiple Pfafstetter levels
using discharge data. For each subbasin, it:
1. Identifies the main stem
2. Extracts discharge values along the main stem
3. Calculates the flow-weighted centroid
4. Outputs results with basin name, level, subbasin code, centroid location, COMID, and RCI

Input:
    - River network shapefile with topology
    - Discharge data CSV (COMID, qout)
    - Pfafstetter codes CSV (COMID, pfafstetter)

Output:
    - CSV file with columns: basin_name, level, subbasin_code, centroid_distance_km, 
      centroid_COMID, rci, outlet_COMID, mainstem_length_km, num_segments, total_discharge

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import os
from typing import List, Tuple, Dict, Optional


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


def find_subbasin_outlets(
    gdf: gpd.GeoDataFrame,
    pfaf_df: pd.DataFrame,
    level: int,
    uparea_col: str = 'uparea',
    pfaf_col: str = 'pfafstetter'
) -> pd.DataFrame:
    """
    Find outlet segments for all subbasins at a given Pfafstetter level.
    
    Args:
        gdf: GeoDataFrame with river network data
        pfaf_df: DataFrame with Pfafstetter codes
        level: Pfafstetter level to analyze
        uparea_col: Column name for upstream area
        pfaf_col: Column name for Pfafstetter code
    
    Returns:
        DataFrame with outlet information for each subbasin
    """
    # Merge Pfafstetter codes
    merged_df = gdf.merge(pfaf_df, on='COMID', how='inner')
    
    # Filter for segments with Pfafstetter code length >= level
    merged_df['pfaf_len'] = merged_df[pfaf_col].astype(str).str.len()
    merged_df = merged_df[merged_df['pfaf_len'] >= level].copy()
    
    # Create subbasin ID from first 'level' digits of Pfafstetter code
    merged_df['subbasin_id'] = merged_df[pfaf_col].astype(str).str[:level]
    
    if merged_df.empty:
        return pd.DataFrame()
    
    # Find outlet for each subbasin (segment with maximum upstream area)
    outlets = merged_df.loc[merged_df.groupby('subbasin_id')[uparea_col].idxmax()]
    
    return outlets[['subbasin_id', 'COMID', uparea_col]].rename(columns={'COMID': 'outlet_COMID'})


# ============================================================================
# 2. DISCHARGE PROCESSING FUNCTIONS
# ============================================================================

def calculate_incremental_discharge(
    gdf: gpd.GeoDataFrame,
    mainstem_comids: List[int],
    q_df: pd.DataFrame,
    comid_col: str = 'COMID',
    q_col: str = 'qout',
    length_col: str = 'lengthkm'
) -> Tuple[List[float], List[float]]:
    """
    Calculate cumulative lengths and incremental discharge along main stem.
    
    The incremental discharge is simply the difference between consecutive
    segments: delta_q[i] = q[i] - q[i-1] (with delta_q[0] = q[0])
    
    Args:
        gdf: GeoDataFrame with river network data
        mainstem_comids: List of COMIDs along main stem (upstream to downstream)
        q_df: DataFrame with discharge data
        comid_col: Name of COMID column
        q_col: Name of discharge column
        length_col: Name of length column
    
    Returns:
        Tuple containing:
            - List of cumulative lengths from source (km)
            - List of incremental discharge values (delta_q)
    """
    # Create lookup dictionaries
    length_dict = gdf.set_index(comid_col)[length_col].to_dict()
    q_dict = q_df.set_index(comid_col)[q_col].to_dict()
    
    # Calculate cumulative lengths and collect Q values
    cum_lengths = []
    q_values = []
    cum_length = 0
    
    for comid in mainstem_comids:
        if comid not in length_dict or comid not in q_dict:
            continue
        
        cum_length += length_dict[comid]
        cum_lengths.append(cum_length)
        q_values.append(q_dict[comid])
    
    # Calculate delta Q: simple adjacent difference
    delta_q_list = []
    for i, q in enumerate(q_values):
        if i == 0:
            delta_q = q  # First segment: delta = its own Q
        else:
            delta_q = q - q_values[i-1]  # Subsequent segments: delta = current - previous
        delta_q_list.append(delta_q)
    
    return cum_lengths, delta_q_list


# ============================================================================
# 3. CENTROID CALCULATION FUNCTIONS (from 0_calculate_centroid.py)
# ============================================================================

def calculate_centroid(
    cumulative_values: List[float], 
    delta_values: List[float]
) -> Tuple[float, float]:
    """
    Calculate the flow centroid and total length of a river main stem.
    
    The centroid represents the point along the main stem where half of the total
    accumulated flow is upstream and half downstream.
    
    Args:
        cumulative_values: List of cumulative distances from source (km)
        delta_values: List of incremental flow values at each segment
    
    Returns:
        Tuple containing (max_length, centroid_position):
            - max_length: Total length of the main stem (km)
            - centroid: Distance from outlet to centroid (km)
    """
    if len(cumulative_values) == 0 or len(delta_values) == 0:
        return 0, 0
    
    cumulative_array = np.array(cumulative_values)
    delta_array = np.array(delta_values)
    total_delta = delta_array.sum()
    
    if total_delta == 0:
        # If no flow, define centroid as the midpoint
        max_length = cumulative_array[-1]
        return max_length, max_length / 2
    
    max_length = cumulative_array[-1]
    
    # Calculate centroid (distance from source)
    weighted_sum = np.sum(cumulative_array * delta_array)
    centroid_from_source = weighted_sum / total_delta
    
    # Convert to distance from outlet
    centroid_from_outlet = max_length - centroid_from_source
    
    return max_length, centroid_from_outlet


def find_nearest_comid(
    centroid: float,
    max_length: float,
    comid_list: List[int],
    cum_lengths: List[float]
) -> int:
    """
    Find the nearest river segment (COMID) to the centroid position.
    
    Args:
        centroid: Distance from outlet to centroid (km)
        max_length: Total length of main stem (km)
        comid_list: List of COMIDs along the main stem
        cum_lengths: List of cumulative distances from source
    
    Returns:
        COMID of the river segment closest to the centroid
    """
    if len(comid_list) == 0:
        return -1
    
    # Calculate distance from outlet for each segment
    distances_from_outlet = [max_length - cum for cum in cum_lengths]
    
    # Find index of segment closest to centroid
    nearest_idx = np.argmin(np.abs(np.array(distances_from_outlet) - centroid))
    
    return comid_list[nearest_idx]


def calculate_rci(centroid: float, max_length: float) -> float:
    """
    Calculate the Relative Centroid Index (RCI).
    
    RCI = centroid / max_length
    Values near 0 indicate centroid near outlet, near 1 indicate centroid near source.
    
    Args:
        centroid: Distance from outlet to centroid (km)
        max_length: Total length of main stem (km)
    
    Returns:
        RCI value between 0 and 1
    """
    return centroid / max_length if max_length > 0 else 0


# ============================================================================
# 4. MAIN WORKFLOW FUNCTION
# ============================================================================

def calculate_basin_centroids_from_discharge(
    river_network_path: str,
    discharge_path: str,
    pfafstetter_path: str,
    output_path: str,
    basin_name: str,
    levels: List[int] = [1, 2, 3, 4, 5],
    comid_col: str = 'COMID',
    q_col: str = 'qout',
    length_col: str = 'lengthkm',
    uparea_col: str = 'uparea',
    up_cols: List[str] = None,
    pfaf_col: str = 'pfafstetter',
    min_segments: int = 3
) -> pd.DataFrame:
    """
    Calculate flow centroids for all subbasins at multiple Pfafstetter levels.
    
    This function:
    1. Loads river network, discharge, and Pfafstetter data
    2. For each Pfafstetter level, identifies subbasin outlets
    3. Traces main stems and extracts discharge values
    4. Calculates flow-weighted centroids for each subbasin
    5. Saves results to CSV
    
    Args:
        river_network_path: Path to river network shapefile
        discharge_path: Path to discharge CSV (must contain COMID and qout)
        pfafstetter_path: Path to Pfafstetter codes CSV (COMID, pfafstetter)
        output_path: Path for output CSV file
        basin_name: Name of the basin
        levels: List of Pfafstetter levels to process
        comid_col: Name of COMID column
        q_col: Name of discharge column
        length_col: Name of length column
        uparea_col: Name of upstream area column
        up_cols: List of upstream column names
        pfaf_col: Name of Pfafstetter column
        min_segments: Minimum number of segments required
    
    Returns:
        DataFrame with centroid results for all subbasins
    """
    if up_cols is None:
        up_cols = ['up1', 'up2', 'up3', 'up4']
    
    print("=" * 60)
    print(f"Calculating flow centroids for basin: {basin_name}")
    print("=" * 60)
    
    # ------------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------------
    print("\n[Step 1] Loading data...")
    
    # Load river network
    print(f"  Loading river network from: {river_network_path}")
    gdf = gpd.read_file(river_network_path)
    gdf[comid_col] = gdf[comid_col].astype(int)
    gdf[length_col] = pd.to_numeric(gdf[length_col], errors='coerce').fillna(0)
    print(f"  Loaded {len(gdf)} river segments")
    
    # Load discharge data
    print(f"  Loading discharge data from: {discharge_path}")
    q_df = pd.read_csv(discharge_path)
    q_df[comid_col] = q_df[comid_col].astype(int)
    
    if q_col not in q_df.columns:
        raise ValueError(f"Discharge column '{q_col}' not found. Available: {list(q_df.columns)}")
    
    q_df = q_df[[comid_col, q_col]].dropna()
    print(f"  Loaded discharge data for {len(q_df)} segments")
    
    # Load Pfafstetter codes
    print(f"  Loading Pfafstetter codes from: {pfafstetter_path}")
    pfaf_df = pd.read_csv(pfafstetter_path)
    pfaf_df[comid_col] = pfaf_df[comid_col].astype(int)
    
    if pfaf_col not in pfaf_df.columns:
        raise ValueError(f"Pfafstetter column '{pfaf_col}' not found. Available: {list(pfaf_df.columns)}")
    
    pfaf_df = pfaf_df[[comid_col, pfaf_col]].dropna()
    pfaf_df[pfaf_col] = pfaf_df[pfaf_col].astype(str)
    print(f"  Loaded Pfafstetter codes for {len(pfaf_df)} segments")
    
    # ------------------------------------------------------------------------
    # Step 2: Process each Pfafstetter level
    # ------------------------------------------------------------------------
    all_results = []
    
    for level in levels:
        print(f"\n[Step 2] Processing Pfafstetter level {level}...")
        
        # Find subbasin outlets at this level
        outlets_df = find_subbasin_outlets(
            gdf, pfaf_df, level,
            uparea_col=uparea_col,
            pfaf_col=pfaf_col
        )
        
        if outlets_df.empty:
            print(f"  No outlets found for level {level}")
            continue
        
        print(f"  Found {len(outlets_df)} subbasins at level {level}")
        
        # Process each subbasin
        level_results = []
        
        for _, outlet_row in outlets_df.iterrows():
            outlet_comid = outlet_row['outlet_COMID']
            subbasin_id = outlet_row['subbasin_id']
            
            # Trace main stem
            mainstem_comids = trace_main_stem_from_outlet(
                gdf, outlet_comid, uparea_col, up_cols
            )
            
            if len(mainstem_comids) < min_segments:
                print(f"    Warning: Subbasin {subbasin_id} has only {len(mainstem_comids)} segments, skipping")
                continue
            
            # Calculate cumulative lengths and incremental discharge
            cum_lengths, delta_q = calculate_incremental_discharge(
                gdf, mainstem_comids, q_df,
                comid_col=comid_col,
                q_col=q_col
            )
            
            if len(cum_lengths) < min_segments:
                continue
            
            # Calculate centroid
            max_length, centroid = calculate_centroid(cum_lengths, delta_q)
            
            # Find nearest COMID
            centroid_comid = find_nearest_comid(
                centroid, max_length, mainstem_comids, cum_lengths
            )
            
            # Calculate RCI
            rci = calculate_rci(centroid, max_length)
            
            # Store results
            level_results.append({
                'basin_name': basin_name,
                'level': level,
                'subbasin_code': subbasin_id,
                'outlet_COMID': outlet_comid,
                'centroid_COMID': centroid_comid,
                'centroid_distance_km': centroid,
                'mainstem_length_km': max_length,
                'rci': rci,
                'num_segments': len(mainstem_comids),
                'total_discharge': np.sum(delta_q)
            })
        
        if level_results:
            level_df = pd.DataFrame(level_results)
            all_results.append(level_df)
            print(f"  Calculated centroids for {len(level_results)} subbasins")
    
    # ------------------------------------------------------------------------
    # Step 3: Combine and save results
    # ------------------------------------------------------------------------
    print("\n[Step 3] Saving results...")
    
    if not all_results:
        raise ValueError("No results generated for any level")
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    
    # Save to CSV
    final_df.to_csv(output_path, index=False)
    
    print(f"  Results saved to: {output_path}")
    print(f"  Total records: {len(final_df)}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY BY LEVEL")
    print("=" * 60)
    summary = final_df.groupby('level').agg({
        'subbasin_code': 'count',
        'rci': 'mean',
        'mainstem_length_km': 'mean',
        'total_discharge': 'mean'
    }).round(3)
    summary.columns = ['num_basins', 'mean_rci', 'mean_length_km', 'mean_discharge']
    print(summary)
    print("=" * 60)
    
    return final_df


# ============================================================================
# 5. MAIN FUNCTION - EDIT WITH YOUR DATA PATHS
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
    
    # Path to your discharge data CSV
    # Must contain columns: COMID, qout (or your discharge column name)
    DISCHARGE_PATH = "data/poyang_discharge.csv"
    
    # Path to your Pfafstetter codes CSV
    # Must contain columns: COMID, pfafstetter
    PFAFSTETTER_PATH = "data/r1_yangtze_pfafstetter_original.csv"
    
    # Output path for results
    OUTPUT_PATH = "centroid_results_levels.csv"
    
    # Name of your basin
    BASIN_NAME = "poyang"
    
    # Pfafstetter levels to process (1-5)
    LEVELS = [1, 2, 3, 4, 5]
    
    # Column names in your files (modify if different)
    COMID_COL = "COMID"
    Q_COL = "qout"  # Name of discharge column in your CSV
    LENGTH_COL = "lengthkm"
    UPAREA_COL = "uparea"
    UP_COLS = ["up1", "up2", "up3", "up4"]
    PFAF_COL = "pfafstetter"
    
    # Minimum number of segments required on main stem
    MIN_SEGMENTS = 3
    
    # =========================================================================
    # END OF USER INPUT SECTION
    # =========================================================================
    
    # Validate input files exist
    if not os.path.exists(RIVER_NETWORK_PATH):
        raise FileNotFoundError(f"River network file not found: {RIVER_NETWORK_PATH}")
    
    if not os.path.exists(DISCHARGE_PATH):
        raise FileNotFoundError(f"Discharge file not found: {DISCHARGE_PATH}")
    
    if not os.path.exists(PFAFSTETTER_PATH):
        raise FileNotFoundError(f"Pfafstetter file not found: {PFAFSTETTER_PATH}")
    
    # Run the centroid calculation
    results = calculate_basin_centroids_from_discharge(
        river_network_path=RIVER_NETWORK_PATH,
        discharge_path=DISCHARGE_PATH,
        pfafstetter_path=PFAFSTETTER_PATH,
        output_path=OUTPUT_PATH,
        basin_name=BASIN_NAME,
        levels=LEVELS,
        comid_col=COMID_COL,
        q_col=Q_COL,
        length_col=LENGTH_COL,
        uparea_col=UPAREA_COL,
        up_cols=UP_COLS,
        pfaf_col=PFAF_COL,
        min_segments=MIN_SEGMENTS
    )
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Basin: {BASIN_NAME}")
    print(f"Total subbasins processed: {len(results)}")
    print(f"Levels processed: {sorted(results['level'].unique())}")
    print(f"Average RCI: {results['rci'].mean():.4f}")
    print(f"Average main stem length: {results['mainstem_length_km'].mean():.2f} km")
    print(f"Average discharge: {results['total_discharge'].mean():.2f}")
    print("=" * 60)
    
    # Show first few results
    print("\nSample results (first 5 rows):")
    print(results[['level', 'subbasin_code', 'centroid_COMID', 'rci']].head())


if __name__ == "__main__":
    main()