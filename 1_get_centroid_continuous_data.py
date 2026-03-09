"""
River Network Centroid Analysis

This module calculates the mass centroid of a river basin using continuous river network
data (discharge) along the main stem.

The centroid represents the point along the main stem where half of the total
accumulated mass is upstream and half downstream.

Input:
    - River network shapefile with topology
    - Discharge CSV (COMID, q)

Output:
    - CSV file with basin name, centroid location (distance to outlet), centroid COMID, and RCI

Author: Ziyun Yin
Date: 04/03/2026
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from typing import List, Tuple


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


def extract_main_stem_data(
    gdf: gpd.GeoDataFrame,
    outlet_comid: int,
    q_df: pd.DataFrame,
    comid_col: str = 'COMID',
    q_col: str = 'q',
    length_col: str = 'lengthkm',
    uparea_col: str = 'uparea',
    up_cols: List[str] = None
) -> Tuple[List[int], List[float], List[float]]:
    """
    Extract main stem COMIDs, cumulative lengths, and calculate delta Q.
    
    The logic is simple:
    1. Trace main stem to get ordered COMID list
    2. Get Q values for each COMID
    3. Calculate delta Q = Q_i - Q_{i-1} (with delta Q_0 = Q_0)
    
    Args:
        gdf: GeoDataFrame with river network data
        outlet_comid: COMID of the outlet segment
        q_df: DataFrame with discharge values for each COMID
        comid_col: Name of COMID column
        q_col: Name of discharge column
        length_col: Name of length column
        uparea_col: Name of upstream area column
        up_cols: List of upstream column names
    
    Returns:
        Tuple containing:
            - List of COMIDs along the main stem (upstream to downstream)
            - List of cumulative lengths from source (km)
            - List of delta Q values (incremental discharge)
    """
    # Trace main stem (returns from upstream to downstream)
    comid_list = trace_main_stem_from_outlet(gdf, outlet_comid, uparea_col, up_cols)
    
    if not comid_list:
        return [], [], []
    
    # Create discharge lookup dictionary
    q_dict = q_df.set_index(comid_col)[q_col].to_dict()
    
    # Create length lookup dictionary
    length_dict = gdf.set_index(comid_col)[length_col].to_dict()
    
    # Calculate cumulative lengths and delta Q
    comids_valid = []
    cum_lengths = []
    q_values = []
    
    # First pass: collect valid COMIDs and their Q values
    cum_length = 0
    for comid in comid_list:
        if comid in q_dict and comid in length_dict:
            comids_valid.append(comid)
            cum_length += length_dict[comid]
            cum_lengths.append(cum_length)
            q_values.append(q_dict[comid])
    
    # Calculate delta Q: Q_i - Q_{i-1} (with delta Q_0 = Q_0)
    delta_q_list = []
    for i, q in enumerate(q_values):
        if i == 0:
            delta_q = q
        else:
            delta_q = q - q_values[i-1]
        delta_q_list.append(delta_q)
    return comids_valid, cum_lengths, delta_q_list


def calculate_centroid(
    cumulative_lengths: List[float], 
    delta_q: List[float]
) -> Tuple[float, float]:
    """
    Calculate the flow centroid and total length of a river main stem.
    
    The centroid represents the point along the main stem where half of the total
    accumulated flow is upstream and half downstream.
    
    Args:
        cumulative_lengths: List of cumulative distances from source (km)
        delta_q: List of incremental discharge values at each segment
    
    Returns:
        Tuple containing (max_length, centroid_position):
            - max_length: Total length of the main stem (km)
            - centroid: Distance from outlet to centroid (km)
    """
    if len(cumulative_lengths) == 0 or len(delta_q) == 0:
        return 0, 0
    
    cum_array = np.array(cumulative_lengths)
    delta_array = np.array(delta_q)
    total_q = delta_array.sum()
    
    if total_q == 0:
        # If no flow, define centroid as the midpoint
        max_length = cum_array[-1]
        return max_length, max_length / 2
    
    max_length = cum_array[-1]
    
    # Calculate centroid (distance from source)
    weighted_sum = np.sum(cum_array * delta_array)
    centroid_from_source = weighted_sum / total_q
    
    # Convert to distance from outlet
    centroid_from_outlet = max_length - centroid_from_source
    
    return max_length, centroid_from_outlet


def find_centroid_comid(
    centroid: float,
    max_length: float,
    comid_list: List[int],
    cum_lengths: List[float]
) -> int:
    """
    Find the river segment (COMID) containing the centroid position.
    
    Args:
        centroid: Distance from outlet to centroid (km)
        max_length: Total length of main stem (km)
        comid_list: List of COMIDs along the main stem
        cum_lengths: List of cumulative distances from source
    
    Returns:
        COMID of the river segment containing the centroid
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


def calculate_basin_centroid(
    river_network_path: str,
    discharge_path: str,
    output_path: str,
    basin_name: str,
    q_col: str = 'q',
    comid_col: str = 'COMID',
    length_col: str = 'lengthkm',
    uparea_col: str = 'uparea',
    up_cols: List[str] = None,
    min_segments: int = 3
) -> pd.DataFrame:
    """
    Calculate the centroid for a river basin using discharge data.
    
    This function:
    1. Loads river network and discharge data
    2. Identifies the basin outlet
    3. Traces the main stem
    4. Calculates delta Q along the main stem
    5. Computes the flow-weighted centroid
    6. Saves results to CSV
    
    Args:
        river_network_path: Path to river network shapefile
        discharge_path: Path to discharge CSV
        output_path: Path for output CSV file
        basin_name: Name of the basin
        q_col: Name of discharge column in CSV
        comid_col: Name of COMID column in all files
        length_col: Name of length column in river network
        uparea_col: Name of upstream area column
        up_cols: List of upstream column names
        min_segments: Minimum number of segments required
    
    Returns:
        DataFrame with centroid results
    """
    if up_cols is None:
        up_cols = ['up1', 'up2', 'up3', 'up4']
    
    print("=" * 60)
    print(f"Calculating centroid for basin: {basin_name}")
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
    
    # Ensure discharge column exists
    if q_col not in q_df.columns:
        raise ValueError(f"Discharge column '{q_col}' not found. Available columns: {list(q_df.columns)}")
    
    q_df = q_df[[comid_col, q_col]].dropna()
    print(f"  Loaded discharge data for {len(q_df)} segments")
    
    # ------------------------------------------------------------------------
    # Step 2: Find basin outlet
    # ------------------------------------------------------------------------
    print("\n[Step 2] Identifying basin outlet...")
    
    outlet_comid = find_basin_outlet(gdf, uparea_col)
    outlet_area = gdf[gdf[comid_col] == outlet_comid][uparea_col].values[0]
    
    print(f"  Outlet COMID: {outlet_comid}")
    print(f"  Upstream area: {outlet_area:.2f} km²")
    
    # ------------------------------------------------------------------------
    # Step 3: Extract main stem data
    # ------------------------------------------------------------------------
    print("\n[Step 3] Tracing main stem and calculating delta Q...")
    
    comid_list, cum_lengths, delta_q = extract_main_stem_data(
        gdf, outlet_comid, q_df,
        comid_col=comid_col,
        q_col=q_col,
        length_col=length_col,
        uparea_col=uparea_col,
        up_cols=up_cols
    )
    
    if len(comid_list) < min_segments:
        raise ValueError(f"Main stem has only {len(comid_list)} segments (minimum {min_segments} required)")
    
    print(f"  Main stem length: {len(comid_list)} segments")
    print(f"  Total length: {cum_lengths[-1]:.2f} km")
    print(f"  Outlet discharge: {q_df[q_df[comid_col] == outlet_comid][q_col].values[0]:.2f}")
    print(f"  Sum of delta Q: {np.sum(delta_q):.2f} (should equal outlet discharge)")
    
    # ------------------------------------------------------------------------
    # Step 4: Calculate centroid
    # ------------------------------------------------------------------------
    print("\n[Step 4] Calculating centroid...")
    
    max_length, centroid = calculate_centroid(cum_lengths, delta_q)
    centroid_comid = find_centroid_comid(centroid, max_length, comid_list, cum_lengths)
    rci = calculate_rci(centroid, max_length)
    
    print(f"  Centroid distance from outlet: {centroid:.2f} km")
    print(f"  Centroid COMID: {centroid_comid}")
    print(f"  RCI: {rci:.4f}")
    
    # ------------------------------------------------------------------------
    # Step 5: Create and save results
    # ------------------------------------------------------------------------
    print("\n[Step 5] Saving results...")
    
    results = pd.DataFrame([{
        'basin_name': basin_name,
        'outlet_COMID': outlet_comid,
        'centroid_COMID': centroid_comid,
        'centroid_distance_km': centroid,
        'mainstem_length_km': max_length,
        'rci': rci,
        'num_segments': len(comid_list),
        'total_discharge': np.sum(delta_q),
        'outlet_uparea_km2': outlet_area
    }])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    
    # Save to CSV
    results.to_csv(output_path, index=False)
    
    print(f"  Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Calculation complete!")
    print("=" * 60)
    
    return results


def main():
    """
    Main function - EDIT THIS SECTION WITH YOUR DATA PATHS
    """
    # =========================================================================
    # USER INPUT SECTION - PLEASE MODIFY THESE PATHS
    # =========================================================================
    
    # Path to your river network shapefile
    RIVER_NETWORK_PATH = "data/Ganjiang_reach.shp"
    
    # Path to your discharge data CSV file
    DISCHARGE_PATH = "data/poyang_discharge.csv"
    
    # Output path for results
    OUTPUT_PATH = "centroid_results.csv"
    
    # Name of your basin
    BASIN_NAME = "Poyang"
    
    # Name of the discharge column in your CSV file
    Q_COLUMN = "qout"
    
    # Column names in your shapefile
    COMID_COL = "COMID"
    LENGTH_COL = "lengthkm"
    UPAREA_COL = "uparea"
    UP_COLS = ["up1", "up2", "up3", "up4"]
    
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
    
    # Run the centroid calculation
    results = calculate_basin_centroid(
        river_network_path=RIVER_NETWORK_PATH,
        discharge_path=DISCHARGE_PATH,
        output_path=OUTPUT_PATH,
        basin_name=BASIN_NAME,
        q_col=Q_COLUMN,
        comid_col=COMID_COL,
        length_col=LENGTH_COL,
        uparea_col=UPAREA_COL,
        up_cols=UP_COLS,
        min_segments=MIN_SEGMENTS
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
    print(f"Number of segments: {results['num_segments'].values[0]}")
    print(f"Total discharge: {results['total_discharge'].values[0]:.2f}")
    print(f"Outlet upstream area: {results['outlet_uparea_km2'].values[0]:.2f} km²")
    print("=" * 60)


if __name__ == "__main__":
    main()