import os
import pickle
import zipfile
import pandas as pd
from typing import List
import tempfile


def lookup_activity_by_pt_root_ids(
    pt_root_ids: List[int],
    coregistration_df: pd.DataFrame,
    scan_dir: str = ".",
    return_full_data: bool = False
) -> pd.DataFrame:
    """
    Look up activity data for specific pt_root_ids without loading all files into RAM.

    This function:
    1. Finds which unit_ids correspond to the given pt_root_ids
    2. Identifies which scan zip files to check
    3. Iterates through relevant zip files one at a time
    4. Extracts only the matching unit data

    Parameters:
    -----------
    pt_root_ids : list of int
        List of pt_root_id values to look up
    coregistration_df : pd.DataFrame
        DataFrame with columns: pt_root_id, unit_id, session, scan_idx
    scan_dir : str
        Directory containing the scan_*_*_coreg.zip files
    return_full_data : bool
        If True, returns full activity traces. If False, returns only metadata

    Returns:
    --------
    pd.DataFrame with columns:
        - pt_root_id
        - unit_id
        - session
        - scan_idx
        - [if return_full_data=True: nframes, fps, spike_trace, calcium_trace]
    """
    # Filter coregistration table to only the pt_root_ids we want
    matches = coregistration_df[coregistration_df['pt_root_id'].isin(pt_root_ids)].copy()

    if len(matches) == 0:
        print(f"⚠️ No matches found for the provided pt_root_ids")
        return pd.DataFrame()

    print(f"Found {len(matches)} matches for {len(pt_root_ids)} pt_root_ids")
    print(f"  - Spanning {matches['session'].nunique()} sessions")
    print(f"  - Spanning {matches.groupby(['session', 'scan_idx']).ngroups} scans")

    # Group by scan to know which zip files to check
    scan_groups = matches.groupby(['session', 'scan_idx'])

    results = []

    # Iterate through each scan zip file
    for (session, scan_idx), group in scan_groups:
        zip_filename = f"scan_{session}_{scan_idx}_coreg.zip"
        zip_path = os.path.join(scan_dir, zip_filename)

        if not os.path.exists(zip_path):
            print(f"⚠️ Warning: {zip_filename} not found, skipping")
            continue

        print(f"Processing {zip_filename} ({len(group)} units to extract)...")

        # Get the unit_ids we need from this scan
        target_unit_ids = set(group['unit_id'].values)

        # Create a temporary directory to extract only what we need
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Get list of files in the zip
                all_files = zf.namelist()

                # Extract only the unit files we need
                for filename in all_files:
                    # Parse unit_id from filename (e.g., "unit_1234.pkl")
                    if filename.startswith('unit_') and filename.endswith('.pkl'):
                        try:
                            unit_id = int(filename.replace('unit_', '').replace('.pkl', ''))

                            if unit_id in target_unit_ids:
                                zf.extract(filename, temp_dir)

                                # Load the pickle file
                                pkl_path = os.path.join(temp_dir, filename)
                                with open(pkl_path, 'rb') as f:
                                    data = pickle.load(f)

                                # Get corresponding pt_root_id from the matches
                                pt_root_id_row = group[group['unit_id'] == unit_id]
                                if len(pt_root_id_row) > 0:
                                    pt_root_id = pt_root_id_row.iloc[0]['pt_root_id']

                                    result = {
                                        'pt_root_id': pt_root_id,
                                        'unit_id': unit_id,
                                        'session': session,
                                        'scan_idx': scan_idx,
                                    }

                                    if return_full_data:
                                        # all data from the pickle
                                        result.update(data)
                                    else:
                                        # metadata
                                        result['nframes'] = data.get('nframes', None)
                                        result['fps'] = data.get('fps', None)

                                    results.append(result)
                        except (ValueError, KeyError) as e:
                            print(f"  Warning: Could not parse {filename}: {e}")
                            continue

    results_df = pd.DataFrame(results)

    print(f"\nExtracted {len(results_df)} units")

    return results_df
