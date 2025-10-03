import sys
from lookup_activity_by_pt_root_id import lookup_activity_by_pt_root_ids
from caveclient import CAVEclient
import pandas as pd
import argparse

def main(args):
    print("Testing activity lookup by pt_root_id")
    print("="*60)

    print("\n1. Loading coregistration table from database...")
    # client = CAVEclient()
    # client.auth.save_token(token="e4c43e265b8b98779ec2f7c906212703")
    client = CAVEclient('minnie65_public')
    client.materialize.version = 1507

    # coregistration_df = client.materialize.query_table('coregistration_manual_v4')
    response1 = client.materialize.query_table(
        'coregistration_manual_v4',
        return_df=False
    )
    coregistration_df = pd.DataFrame(response1)
    print(f"Loaded coregistration table with {len(coregistration_df)} entries")
    print(f"Columns: {coregistration_df.columns.tolist()}")

    print("\n2. Example pt_root_ids in the coregistration table:")
    sample_pt_root_ids = coregistration_df['pt_root_id'].head(10).tolist()
    for i, pt_id in enumerate(sample_pt_root_ids[:5], 1):
        print(f"   {i}. {pt_id}")

    if args.debug:
        print("\n3. Testing lookup with first 3 pt_root_ids...")
        test_pt_root_ids = sample_pt_root_ids[:3]
        print(f"   {len(test_pt_root_ids)} pt_root_ids to test")
    else:
        test_pt_root_ids = args.pt_root_ids
        print(f"   {len(test_pt_root_ids)} pt_root_ids to test")
        print("\n3. Testing lookup with all pt_root_ids...")

    # metadata only, fast
    results = lookup_activity_by_pt_root_ids(
        pt_root_ids=test_pt_root_ids,
        coregistration_df=coregistration_df,
        scan_dir=args.scan_dir,
        return_full_data=False
    )

    print("\n4. Results (metadata only):")
    print(results)

    # now get the full data with traces
    print("\n5. Getting full data with activity traces...")
    results_full = lookup_activity_by_pt_root_ids(
        pt_root_ids=test_pt_root_ids,
        coregistration_df=coregistration_df,
        scan_dir=args.scan_dir,
        return_full_data=True
    )

    print("\n6. Full results columns:")
    print(f"   {results_full.columns.tolist()}")

    if len(results_full) > 0:
        print("\n7. Sample of first result:")
        first_row = results_full.iloc[0]
        print(f"   pt_root_id: {first_row['pt_root_id']}")
        print(f"   unit_id: {first_row['unit_id']}")
        print(f"   session: {first_row['session']}")
        print(f"   scan_idx: {first_row['scan_idx']}")
        print(f"   nframes: {first_row['nframes']}")
        print(f"   fps: {first_row['fps']}")
        if 'spike_trace' in first_row:
            print(f"   spike_trace shape: {first_row['spike_trace'].shape}")
        if 'calcium_trace' in first_row:
            print(f"   calcium_trace shape: {first_row['calcium_trace'].shape}")

    print("\n" + "="*60)
    print("Done")

    return results, results_full


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test activity lookup by pt_root_id")
    parser.add_argument(
        "--scan_dir",
        type=str,
        default="/Users/artliang/Documents/Myelin Plasticity",
        help="Directory containing scan_*_*_coreg.zip files (default: %(default)s)"
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Whether to run in debug mode (default: %(default)s)"
    )
    parser.add_argument(
        "--pt_root_ids",
        type=list,
        default=None,
        help="List of pt_root_ids to test (default: %(default)s)"
    )
    args = parser.parse_args()
    results_meta, results_full = main(args)
