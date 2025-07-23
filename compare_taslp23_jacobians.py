#!/usr/bin/env python3
"""
Numerical comparison of TASLP23 outputs from 'make jt23' vs 'make jt23f'
Compares ./outputs/taslp23/x/ and ./outputs/taslp23_fast/x/
"""

import h5py
import numpy as np
import os
from typing import Dict, Tuple, List
from pathlib import Path

def load_h5_structure(filepath: str) -> Dict:
    """Load and return the structure and data from an HDF5 file"""
    result = {}
    with h5py.File(filepath, 'r') as f:
        def walk_h5(name, obj):
            if isinstance(obj, h5py.Dataset):
                result[name] = np.array(obj)
        f.visititems(walk_h5)
    return result

def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str) -> Dict:
    """Compare two numpy arrays and return detailed statistics"""
    if arr1.shape != arr2.shape:
        return {
            'name': name,
            'shapes_match': False,
            'shape1': arr1.shape,
            'shape2': arr2.shape,
            'error': 'Shape mismatch'
        }

    # Compute differences
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff) / (np.abs(arr1) + 1e-10)  # Add small epsilon to avoid division by zero

    # Basic statistics
    stats = {
        'name': name,
        'shapes_match': True,
        'shape': arr1.shape,
        'dtype': str(arr1.dtype),

        # Absolute differences
        'max_abs_diff': float(np.max(abs_diff)),
        'mean_abs_diff': float(np.mean(abs_diff)),
        'std_abs_diff': float(np.std(abs_diff)),
        'rms_diff': float(np.sqrt(np.mean(diff**2))),

        # Relative differences
        'max_rel_diff': float(np.max(rel_diff)),
        'mean_rel_diff': float(np.mean(rel_diff)),

        # Value ranges
        'arr1_min': float(np.min(arr1)),
        'arr1_max': float(np.max(arr1)),
        'arr2_min': float(np.min(arr2)),
        'arr2_max': float(np.max(arr2)),

        # Exact equality
        'num_exact_matches': int(np.sum(arr1 == arr2)),
        'num_total_elements': int(arr1.size),
        'fraction_exact_matches': float(np.sum(arr1 == arr2)) / arr1.size,

        # Near equality (within floating point precision)
        'num_close_matches': int(np.sum(np.isclose(arr1, arr2, rtol=1e-15, atol=1e-15))),
        'fraction_close_matches': float(np.sum(np.isclose(arr1, arr2, rtol=1e-15, atol=1e-15))) / arr1.size,

        # Audio-specific metrics (assuming audio data)
        'snr_db': float(10 * np.log10(np.mean(arr1**2) / (np.mean(diff**2) + 1e-10))),
    }

    return stats

def compare_h5_files(file1: str, file2: str) -> Tuple[List[Dict], bool]:
    """Compare two HDF5 files and return detailed comparison results"""

    print(f"Loading {file1}...")
    data1 = load_h5_structure(file1)
    print(f"Loading {file2}...")
    data2 = load_h5_structure(file2)

    # Check if they have the same keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    if keys1 != keys2:
        print(f"WARNING: Different datasets found!")
        print(f"  Only in file1: {keys1 - keys2}")
        print(f"  Only in file2: {keys2 - keys1}")
        return [], False

    results = []
    all_identical = True

    for key in sorted(keys1):
        print(f"  Comparing dataset: {key}")
        result = compare_arrays(data1[key], data2[key], key)
        results.append(result)

        # Check if arrays are identical
        if result.get('max_abs_diff', float('inf')) > 0:
            all_identical = False

    return results, all_identical

def print_comparison_summary(results: List[Dict], file1: str, file2: str, all_identical: bool):
    """Print a formatted summary of comparison results"""

    print("\n" + "="*80)
    print(f"COMPARISON SUMMARY")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print("="*80)

    if all_identical:
        print("✓ FILES ARE NUMERICALLY IDENTICAL")
        return

    print("Files differ. Detailed analysis:")
    print()

    for result in results:
        name = result['name']
        print(f"Dataset: {name}")
        print(f"  Shape: {result['shape']}")
        print(f"  Data type: {result['dtype']}")

        if result['max_abs_diff'] == 0:
            print("  ✓ IDENTICAL")
        else:
            print(f"  Max absolute difference: {result['max_abs_diff']:.2e}")
            print(f"  RMS difference: {result['rms_diff']:.2e}")
            print(f"  Mean absolute difference: {result['mean_abs_diff']:.2e}")
            print(f"  Max relative difference: {result['max_rel_diff']:.2e}")
            print(f"  SNR: {result['snr_db']:.1f} dB")
            print(f"  Exact matches: {result['fraction_exact_matches']:.1%}")
            print(f"  Close matches (1e-15 tol): {result['fraction_close_matches']:.1%}")

            # Classification
            if result['max_abs_diff'] < 1e-15:
                print("  → Differences likely due to floating point precision")
            elif result['max_abs_diff'] < 1e-10:
                print("  → Very small numerical differences")
            elif result['max_abs_diff'] < 1e-6:
                print("  → Small numerical differences")
            else:
                print("  → Significant numerical differences")

        print()

def main():
    """Main comparison function"""

    # Define paths
    base_dir = Path("./outputs")
    taslp23_dir = base_dir / "taslp23" / "x"
    taslp23_fast_dir = base_dir / "taslp23_fast" / "x"

    # Check if directories exist
    if not taslp23_dir.exists():
        print(f"ERROR: Directory not found: {taslp23_dir}")
        return 1

    if not taslp23_fast_dir.exists():
        print(f"ERROR: Directory not found: {taslp23_fast_dir}")
        return 1

    # Find HDF5 files
    h5_files = [f for f in os.listdir(taslp23_dir) if f.endswith('.h5')]

    if not h5_files:
        print(f"ERROR: No HDF5 files found in {taslp23_dir}")
        return 1

    print(f"Found {len(h5_files)} HDF5 files to compare:")
    for f in sorted(h5_files):
        print(f"  {f}")
    print()

    # Compare each file
    all_files_identical = True

    for h5_file in sorted(h5_files):
        file1 = taslp23_dir / h5_file
        file2 = taslp23_fast_dir / h5_file

        if not file2.exists():
            print(f"ERROR: File not found in fast directory: {file2}")
            continue

        print(f"\n{'*'*60}")
        print(f"COMPARING: {h5_file}")
        print('*'*60)

        results, files_identical = compare_h5_files(str(file1), str(file2))

        if results:  # Only print if we got valid results
            print_comparison_summary(results, str(file1), str(file2), files_identical)

            if not files_identical:
                all_files_identical = False

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    if all_files_identical:
        print("✓ ALL FILES ARE NUMERICALLY IDENTICAL")
        print("The 'fast' and 'regular' TASLP23 audio generations produced identical results.")
    else:
        print("⚠ FILES DIFFER")
        print("The 'fast' and 'regular' TASLP23 audio generations produced different results.")
        print("See detailed analysis above for specific differences.")

    return 0 if all_files_identical else 1

if __name__ == "__main__":
    exit(main())
