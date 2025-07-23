#!/usr/bin/env python3
"""
Numerical comparison of TASLP23 M matrices from Mt23, Mt23f, and Mt23fp
Runs all three variants and compares the resulting M, sigma, and JdagJ matrices
"""

import argparse
import h5py
import numpy as np
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple, List


def run_command(cmd: List[str], description: str, cwd: str = None) -> Tuple[bool, float]:
    """Run a shell command and return success status and elapsed time"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    start_time = time.time()

    try:
        # Use simpler subprocess.run with timeout for better control
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,  # Don't raise exception on non-zero exit
            timeout=None,  # No timeout - let user interrupt
            text=True
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            print(f"⏱ Elapsed time: {format_duration(elapsed_time)}")
            return True, elapsed_time
        else:
            print(f"✗ FAILED: {description}")
            print(f"Return code: {result.returncode}")
            print(f"⏱ Elapsed time: {format_duration(elapsed_time)}")
            return False, elapsed_time

    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"\n⚠ KeyboardInterrupt received!")
        print(f"⏱ Elapsed time before interruption: {format_duration(elapsed_time)}")
        raise  # Re-raise to be caught by main

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"✗ TIMEOUT: {description}")
        print(f"⏱ Elapsed time: {format_duration(elapsed_time)}")
        return False, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ ERROR: {description}")
        print(f"Exception: {e}")
        print(f"⏱ Elapsed time: {format_duration(elapsed_time)}")
        return False, elapsed_time


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def load_M_matrices(h5_path: str) -> Dict[str, Dict]:
    """Load M, sigma, and JdagJ matrices from HDF5 file"""
    matrices = {'M': {}, 'sigma': {}, 'JdagJ': {}}

    try:
        with h5py.File(h5_path, 'r') as f:
            # Load M matrices
            if 'M' in f:
                for key in f['M'].keys():
                    matrices['M'][key] = np.array(f['M'][key])

            # Load sigma (eigenvalues)
            if 'sigma' in f:
                for key in f['sigma'].keys():
                    matrices['sigma'][key] = np.array(f['sigma'][key])

            # Load JdagJ matrices (if present)
            if 'JdagJ' in f:
                for key in f['JdagJ'].keys():
                    matrices['JdagJ'][key] = np.array(f['JdagJ'][key])

    except Exception as e:
        print(f"Error loading {h5_path}: {e}")

    return matrices


def compare_matrix_sets(matrices1: Dict, matrices2: Dict, name1: str, name2: str) -> Dict:
    """Compare two sets of matrices and return detailed statistics"""
    results = {'M': {}, 'sigma': {}, 'JdagJ': {}}

    for matrix_type in ['M', 'sigma', 'JdagJ']:
        type_results = {}

        # Get common keys
        keys1 = set(matrices1[matrix_type].keys())
        keys2 = set(matrices2[matrix_type].keys())
        common_keys = keys1.intersection(keys2)

        if not common_keys:
            type_results['error'] = f"No common {matrix_type} matrices found"
            results[matrix_type] = type_results
            continue

        # Compare each matrix
        max_diffs = []
        rms_diffs = []
        exact_matches = 0
        close_matches = 0
        total_matrices = len(common_keys)

        for key in sorted(common_keys):
            arr1 = matrices1[matrix_type][key]
            arr2 = matrices2[matrix_type][key]

            if arr1.shape != arr2.shape:
                continue

            diff = arr1 - arr2
            abs_diff = np.abs(diff)
            max_diff = float(np.max(abs_diff).real if np.iscomplexobj(abs_diff) else np.max(abs_diff))
            rms_diff = float(np.sqrt(np.mean(diff**2)).real if np.iscomplexobj(diff) else np.sqrt(np.mean(diff**2)))

            max_diffs.append(max_diff)
            rms_diffs.append(rms_diff)

            if max_diff == 0:
                exact_matches += 1
            elif np.allclose(arr1, arr2, rtol=1e-15, atol=1e-15):
                close_matches += 1

        # Aggregate statistics
        if max_diffs:
            type_results.update({
                'total_matrices': total_matrices,
                'max_abs_diff': float(np.max(max_diffs)),
                'mean_max_diff': float(np.mean(max_diffs)),
                'max_rms_diff': float(np.max(rms_diffs)),
                'mean_rms_diff': float(np.mean(rms_diffs)),
                'exact_matches': exact_matches,
                'close_matches': close_matches,
                'fraction_exact': exact_matches / total_matrices,
                'fraction_close': (exact_matches + close_matches) / total_matrices,
            })

        results[matrix_type] = type_results

    return results


def print_comparison_results(results: Dict, name1: str, name2: str):
    """Print formatted comparison results"""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {name1} vs {name2}")
    print('='*80)

    for matrix_type in ['M', 'sigma', 'JdagJ']:
        print(f"\n{matrix_type} Matrices:")
        print('-' * 40)

        type_results = results[matrix_type]

        if 'error' in type_results:
            print(f"  ⚠ {type_results['error']}")
            continue

        if not type_results:
            print(f"  No {matrix_type} matrices found")
            continue

        total = type_results['total_matrices']
        exact = type_results['exact_matches']
        close = type_results['close_matches']

        print(f"  Total matrices compared: {total}")
        print(f"  Exact matches: {exact} ({type_results['fraction_exact']:.1%})")
        print(f"  Close matches (1e-15 tol): {close} ({type_results['fraction_close']:.1%})")

        if type_results['max_abs_diff'] == 0:
            print("  ✓ ALL MATRICES IDENTICAL")
        else:
            print(f"  Max absolute difference: {type_results['max_abs_diff']:.2e}")
            print(f"  Mean max difference: {type_results['mean_max_diff']:.2e}")
            print(f"  Max RMS difference: {type_results['max_rms_diff']:.2e}")
            print(f"  Mean RMS difference: {type_results['mean_rms_diff']:.2e}")

            # Classification
            max_diff = type_results['max_abs_diff']
            if max_diff < 1e-15:
                print("  → Differences likely due to floating point precision")
            elif max_diff < 1e-10:
                print("  → Very small numerical differences")
            elif max_diff < 1e-6:
                print("  → Small numerical differences")
            else:
                print("  → Significant numerical differences")


def run_M_computation(variant: str, temp_dir: Path, n_samples: int) -> Tuple[bool, float]:
    """Run one variant of M matrix computation"""
    output_dir = temp_dir / variant
    output_dir.mkdir(exist_ok=True)

    # Map variant names to make commands
    variant_map = {
        'original': 'Mt23',
        'fast': 'Mt23f',
        'precision': 'Mt23fp'
    }

    make_target = variant_map[variant]

    # Use direct Python execution with activated virtual environment
    venv_python = Path.cwd() / '.venv' / 'bin' / 'python'

    if variant == 'original':
        cmd = [
            str(venv_python),
            'taslp23/12_compute_lmastep.py',
            str(output_dir / 'taslp23'),
            '0',
            str(n_samples)
        ]
    else:
        force_cpu = '1' if variant == 'precision' else '0'
        cmd = [
            str(venv_python),
            'taslp23/12_compute_lmastep_fast.py',
            str(output_dir / 'taslp23'),
            '0',
            str(n_samples),
            force_cpu
        ]

    return run_command(
        cmd,
        f"Computing M matrices ({variant}) for {n_samples} samples",
        cwd=str(Path.cwd())
    )


def cleanup_temp_directory(temp_dir: Path, force: bool = False):
    """Clean up temporary directory"""
    if temp_dir.exists() and (force or temp_dir.name.startswith('taslp23_M_comparison_')):
        print(f"\nCleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up {temp_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare TASLP23 M matrices from Mt23, Mt23f, and Mt23fp variants"
    )
    parser.add_argument(
        'n_samples',
        type=int,
        default=100,
        nargs='?',
        help="Number of samples to process (default: 100)"
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        help="Temporary directory for outputs (default: auto-generated)"
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help="Keep temporary directory after completion"
    )

    args = parser.parse_args()

    print("TASLP23 M Matrix Comparison Tool")
    print("=" * 50)
    print(f"Processing {args.n_samples} samples")
    print("Press Ctrl+C to interrupt and clean up gracefully")

    # Create temporary directory
    if args.temp_dir:
        temp_base = Path(args.temp_dir)
        temp_base.mkdir(exist_ok=True)
        temp_dir = temp_base
        cleanup_temp = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix='taslp23_M_comparison_'))
        cleanup_temp = not args.keep_temp

    print(f"Temporary directory: {temp_dir}")

    def cleanup_and_exit(signum=None, frame=None):
        print(f"\n⚠ Received interrupt signal. Cleaning up...")
        if cleanup_temp:
            cleanup_temp_directory(temp_dir, force=True)
        print("Cleanup complete. Exiting.")
        sys.exit(1)

    # Set up signal handlers for the main script
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    try:
        # Run all three variants
        variants = ['original', 'fast', 'precision']
        success_flags = {}
        timing_results = {}

        for variant in variants:
            print(f"\n{'*'*60}")
            print(f"RUNNING VARIANT: {variant}")
            print('*'*60)

            success, elapsed_time = run_M_computation(variant, temp_dir, args.n_samples)
            success_flags[variant] = success
            timing_results[variant] = elapsed_time

            if not success:
                print(f"⚠ Warning: {variant} variant failed")

        # Check for successful runs
        successful_variants = [v for v, success in success_flags.items() if success]

        if len(successful_variants) < 2:
            print("\n❌ ERROR: Need at least 2 successful runs for comparison")
            return 1

        print(f"\n✓ Successfully completed {len(successful_variants)} variants: {successful_variants}")

        # Report timing results
        print(f"\n{'='*80}")
        print("TIMING SUMMARY")
        print('='*80)

        timing_data = []
        for variant in variants:
            if variant in timing_results:
                elapsed = timing_results[variant]
                status = "SUCCESS" if success_flags[variant] else "FAILED"
                timing_data.append((variant, elapsed, status))
                print(f"{variant:>10}: {format_duration(elapsed):>12} ({status})")

        # Calculate speedup ratios if we have successful runs
        successful_times = [(v, t) for v, t, s in timing_data if s == "SUCCESS"]

        if len(successful_times) >= 2:
            print(f"\nSPEEDUP ANALYSIS:")
            print("-" * 40)

            # Find original as baseline
            original_time = None
            for variant, elapsed in successful_times:
                if variant == 'original':
                    original_time = elapsed
                    break

            if original_time:
                for variant, elapsed in successful_times:
                    if variant != 'original':
                        speedup = original_time / elapsed
                        if speedup > 1:
                            print(f"{variant:>10}: {speedup:.2f}x faster than original")
                        else:
                            print(f"{variant:>10}: {1/speedup:.2f}x slower than original")
            else:
                # Compare all pairs
                for i, (v1, t1) in enumerate(successful_times):
                    for v2, t2 in successful_times[i+1:]:
                        ratio = t1 / t2
                        if ratio > 1:
                            print(f"{v2:>10}: {ratio:.2f}x faster than {v1}")
                        else:
                            print(f"{v1:>10}: {1/ratio:.2f}x faster than {v2}")

        # Load results from successful variants
        variant_matrices = {}

        for variant in successful_variants:
            variant_dir = temp_dir / variant / "taslp23" / "M_log"

            # Find H5 files
            h5_files = list(variant_dir.glob("*.h5")) if variant_dir.exists() else []

            if not h5_files:
                print(f"⚠ Warning: No H5 files found for {variant} variant")
                continue

            # Load matrices from all folds
            variant_matrices[variant] = {'M': {}, 'sigma': {}, 'JdagJ': {}}

            for h5_file in h5_files:
                fold_matrices = load_M_matrices(str(h5_file))

                # Merge matrices from this fold
                for matrix_type in ['M', 'sigma', 'JdagJ']:
                    variant_matrices[variant][matrix_type].update(fold_matrices[matrix_type])

            total_M = len(variant_matrices[variant]['M'])
            total_sigma = len(variant_matrices[variant]['sigma'])
            total_JdagJ = len(variant_matrices[variant]['JdagJ'])

            print(f"Loaded {variant}: {total_M} M matrices, {total_sigma} sigma vectors, {total_JdagJ} JdagJ matrices")

        # Compare all pairs
        variants_with_data = list(variant_matrices.keys())

        if len(variants_with_data) < 2:
            print("\n❌ ERROR: Need at least 2 variants with data for comparison")
            return 1

        print(f"\n{'='*80}")
        print("STARTING NUMERICAL COMPARISONS")
        print('='*80)

        all_identical = True

        # Compare each pair
        for i, variant1 in enumerate(variants_with_data):
            for variant2 in variants_with_data[i+1:]:
                results = compare_matrix_sets(
                    variant_matrices[variant1],
                    variant_matrices[variant2],
                    variant1,
                    variant2
                )

                print_comparison_results(results, variant1, variant2)

                # Check if any matrices differ
                for matrix_type in ['M', 'sigma', 'JdagJ']:
                    type_results = results[matrix_type]
                    if type_results and 'max_abs_diff' in type_results:
                        if type_results['max_abs_diff'] > 0:
                            all_identical = False

        # Final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print('='*80)

        if all_identical:
            print("✓ ALL VARIANTS PRODUCED NUMERICALLY IDENTICAL MATRICES")
            print("The Mt23, Mt23f, and Mt23fp implementations are consistent.")
        else:
            print("⚠ VARIANTS PRODUCED DIFFERENT MATRICES")
            print("See detailed analysis above for specific differences.")
            print("This may indicate precision differences between implementations.")

        return 0 if all_identical else 1

    except KeyboardInterrupt:
        print(f"\n⚠ KeyboardInterrupt received in main. Cleaning up...")
        if cleanup_temp:
            cleanup_temp_directory(temp_dir, force=True)
        print("Cleanup complete. Exiting.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if cleanup_temp:
            cleanup_temp_directory(temp_dir, force=True)
        return 1
    finally:
        # Normal cleanup
        if cleanup_temp and temp_dir.exists():
            cleanup_temp_directory(temp_dir)
        elif temp_dir.exists():
            print(f"\nTemporary directory kept: {temp_dir}")


if __name__ == "__main__":
    exit(main())
