#!/usr/bin/env python3
"""
Parallel Cache Generator for Server Cluster
Generates gravitational lensing cache using multiprocessing

Based on slider ranges:
- center_x: -0.6 to 0.6, step 0.01 (121 values)
- center_y: -0.6 to 0.6, step 0.01 (121 values)
- einstein_radius: 0.5 to 1.5, step 0.01 (101 values)
- axis_ratio: 0.3 to 1.9, step 0.01 (161 values)
- angle: 0 to 179, step 1 (180 values)

Total: 121 * 121 * 101 * 161 * 180 = ~42.6 BILLION combinations

This is WAY too many! See options below.
"""

import numpy as np
import os
import sys
import pickle
import hashlib
import time
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import argparse

# AutoLens imports
import autolens as al
from astropy.modeling.functional_models import Gaussian2D

# Configuration matching your app.py
SRCIMG_SIZE = 100
PIXEL_SCALES = 0.015
IMGPLANE_IMSIZE = 200
CACHE_DIR = 'cache'

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

# Global source image (will be created once per worker)
SOURCE_PLANE_IMAGE = None

def create_source_image():
    """Create the source image"""
    global SOURCE_PLANE_IMAGE
    
    src_extent = SRCIMG_SIZE * PIXEL_SCALES / 2
    y, x = np.mgrid[-src_extent:src_extent:SRCIMG_SIZE*1j, 
                    -src_extent:src_extent:SRCIMG_SIZE*1j]
    
    gaussian = Gaussian2D(amplitude=1.0, x_mean=0.0, y_mean=0.0, 
                        x_stddev=0.1, y_stddev=0.1)
    source_data = gaussian(x, y)
    
    SOURCE_PLANE_IMAGE = al.Array2D.no_mask(values=source_data, pixel_scales=PIXEL_SCALES)

def get_cache_key(params):
    """Generate cache key from parameters"""
    key_str = f"{params['center_x']:.2f}_{params['center_y']:.2f}_{params['einstein_radius']:.2f}_{params['axis_ratio']:.2f}_{params['angle']}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cache_path(cache_key):
    """Get cache file path"""
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

def cache_exists(cache_key):
    """Check if cache file exists"""
    return os.path.exists(get_cache_path(cache_key))

def save_to_cache(cache_key, data):
    """Save computation to cache"""
    cache_path = get_cache_path(cache_key)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def compute_lensed_image(lens_params):
    """Compute lensing for given parameters"""
    # Create lens galaxy
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(lens_params['center_x'], lens_params['center_y']),
            einstein_radius=lens_params['einstein_radius'],
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=lens_params['axis_ratio'], 
                angle=lens_params['angle']
            ),
        )
    )
    
    # Create tracer
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=7.0)])
    
    # Create grid
    grid = al.Grid2D.uniform(
        shape_native=(IMGPLANE_IMSIZE, IMGPLANE_IMSIZE),
        pixel_scales=PIXEL_SCALES,
        over_sample_size=4,
    )
    
    # Get critical curves and caustics
    try:
        critical_curves = tracer.critical_curves_from(grid=grid)
        caustics = tracer.caustics_from(grid=grid)
    except:
        critical_curves = None
        caustics = None
    
    # Perform ray tracing
    lensed_image = tracer.image_2d_via_input_plane_image_from(
        grid=grid, 
        plane_image=SOURCE_PLANE_IMAGE
    )
    
    # Calculate magnification
    source_flux = np.sum(SOURCE_PLANE_IMAGE.native)
    lensed_flux = np.sum(lensed_image.native)
    magnification = lensed_flux / source_flux if source_flux > 0 else 0
    
    return {
        'lensed_image': lensed_image.native,
        'critical_curves': critical_curves,
        'caustics': caustics,
        'magnification': magnification
    }

def process_parameter_set(params, progress_dict=None, worker_id=None):
    """Process a single parameter set (worker function)"""
    global SOURCE_PLANE_IMAGE
    
    # Initialize source image if not already done (once per worker)
    if SOURCE_PLANE_IMAGE is None:
        create_source_image()
    
    cache_key = get_cache_key(params)
    
    # Skip if already cached
    if cache_exists(cache_key):
        if progress_dict is not None:
            progress_dict['skipped'] += 1
        return {'status': 'skipped', 'cache_key': cache_key}
    
    try:
        # Compute lensing
        result = compute_lensed_image(params)
        
        # Save to cache
        save_to_cache(cache_key, result)
        
        if progress_dict is not None:
            progress_dict['created'] += 1
        
        return {'status': 'created', 'cache_key': cache_key}
    
    except Exception as e:
        if progress_dict is not None:
            progress_dict['errors'] += 1
        return {'status': 'error', 'cache_key': cache_key, 'error': str(e)}

def generate_parameter_combinations():
    """
    Generate exact parameter combinations matching HTML sliders
    
    10 discrete steps per slider = 100,000 total combinations
    """
    
    # Exact 10 steps matching HTML sliders
    center_x_range = np.linspace(-0.6, 0.6, 10)          # step ~0.133
    center_y_range = np.linspace(-0.6, 0.6, 10)          # step ~0.133
    einstein_radius_range = np.linspace(0.5, 1.5, 10)    # step ~0.111
    axis_ratio_range = np.linspace(0.3, 1.9, 10)         # step ~0.178
    angle_range = np.linspace(0, 180, 10)                # step 20
    
    print(f"\n📊 Parameter ranges (10 discrete steps each):")
    print(f"   center_x: {len(center_x_range)} values from {center_x_range[0]:.3f} to {center_x_range[-1]:.3f}")
    print(f"   center_y: {len(center_y_range)} values from {center_y_range[0]:.3f} to {center_y_range[-1]:.3f}")
    print(f"   einstein_radius: {len(einstein_radius_range)} values from {einstein_radius_range[0]:.3f} to {einstein_radius_range[-1]:.3f}")
    print(f"   axis_ratio: {len(axis_ratio_range)} values from {axis_ratio_range[0]:.3f} to {axis_ratio_range[-1]:.3f}")
    print(f"   angle: {len(angle_range)} values from {angle_range[0]:.0f} to {angle_range[-1]:.0f}")
    
    total = (len(center_x_range) * len(center_y_range) * len(einstein_radius_range) * 
             len(axis_ratio_range) * len(angle_range))
    print(f"\n📊 Total combinations: {total:,}")
    
    # Generate all combinations
    combinations = []
    for cx in center_x_range:
        for cy in center_y_range:
            for er in einstein_radius_range:
                for ar in axis_ratio_range:
                    for angle in angle_range:
                        combinations.append({
                            'center_x': float(cx),
                            'center_y': float(cy),
                            'einstein_radius': float(er),
                            'axis_ratio': float(ar),
                            'angle': float(angle)
                        })
    
    return combinations

def print_progress(created, skipped, errors, total, start_time):
    """Print progress statistics"""
    completed = created + skipped + errors
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = total - completed
    eta = remaining / rate if rate > 0 else 0
    
    print(f"\r📊 Progress: {completed:,}/{total:,} ({100*completed/total:.1f}%) | "
          f"Created: {created:,} | Skipped: {skipped:,} | Errors: {errors:,} | "
          f"Rate: {rate:.1f}/s | ETA: {eta/3600:.1f}h", end='', flush=True)

def main():
    parser = argparse.ArgumentParser(description='Generate gravitational lensing cache in parallel')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Chunk size for multiprocessing (default: 100)')
    
    args = parser.parse_args()
    
    # Determine number of workers
    n_workers = args.workers if args.workers else cpu_count()
    
    print("=" * 80)
    print("PARALLEL CACHE GENERATOR FOR GRAVITATIONAL LENSING")
    print("10 DISCRETE STEPS PER SLIDER = 100,000 COMBINATIONS")
    print("=" * 80)
    print(f"\n🖥️  Workers: {n_workers} (CPU count: {cpu_count()})")
    print(f"📁 Cache directory: {os.path.abspath(CACHE_DIR)}")
    
    # Generate parameter combinations
    print(f"\n⚙️  Generating parameter combinations...")
    combinations = generate_parameter_combinations()
    total = len(combinations)
    
    # Estimate time and size
    time_per_computation = 2  # seconds (conservative)
    estimated_time_serial = total * time_per_computation / 3600  # hours
    estimated_time_parallel = estimated_time_serial / n_workers
    estimated_size_mb = total * 50 / 1024  # MB (50 KB per cache file)
    
    print(f"\n⏱️  Estimated time:")
    print(f"   Serial: {estimated_time_serial:.1f} hours")
    print(f"   Parallel ({n_workers} workers): {estimated_time_parallel:.1f} hours")
    print(f"💾 Estimated cache size: {estimated_size_mb:.1f} MB ({estimated_size_mb/1024:.1f} GB)")
    
    # Confirm
    response = input(f"\n⚠️  Generate {total:,} cache files? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    # Create shared progress dictionary
    with Manager() as manager:
        progress_dict = manager.dict()
        progress_dict['created'] = 0
        progress_dict['skipped'] = 0
        progress_dict['errors'] = 0
        
        # Create worker pool
        print(f"\n🚀 Starting {n_workers} workers...")
        start_time = time.time()
        
        # Partial function with progress dict
        worker_func = partial(process_parameter_set, progress_dict=progress_dict)
        
        # Process in parallel
        with Pool(processes=n_workers) as pool:
            # Use imap for progress tracking
            for i, result in enumerate(pool.imap(worker_func, combinations, 
                                                   chunksize=args.chunk_size)):
                if (i + 1) % 100 == 0:  # Update every 100 items
                    print_progress(progress_dict['created'], 
                                 progress_dict['skipped'],
                                 progress_dict['errors'],
                                 total, start_time)
        
        # Final progress
        print_progress(progress_dict['created'], 
                     progress_dict['skipped'],
                     progress_dict['errors'],
                     total, start_time)
        print()  # New line
        
        # Final statistics
        elapsed = time.time() - start_time
        print(f"\n✅ Cache generation complete!")
        print(f"   Created: {progress_dict['created']:,} new entries")
        print(f"   Skipped: {progress_dict['skipped']:,} existing entries")
        print(f"   Errors: {progress_dict['errors']:,}")
        print(f"   Total: {progress_dict['created'] + progress_dict['skipped']:,}")
        print(f"   Time: {elapsed/3600:.2f} hours")
        print(f"   Rate: {total/elapsed:.1f} combinations/second")
    
    # Cache statistics
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
    
    print(f"\n💾 Final cache statistics:")
    print(f"   Files: {len(cache_files):,}")
    print(f"   Size: {total_size / (1024**2):.2f} MB ({total_size / (1024**3):.3f} GB)")
    print(f"   Average: {total_size / len(cache_files) / 1024:.2f} KB per file")
    
    print(f"\n📦 Next steps:")
    print(f"   1. Verify cache: ls -lh {CACHE_DIR}")
    print(f"   2. Transfer to local: rsync -avz server:{os.path.abspath(CACHE_DIR)}/ ./cache/")
    print(f"   3. Commit: git add cache/ && git commit -m 'Add server-generated cache'")
    print(f"   4. Push: git push")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Partial cache saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
