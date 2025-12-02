#!/usr/bin/env python3
"""
Cache Inspector for Gravitational Lensing Simulator

View statistics and contents of the cache directory.

Usage:
    python inspect_cache.py
"""

import os
import pickle
from pathlib import Path

CACHE_DIR = 'cache'

def format_bytes(bytes_size):
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"

def inspect_cache():
    """Inspect cache contents"""
    print("\n" + "="*70)
    print("GRAVITATIONAL LENSING CACHE INSPECTOR")
    print("="*70)
    
    if not os.path.exists(CACHE_DIR):
        print(f"\n✗ Cache directory '{CACHE_DIR}' does not exist")
        print("  Run the simulator first to create the cache.")
        return
    
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    if not cache_files:
        print(f"\n✗ Cache is empty (no .pkl files found)")
        return
    
    print(f"\n📁 Cache Directory: {os.path.abspath(CACHE_DIR)}")
    print(f"📊 Total Files: {len(cache_files)}")
    
    total_size = 0
    file_sizes = []
    
    for filename in cache_files:
        filepath = os.path.join(CACHE_DIR, filename)
        size = os.path.getsize(filepath)
        total_size += size
        file_sizes.append(size)
    
    avg_size = total_size / len(cache_files) if cache_files else 0
    
    print(f"💾 Total Size: {format_bytes(total_size)}")
    print(f"📈 Average File Size: {format_bytes(avg_size)}")
    print(f"📉 Smallest File: {format_bytes(min(file_sizes))}")
    print(f"📈 Largest File: {format_bytes(max(file_sizes))}")
    
    print("\n" + "-"*70)
    print("SAMPLE CACHE ENTRIES (first 10)")
    print("-"*70)
    
    for i, filename in enumerate(cache_files[:10]):
        filepath = os.path.join(CACHE_DIR, filename)
        size = os.path.getsize(filepath)
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            has_image = 'image' in data
            has_source = 'source_with_caustic' in data
            
            print(f"\n{i+1}. {filename}")
            print(f"   Size: {format_bytes(size)}")
            print(f"   Contains: ", end="")
            if has_image:
                print("✓ lensed image ", end="")
            if has_source:
                print("✓ source w/ caustic", end="")
            print()
            
        except Exception as e:
            print(f"\n{i+1}. {filename}")
            print(f"   Size: {format_bytes(size)}")
            print(f"   ✗ Error reading: {e}")
    
    if len(cache_files) > 10:
        print(f"\n... and {len(cache_files) - 10} more files")
    
    print("\n" + "="*70)
    print(f"✓ Cache inspection complete")
    print("="*70 + "\n")

def estimate_cache_capacity():
    """Estimate how many more entries can fit"""
    print("\n" + "="*70)
    print("CACHE CAPACITY ESTIMATION")
    print("="*70)
    
    if not os.path.exists(CACHE_DIR):
        print("\n✗ Cache directory does not exist")
        return
    
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    if not cache_files:
        print("\n✗ Cache is empty - no data to estimate from")
        return
    
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
    avg_size = total_size / len(cache_files)
    
    # Estimate for different size limits
    size_limits = [100, 500, 1000, 5000]  # MB
    
    print(f"\nCurrent: {len(cache_files)} entries, {format_bytes(total_size)}")
    print(f"Average entry size: {format_bytes(avg_size)}\n")
    
    print("Estimated capacity for different size limits:")
    print("-" * 50)
    
    for limit_mb in size_limits:
        limit_bytes = limit_mb * 1024 * 1024
        estimated_entries = int(limit_bytes / avg_size)
        print(f"  {limit_mb:4d} MB → ~{estimated_entries:5d} entries")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    inspect_cache()
    estimate_cache_capacity()
