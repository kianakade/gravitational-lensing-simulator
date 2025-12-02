#!/usr/bin/env python3
"""
Pre-generate cache for gravitational lensing simulator

Run this script once to generate all cached images.
New users will then have instant access to pre-computed results.

Usage:
    python pregenerate_cache.py
"""

import requests
import time
import sys

API_BASE_URL = 'http://localhost:5002'

def check_server():
    """Check if the server is running"""
    try:
        response = requests.get(f'{API_BASE_URL}/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is running")
            print(f"  Current cache: {data['cache']['count']} images, {data['cache']['size_mb']:.2f} MB")
            return True
    except requests.exceptions.RequestException:
        print("✗ Server is not running!")
        print("  Please start the Flask server first: python app.py")
        return False

def pregenerate_cache():
    """Pre-generate the cache"""
    print("\n" + "="*60)
    print("GRAVITATIONAL LENSING CACHE PRE-GENERATION")
    print("="*60)
    
    if not check_server():
        return
    
    print("\nStarting cache pre-generation...")
    print("This will compute ~180 configurations and may take 30-60 seconds.\n")
    
    try:
        start_time = time.time()
        response = requests.post(
            f'{API_BASE_URL}/api/cache/precompute',
            json={},
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            elapsed = time.time() - start_time
            
            print("\n" + "="*60)
            print("✓ CACHE PRE-GENERATION COMPLETE!")
            print("="*60)
            print(f"  Total configurations: {data['total']}")
            print(f"  Newly computed: {data['computed']}")
            print(f"  Already cached: {data['cached']}")
            print(f"  Total cache size: {data['cache_stats']['size_mb']:.2f} MB")
            print(f"  Time taken: {elapsed:.1f} seconds")
            print("\nUsers will now experience instant loading for common parameters!")
            
        else:
            print(f"\n✗ Error: Server returned status {response.status_code}")
            print(f"  {response.text}")
            
    except requests.exceptions.Timeout:
        print("\n✗ Request timed out. The server may be overloaded.")
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Network error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

def clear_cache():
    """Clear the cache"""
    print("\n" + "="*60)
    print("CLEARING CACHE")
    print("="*60)
    
    if not check_server():
        return
    
    confirm = input("\nAre you sure you want to clear the cache? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cache clearing cancelled.")
        return
    
    try:
        response = requests.post(f'{API_BASE_URL}/api/cache/clear', json={})
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ {data['message']}")
        else:
            print(f"\n✗ Error: {response.text}")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--clear':
        clear_cache()
    else:
        pregenerate_cache()
    
    print()
