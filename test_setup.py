"""
Test script to verify the gravitational lensing simulator setup
Run this to test that AutoLens and all dependencies work correctly
"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import autolens as al
        print("✓ AutoLens imported successfully")
        print(f"  AutoLens version: {al.__version__}")
    except ImportError as e:
        print(f"✗ AutoLens import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        from astropy.io import fits
        print("✓ Astropy imported successfully")
    except ImportError as e:
        print(f"✗ Astropy import failed: {e}")
        return False
    
    return True

def test_autolens_basic():
    """Test basic AutoLens functionality"""
    print("\nTesting AutoLens basic functionality...")
    
    try:
        import autolens as al
        import numpy as np
        
        # Create a simple grid
        grid = al.Grid2D.uniform(
            shape_native=(50, 50),
            pixel_scales=0.1
        )
        print("✓ Created AutoLens grid")
        
        # Create a simple lens
        lens = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.0
            )
        )
        print("✓ Created lens galaxy")
        
        # Create a tracer
        tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])
        print("✓ Created tracer")
        
        # Create a simple source image
        source_data = np.zeros((50, 50))
        source_data[20:30, 20:30] = 1.0
        source_image = al.Array2D.no_mask(values=source_data, pixel_scales=0.1)
        print("✓ Created source image")
        
        # Perform lensing
        lensed = tracer.image_2d_via_input_plane_image_from(
            grid=grid,
            plane_image=source_image
        )
        print("✓ Performed ray tracing")
        print(f"  Lensed image shape: {lensed.shape}")
        print(f"  Lensed image min/max: {lensed.native.min():.3f} / {lensed.native.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ AutoLens test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_app():
    """Test that the Flask app can be imported"""
    print("\nTesting Flask app...")
    
    try:
        # Try to import the app
        import app as flask_app
        print("✓ Flask app imported successfully")
        
        # Check if app object exists
        if hasattr(flask_app, 'app'):
            print("✓ Flask app object found")
        else:
            print("✗ Flask app object not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Flask app import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_source_image_creation():
    """Test creating a synthetic source image"""
    print("\nTesting source image creation...")
    
    try:
        import create_source_image
        import os
        
        # Create test image
        test_filename = 'test_source.fits'
        create_source_image.create_synthetic_source(
            filename=test_filename,
            size=100
        )
        print("✓ Created synthetic source image")
        
        # Verify file exists
        if os.path.exists(test_filename):
            print("✓ FITS file created successfully")
            
            # Try to load it with AutoLens
            import autolens as al
            test_image = al.Array2D.from_fits(
                file_path=test_filename,
                pixel_scales=0.01
            )
            print(f"✓ Loaded with AutoLens, shape: {test_image.shape}")
            
            # Clean up
            os.remove(test_filename)
            print("✓ Cleaned up test file")
            
            return True
        else:
            print("✗ FITS file not created")
            return False
            
    except Exception as e:
        print(f"✗ Source image creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Gravitational Lensing Simulator - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("AutoLens Functionality", test_autolens_basic),
        ("Source Image Creation", test_source_image_creation),
        ("Flask App", test_flask_app),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print('=' * 60)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "PASSED ✓" if result else "FAILED ✗"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready for deployment.")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the local server")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Push to GitHub and deploy to Railway")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Run: pip install -r requirements.txt")
        print("- Check Python version (need 3.10+)")
        print("- Verify all dependencies installed correctly")
        return 1

if __name__ == '__main__':
    sys.exit(main())
