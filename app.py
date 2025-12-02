"""
Flask backend for gravitational lensing simulator
Uses AutoLens for forward modeling pipeline
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from os import path
import os
import sys
import autofit as af
import autolens as al
import autolens.plot as aplt
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import Column
from astropy.cosmology import Planck15
import astropy.constants as co
import autofit as af
import numpy as np 
from pathlib import Path
import multiprocessing 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from astropy.modeling.functional_models import Gaussian2D
from astropy.io import fits
from astropy.wcs import WCS
import time
from matplotlib.colors import LinearSegmentedColormap
import io
import base64
import pickle
import hashlib
import json
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
CORS(app)

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Custom colormap - grayscale
cmap_custom = 'Greys'

# Global configuration
SRCIMG_SIZE = 125
PIXEL_SCALES = 0.01
IMGPLANE_IMSIZE = 300
CACHE_DIR = 'cache'

# Global source plane image
SOURCE_PLANE_IMAGE = None
SOURCE_EXTENT = None

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(params):
    """Generate a cache key from parameters"""
    key_str = f"{params['center_x']:.2f}_{params['center_y']:.2f}_{params['einstein_radius']:.2f}_{params['axis_ratio']:.2f}_{params['angle']}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cache_path(cache_key):
    """Get the file path for a cache key"""
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

def save_to_cache(cache_key, data):
    """Save result to disk cache"""
    try:
        cache_path = get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving to cache: {e}")
        return False

def load_from_cache(cache_key):
    """Load result from disk cache"""
    try:
        cache_path = get_cache_path(cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading from cache: {e}")
    return None

def get_cache_stats():
    """Get statistics about the cache"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
        total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
        return {
            'count': len(cache_files),
            'size_mb': total_size / (1024 * 1024)
        }
    except:
        return {'count': 0, 'size_mb': 0}

def create_source_image():
    """Create simple Gaussian source in the source plane"""
    global SOURCE_PLANE_IMAGE, SOURCE_EXTENT
    
    # Create grid in ARCSECONDS to match AutoLens coordinate system
    src_extent = SRCIMG_SIZE * PIXEL_SCALES / 2  # Half extent in arcsec
    y, x = np.mgrid[-src_extent:src_extent:SRCIMG_SIZE*1j, 
                    -src_extent:src_extent:SRCIMG_SIZE*1j]
    
    # Now the Gaussian is defined in arcseconds
    gaussian = Gaussian2D(amplitude=1.0, x_mean=0.0, y_mean=0.0, 
                        x_stddev=0.1, y_stddev=0.1)  # stddev in arcsec
    source_data = gaussian(x, y)
    
    # Load in image in the Object way that Autolens needs
    SOURCE_PLANE_IMAGE = al.Array2D.no_mask(values=source_data, pixel_scales=PIXEL_SCALES)
    
    # Calculate source plane extent in arcseconds
    src_extent_arcsec = SRCIMG_SIZE * PIXEL_SCALES
    SOURCE_EXTENT = src_extent_arcsec / 2
    
    print(f"Created source image: {SRCIMG_SIZE}x{SRCIMG_SIZE} pixels")
    print(f"Pixel scale: {PIXEL_SCALES} arcsec/pixel")
    print(f"Source extent: ±{SOURCE_EXTENT} arcsec")

def compute_lensed_image(lens_params):
    """
    Compute the lensed image given lens parameters
    
    Parameters from lens_params dict:
    - center_x: x-coordinate of lens center (arcsec, relative to source plane)
    - center_y: y-coordinate of lens center (arcsec, relative to source plane)
    - einstein_radius: Einstein radius (arcsec)
    - axis_ratio: axis ratio (0.1-1.9)
    - angle: position angle in degrees (0-179)
    """
    
    lens_z = 0.5
    lens_center_x = lens_params['center_x']
    lens_center_y = lens_params['center_y']
    lens_einstein_radius = lens_params['einstein_radius']
    lens_ax_ratio = lens_params['axis_ratio']
    lens_angle = lens_params['angle']
    
    # Create the lens galaxy
    lens = al.Galaxy(
        redshift=lens_z,
        mass=al.mp.Isothermal(
            centre=(lens_center_x, lens_center_y),
            einstein_radius=lens_einstein_radius,
            ell_comps=al.convert.ell_comps_from(axis_ratio=lens_ax_ratio, angle=lens_angle),
        )
    )
    
    # Create tracer
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=7.0)])
    
    # Create grid for image plane
    grid = al.Grid2D.uniform(
        shape_native=(IMGPLANE_IMSIZE, IMGPLANE_IMSIZE),
        pixel_scales=PIXEL_SCALES,
        over_sample_size=8,
    )
    
    # Get critical curves and caustics
    try:
        tangential_critical_curve_list = tracer.tangential_critical_curve_list_from(grid=grid)
        radial_critical_curves_list = tracer.radial_critical_curve_list_from(grid=grid)
        tangential_caustic_list = tracer.tangential_caustic_list_from(grid=grid)
        radial_caustics_list = tracer.radial_caustic_list_from(grid=grid)
    except:
        tangential_critical_curve_list = []
        radial_critical_curves_list = []
        tangential_caustic_list = []
        radial_caustics_list = []
    
    # Perform the forward modeling ray tracing
    image_plane_image = tracer.image_2d_via_input_plane_image_from(
        grid=grid, 
        plane_image=SOURCE_PLANE_IMAGE
    )

    magnification = np.nansum(image_plane_image.native.values) / np.nansum(SOURCE_PLANE_IMAGE.native)
    
    return {
        'image': image_plane_image,
        'tangential_critical_curve': tangential_critical_curve_list,
        'radial_critical_curve': radial_critical_curves_list,
        'tangential_caustic': tangential_caustic_list,
        'radial_caustic': radial_caustics_list,
        'lens_center': (lens_center_x, lens_center_y),
        'magnification': magnification
    }

def array_to_base64(array, cmap=cmap_custom, extent=None):
    """Convert numpy array to base64-encoded PNG"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if extent is not None:
        im = ax.imshow(array, origin='lower', cmap=cmap, 
                      interpolation='nearest', extent=extent)
    else:
        im = ax.imshow(array, origin='lower', cmap=cmap, interpolation='nearest')
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def create_composite_image(lensing_result):
    """Create a composite image with lensed image and critical curves"""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calculate the extent in arcseconds for image plane
    img_extent = np.array([-IMGPLANE_IMSIZE/2, IMGPLANE_IMSIZE/2, 
                           -IMGPLANE_IMSIZE/2, IMGPLANE_IMSIZE/2]) * PIXEL_SCALES
    
    # Plot lensed image
    im = ax.imshow(lensing_result['image'].native.values, origin='lower', 
                   cmap=cmap_custom, interpolation='nearest', extent=img_extent)
    
    # Track legend handles
    legend_handles = []
    
    # Add tangential critical curve if available (in red)
    if lensing_result['tangential_critical_curve']:
        try:
            lens_center_x, lens_center_y = lensing_result['lens_center']
            # Access the first critical curve and shift by lens center
            crit_x = lensing_result['tangential_critical_curve'][0][:,0] - lens_center_x
            crit_y = lensing_result['tangential_critical_curve'][0][:,1] - lens_center_y
            line, = ax.plot(crit_x, crit_y, color='red', linewidth=1.0, linestyle='--', 
                           label='Tangential Critical Curve', alpha=0.9)
            legend_handles.append(line)
        except Exception as e:
            print(f"Error plotting tangential critical curve: {e}")
    
    # Add radial critical curve if available
    if lensing_result.get('radial_critical_curve'):
        try:
            lens_center_x, lens_center_y = lensing_result['lens_center']
            crit_x = lensing_result['radial_critical_curve'][0][:,0] - lens_center_x
            crit_y = lensing_result['radial_critical_curve'][0][:,1] - lens_center_y
            line, = ax.plot(crit_x, crit_y, color='red', linewidth=1.0, linestyle='--', 
                           label='Radial Critical Curve', alpha=0.7)
            legend_handles.append(line)
        except Exception as e:
            print(f"Error plotting radial critical curve: {e}")
    
    # Add legend if we have any curves
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', 
                 framealpha=0.8, facecolor='black', edgecolor='white',
                 fontsize=9, labelcolor='white')
    
    # Add magnification text
    if 'magnification' in lensing_result:
        mag = lensing_result['magnification']
        ax.text(0.02, 0.98, f'Magnification: {mag:.2f}×', 
               transform=ax.transAxes, 
               fontsize=11, 
               fontweight='bold',
               color='white',
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', 
                        edgecolor='white',
                        alpha=0.8))
    
    ax.set_xlabel('X (arcsec)', color='white')
    ax.set_ylabel('Y (arcsec)', color='white')
    ax.tick_params(colors='white')
    
    # Make background transparent and axis labels white
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
    buf.seek(0)
    plt.close(fig)
    
    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def create_source_composite(caustics=None, lens_center=None):
    """Create source plane image with caustic"""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calculate the extent in arcseconds for source plane
    src_extent = np.array([-SRCIMG_SIZE/2, SRCIMG_SIZE/2, 
                           -SRCIMG_SIZE/2, SRCIMG_SIZE/2]) * PIXEL_SCALES
    
    # Plot source image
    im = ax.imshow(SOURCE_PLANE_IMAGE.native, origin='lower', cmap=cmap_custom, 
                   interpolation='nearest', extent=src_extent)
    
    # Track legend handles
    legend_handles = []
    
    # Add caustics if provided
    if caustics:
        # Plot tangential caustic (in red)
        if caustics.get('tangential_caustic') and len(caustics['tangential_caustic']) > 0:
            try:
                caustic_x = caustics['tangential_caustic'][0][:,0].values 
                caustic_y = caustics['tangential_caustic'][0][:,1].values
                line, = ax.plot(caustic_x, caustic_y, color='red', linewidth=1.0, 
                               linestyle='--', label='Tangential Caustic', alpha=0.9)
                legend_handles.append(line)
            except Exception as e:
                print(f"Error plotting tangential caustic: {e}")
        
        # Plot radial caustic
        if caustics.get('radial_caustic') and len(caustics['radial_caustic']) > 0:
            try:
                caustic_x = caustics['radial_caustic'][0][:,0].values
                caustic_y = caustics['radial_caustic'][0][:,1].values
                line, = ax.plot(caustic_x, caustic_y, color='red', linewidth=1.0, 
                               linestyle='--', label='Radial Caustic', alpha=0.7)
                legend_handles.append(line)
            except Exception as e:
                print(f"Error plotting radial caustic: {e}")
    
    # Add lens position marker if provided
    if lens_center:
        lens_center_x, lens_center_y = lens_center
        scatter = ax.scatter(lens_center_x, lens_center_y, color='red', s=100, 
                           marker='x', linewidths=2, label='Lens Position')
        legend_handles.append(scatter)
    
    # Add legend if we have any elements
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', 
                 framealpha=0.8, facecolor='black', edgecolor='white',
                 fontsize=9, labelcolor='white')
    
    ax.set_xlabel('X (arcsec)', color='white')
    ax.set_ylabel('Y (arcsec)', color='white')
    ax.tick_params(colors='white')
    
    # Make background transparent and axis labels white
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
    buf.seek(0)
    plt.close(fig)
    
    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# ============================================================================
# RGB Image Lensing Functions
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def smooth_and_resize_image(image_path, target_size=(150, 150)):
    """Load and resize image to target size with smoothing"""
    img = Image.open(image_path).convert('RGB')
    img_smoothed = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img_smoothed)
    return img_array

def forward_model_rgb_channel(source_channel_array, pixel_scales=0.01):
    """Forward model a single RGB channel through the lens"""
    # Fixed lens parameters for fun/demo mode
    lens_z = 0.5
    lens_center_x = 0.0
    lens_center_y = 0.0
    lens_einstein_radius = 1.5
    lens_ax_ratio = 0.5
    lens_angle = 0.0
    
    # Create lens
    lens = al.Galaxy(
        redshift=lens_z,
        mass=al.mp.Isothermal(
            centre=(lens_center_x, lens_center_y),
            einstein_radius=lens_einstein_radius,
            ell_comps=al.convert.ell_comps_from(axis_ratio=lens_ax_ratio, angle=lens_angle),
        )
    )
    
    # Create tracer
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=7.0)])
    
    # Create grid for image plane (larger than source for lensing effects)
    grid = al.Grid2D.uniform(
        shape_native=(350, 350),
        pixel_scales=pixel_scales,
        over_sample_size=8,
    )
    
    # Perform ray tracing
    modeled_channel = tracer.image_2d_via_input_plane_image_from(
        grid=grid, 
        plane_image=source_channel_array
    )
    
    return modeled_channel.native.values

def process_rgb_image(image_path):
    """Process an RGB image through gravitational lensing"""
    # Load and resize image
    img_array = smooth_and_resize_image(image_path, target_size=(150, 150))
    
    # Split into RGB channels
    red_channel = img_array[:, :, 0].astype(float)
    green_channel = img_array[:, :, 1].astype(float)
    blue_channel = img_array[:, :, 2].astype(float)
    
    # Convert to AutoLens Array2D objects
    pixel_scales = 0.01
    red_array = al.Array2D.no_mask(values=red_channel, pixel_scales=pixel_scales)
    green_array = al.Array2D.no_mask(values=green_channel, pixel_scales=pixel_scales)
    blue_array = al.Array2D.no_mask(values=blue_channel, pixel_scales=pixel_scales)
    
    # Forward model each channel
    red_modeled = forward_model_rgb_channel(red_array, pixel_scales)
    green_modeled = forward_model_rgb_channel(green_array, pixel_scales)
    blue_modeled = forward_model_rgb_channel(blue_array, pixel_scales)
    
    # Recombine into RGB
    rgb_modeled = np.stack([red_modeled, green_modeled, blue_modeled], axis=-1)
    
    # Normalize to 0-255 range
    if rgb_modeled.max() > 0:
        rgb_modeled = (rgb_modeled / rgb_modeled.max() * 255)
    rgb_modeled = np.clip(rgb_modeled, 0, 255).astype(np.uint8)
    
    return img_array, rgb_modeled

def create_comparison_image(original, lensed):
    """Create side-by-side comparison of original and lensed images"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold', color='white')
    axes[0].axis('off')
    
    # Lensed image
    axes[1].imshow(lensed)
    axes[1].set_title('Gravitationally Lensed', fontsize=14, fontweight='bold', color='white')
    axes[1].axis('off')
    
    # Styling
    fig.patch.set_facecolor('#0d0f15')
    for ax in axes:
        ax.set_facecolor('#0d0f15')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='#0d0f15')
    buf.seek(0)
    plt.close(fig)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (images, etc.)"""
    return send_from_directory('static', filename)

@app.route('/api/source-image', methods=['GET'])
def get_source_image():
    """Return the source plane image"""
    if SOURCE_PLANE_IMAGE is None:
        return jsonify({'error': 'Source image not loaded'}), 500
    
    img_base64 = create_source_composite()
    
    return jsonify({
        'image': img_base64,
        'shape': list(SOURCE_PLANE_IMAGE.native.shape),
        'dimensions': {
            'width': float(SRCIMG_SIZE * PIXEL_SCALES),
            'height': float(SRCIMG_SIZE * PIXEL_SCALES),
            'pixel_scales': float(PIXEL_SCALES),
            'extent': float(SOURCE_EXTENT)
        }
    })

@app.route('/api/lens', methods=['POST'])
def compute_lens():
    """
    Compute lensed image with given parameters
    Expected JSON body:
    {
        "center_x": float (arcsec),
        "center_y": float (arcsec),
        "einstein_radius": float (arcsec),
        "axis_ratio": float (0.1-1.9),
        "angle": float (degrees 0-179)
    }
    """
    try:
        params = request.json
        
        # Validate parameters
        required_params = ['center_x', 'center_y', 'einstein_radius', 'axis_ratio', 'angle']
        for param in required_params:
            if param not in params:
                return jsonify({'error': f'Missing parameter: {param}'}), 400
        
        # Check cache first
        cache_key = get_cache_key(params)
        cached_result = load_from_cache(cache_key)
        
        if cached_result:
            print(f"Cache hit for params: {params}")
            return jsonify({
                'image': cached_result['image'],
                'source_with_caustic': cached_result['source_with_caustic'],
                'params': params,
                'success': True,
                'from_cache': True
            })
        
        print(f"Computing lensing with params: {params}")
        
        # Compute lensed image
        lensing_result = compute_lensed_image(params)
        
        # Convert lensed image to base64 with composite view
        img_base64 = create_composite_image(lensing_result)
        
        # Create source plane with caustics
        caustics_data = {
            'tangential_caustic': lensing_result['tangential_caustic'],
            'radial_caustic': lensing_result['radial_caustic']
        }
        source_img_base64 = create_source_composite(
            caustics=caustics_data,
            lens_center=lensing_result['lens_center']
        )
        
        result = {
            'image': img_base64,
            'source_with_caustic': source_img_base64,
            'params': params,
            'success': True,
            'from_cache': False
        }
        
        # Save to cache
        save_to_cache(cache_key, {
            'image': img_base64,
            'source_with_caustic': source_img_base64
        })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in compute_lens: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    cache_stats = get_cache_stats()
    return jsonify({
        'status': 'healthy',
        'source_loaded': SOURCE_PLANE_IMAGE is not None,
        'config': {
            'srcimg_size': SRCIMG_SIZE,
            'pixel_scales': PIXEL_SCALES,
            'imgplane_imsize': IMGPLANE_IMSIZE
        },
        'cache': cache_stats
    })

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all cached images"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
        for f in cache_files:
            os.remove(os.path.join(CACHE_DIR, f))
        return jsonify({
            'success': True,
            'message': f'Cleared {len(cache_files)} cached images'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/precompute', methods=['POST'])
def precompute_cache():
    """Pre-compute a grid of parameter values for the cache"""
    try:
        # Define sampling grid
        einstein_radius_values = [0.6, 0.8, 1.0, 1.2, 1.4]
        axis_ratio_values = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        angle_values = [0, 30, 60, 90, 120, 150]
        
        total = len(einstein_radius_values) * len(axis_ratio_values) * len(angle_values)
        computed = 0
        cached = 0
        
        results = []
        
        for er in einstein_radius_values:
            for ar in axis_ratio_values:
                for angle in angle_values:
                    params = {
                        'center_x': 0.0,
                        'center_y': 0.0,
                        'einstein_radius': er,
                        'axis_ratio': ar,
                        'angle': angle
                    }
                    
                    cache_key = get_cache_key(params)
                    
                    # Check if already cached
                    if load_from_cache(cache_key):
                        cached += 1
                        continue
                    
                    # Compute and cache
                    try:
                        lensing_result = compute_lensed_image(params)
                        img_base64 = create_composite_image(lensing_result)
                        caustics_data = {
                            'tangential_caustic': lensing_result['tangential_caustic'],
                            'radial_caustic': lensing_result['radial_caustic']
                        }
                        source_img_base64 = create_source_composite(
                            caustics=caustics_data,
                            lens_center=lensing_result['lens_center']
                        )
                        
                        save_to_cache(cache_key, {
                            'image': img_base64,
                            'source_with_caustic': source_img_base64
                        })
                        computed += 1
                    except Exception as e:
                        print(f"Error computing params {params}: {e}")
        
        cache_stats = get_cache_stats()
        
        return jsonify({
            'success': True,
            'total': total,
            'computed': computed,
            'cached': cached,
            'cache_stats': cache_stats
        })
        
    except Exception as e:
        print(f"Error in precompute_cache: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rgb-lens', methods=['POST'])
def rgb_lens():
    """Process an uploaded image through RGB gravitational lensing"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"Processing RGB image: {unique_filename}")
        
        # Process the image
        original, lensed = process_rgb_image(filepath)
        
        # Create comparison visualization
        comparison_base64 = create_comparison_image(original, lensed)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'image': comparison_base64,
            'message': 'Image successfully lensed!'
        })
        
    except Exception as e:
        print(f"Error in rgb_lens: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize source image
    print("Initializing gravitational lensing simulator...")
    create_source_image()
    print("Source image created successfully!")
    
    # Run the app
    port = int(os.environ.get('PORT', 5002))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
