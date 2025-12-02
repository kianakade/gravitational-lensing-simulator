# Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER'S BROWSER                          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │              Frontend (index.html)                     │    │
│  │                                                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │    │
│  │  │  Slider  │  │  Slider  │  │  Slider  │  ...      │    │
│  │  │ Center X │  │ Center Y │  │ Einstein │           │    │
│  │  └──────────┘  └──────────┘  └──────────┘           │    │
│  │                                                        │    │
│  │  ┌────────────────┐  ┌────────────────┐             │    │
│  │  │ Source Image   │  │ Lensed Image   │             │    │
│  │  │  (Original)    │  │  (Computed)    │             │    │
│  │  └────────────────┘  └────────────────┘             │    │
│  └───────────────────────────────────────────────────────┘    │
│                            │                                   │
│                            │ AJAX/Fetch API                    │
│                            │ (JSON)                            │
└────────────────────────────┼───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAILWAY (Cloud Platform)                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │              Flask Backend (app.py)                    │    │
│  │                                                        │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │          API Endpoints                        │    │    │
│  │  │  • GET  /api/source-image                    │    │    │
│  │  │  • POST /api/lens                            │    │    │
│  │  │  • GET  /api/health                          │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │                    │                                  │    │
│  │                    ▼                                  │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │         AutoLens Engine                       │    │    │
│  │  │  • Load source FITS image                    │    │    │
│  │  │  • Create lens galaxy                        │    │    │
│  │  │  • Build ray-tracing grid                    │    │    │
│  │  │  • Compute lensed image                      │    │    │
│  │  │  • Convert to PNG/base64                     │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Initial Page Load
```
User → Browser → Railway → Flask → Source Image → Browser
                                    (from FITS)
```

### 2. Parameter Adjustment
```
User adjusts slider → Update value display → Ready to compute
```

### 3. Lensing Computation
```
User clicks "Compute" → Browser sends parameters (JSON) →
                       Flask receives request →
                       AutoLens computes lensing →
                       Generate PNG image →
                       Encode as base64 →
                       Return to browser →
                       Display lensed image
```

## Technology Stack

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with gradients and animations
- **JavaScript (Vanilla)**: Interactive controls, no frameworks
- **Fetch API**: Communication with backend

### Backend
- **Python 3.10**: Runtime
- **Flask**: Web framework
- **AutoLens**: Gravitational lensing physics engine
- **NumPy**: Array operations
- **Matplotlib**: Image rendering
- **Astropy**: FITS file handling
- **Gunicorn**: Production WSGI server

### Infrastructure
- **Railway**: Cloud hosting platform
- **GitHub**: Version control and source
- **HTTPS**: Automatic SSL certificates

## File Structure

```
gravitational-lensing-simulator/
│
├── app.py                     # Flask application
│   ├── initialize_source_image()   # Load FITS file
│   ├── compute_lensed_image()      # AutoLens ray tracing
│   ├── array_to_base64()           # Image encoding
│   └── API routes:
│       ├── GET  /                  # Serve HTML
│       ├── GET  /api/source-image  # Return source
│       ├── POST /api/lens          # Compute lensing
│       └── GET  /api/health        # Status check
│
├── static/
│   └── index.html             # Frontend interface
│       ├── Slider controls (5 parameters)
│       ├── Image display panels
│       ├── Compute button
│       └── JavaScript event handlers
│
├── requirements.txt           # Python dependencies
├── Procfile                   # Railway config
├── runtime.txt                # Python version
├── create_source_image.py     # FITS generator utility
└── test_setup.py              # Validation script
```

## API Contract

### Request: Compute Lensing
```http
POST /api/lens
Content-Type: application/json

{
  "center_x": 0.0,
  "center_y": 0.0,
  "einstein_radius": 1.0,
  "axis_ratio": 0.9,
  "angle": 0.0
}
```

### Response: Lensed Image
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "params": {
    "center_x": 0.0,
    "center_y": 0.0,
    "einstein_radius": 1.0,
    "axis_ratio": 0.9,
    "angle": 0.0
  }
}
```

## AutoLens Physics

### Ray Tracing Process

1. **Create Grid** (300×300 pixels, 8× oversampled)
   - Physical size determined by pixel_scales
   - Oversampling improves accuracy

2. **Define Lens Galaxy**
   - Mass profile: Isothermal
   - Parameters from sliders
   - Fixed redshift: z = 0.5

3. **Define Source Galaxy**
   - Image from FITS file
   - Fixed redshift: z = 7.0

4. **Build Tracer**
   - Combines lens and source
   - Handles cosmology (Planck15)

5. **Perform Ray Tracing**
   - Each image plane pixel traced to source plane
   - Source intensity interpolated
   - Returns lensed image

### Key Equations

**Einstein Radius:**
```
θ_E = √(4GM/c² × D_LS/D_L/D_S)
```

**Isothermal Profile:**
```
α(θ) = θ_E × (θ/|θ|)
```

**Elliptical Components:**
```
e1, e2 = convert.ell_comps_from(axis_ratio, angle)
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Load source image | ~0.1s | Once at startup |
| Create grid | ~0.1s | Per computation |
| Ray tracing | 2-5s | Depends on grid size |
| Image encoding | ~0.1s | Per computation |
| **Total per compute** | **2-5s** | User experience |

### Optimization Opportunities

1. **Grid Resolution**
   - Current: 300×300
   - Lower resolution = faster (but less detail)
   - Higher resolution = slower (more detail)

2. **Oversampling**
   - Current: 8× subgrid
   - Trade-off between accuracy and speed

3. **Caching**
   - Cache frequently used configurations
   - Pre-compute common lens positions

4. **Parallel Processing**
   - Multiple workers (currently 2)
   - Increase for more concurrent users

## Security Considerations

- **Input Validation**: All parameters validated
- **CORS**: Enabled for cross-origin requests
- **Rate Limiting**: Consider adding for production
- **File Upload**: Currently disabled (source fixed)
- **Environment Variables**: Sensitive data in env vars

## Deployment Configuration

### Railway Settings
```
Build Command:  pip install -r requirements.txt
Start Command:  gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
Python Version: 3.10.13
Port:           $PORT (auto-assigned by Railway)
```

### Environment Variables
```
SOURCE_IMAGE_PATH  = path/to/your/source.fits (optional)
PORT              = auto-assigned by Railway
```

## Monitoring & Debugging

### Health Check
```bash
curl https://your-app.railway.app/api/health
```

### Logs
```bash
railway logs
```

### Common Issues
- **Timeout**: Increase `--timeout` in Procfile
- **Memory**: AutoLens is memory-intensive
- **Build Time**: 5-10 minutes for dependencies

## Future Enhancements

### Short Term
- [ ] Add loading spinner animations
- [ ] Show computation time
- [ ] Add parameter presets
- [ ] Download lensed images

### Medium Term
- [ ] Multiple lens galaxies
- [ ] Different mass profiles (NFW, PowerLaw)
- [ ] External shear
- [ ] Light profile visualization

### Long Term
- [ ] Real-time parameter updates (WebSocket)
- [ ] GPU acceleration
- [ ] Source reconstruction
- [ ] Mass map visualization
