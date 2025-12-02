# 🚀 Quick Start Guide - Gravitational Lensing Simulator

## What You Have

A complete web application that simulates gravitational lensing with interactive controls. All lens parameters (except redshift) are adjustable via sliders.

## Files Structure

```
gravitational-lensing-simulator/
├── app.py                      # Flask backend (AutoLens integration)
├── static/
│   └── index.html             # Frontend with interactive sliders
├── requirements.txt            # Python dependencies
├── Procfile                   # Railway configuration
├── runtime.txt                # Python version
├── create_source_image.py     # Utility to create test image
├── .gitignore                 # Git ignore file
└── README.md                  # Full documentation
```

## Step-by-Step Deployment

### 1️⃣ Test Locally (Optional but Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create a test source image (or use your NGC 1068 FITS file)
python create_source_image.py

# Run the app
python app.py

# Open browser to http://localhost:5000
```

### 2️⃣ Push to GitHub

```bash
# Initialize git (if needed)
git init
git add .
git commit -m "Initial commit: Gravitational lensing simulator"

# Create new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/lensing-simulator.git
git branch -M main
git push -u origin main
```

### 3️⃣ Deploy on Railway

1. **Sign in to Railway**: Go to [railway.app](https://railway.app)

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Wait for Build**:
   - Railway will automatically detect Python
   - Build takes ~5-10 minutes (AutoLens is a large package)
   - Watch the logs for progress

4. **Get Your URL**:
   - Once deployed, Railway provides a public URL
   - Click "Generate Domain" if not automatic
   - Your app will be at: `https://your-app-name.up.railway.app`

### 4️⃣ Using Your NGC 1068 Image

**Option A: Modify the Code**
- Replace the path in `app.py` line 32:
  ```python
  source_path = os.environ.get('SOURCE_IMAGE_PATH', 'your_ngc1068_file.fits')
  ```
- Commit and push

**Option B: Use Environment Variable**
- Upload your FITS file to the repo root
- In Railway dashboard → Variables:
  - Add: `SOURCE_IMAGE_PATH` = `ngc1068_7m+tp_co21_15as_strict_mom0.fits`
- Redeploy

**Option C: Use Railway Volumes** (for larger files)
- Railway → Add Volume
- Upload FITS file to volume
- Set `SOURCE_IMAGE_PATH` to volume path

## 🎛️ How It Works

### Backend (`app.py`)
- Loads your source image once at startup
- `/api/lens` endpoint receives slider values
- Computes lensing using AutoLens
- Returns lensed image as base64 PNG

### Frontend (`index.html`)
- Beautiful responsive interface
- 5 interactive sliders:
  - **Center X & Y**: Position of lens
  - **Einstein Radius**: Lensing strength
  - **Axis Ratio**: Ellipticity (1.0 = circular, <1.0 = elliptical)
  - **Angle**: Position angle of ellipse
- Click "Compute Lensing" to generate result
- Side-by-side source vs. lensed comparison

## 🛠️ Customization Tips

### Change Slider Ranges
Edit `static/index.html` slider attributes:
```html
<input type="range" id="einsteinRadius" 
       min="0.1" max="3.0" step="0.05" value="1.0">
```

### Adjust Grid Resolution
Edit `app.py` line 24:
```python
GRID_SHAPE = (300, 300)  # Increase for more detail (slower)
```

### Change Lens Model
Edit `app.py` line 59-64 to use different mass profiles:
```python
mass=al.mp.Isothermal(...)  # Current
mass=al.mp.NFW(...)         # NFW profile
mass=al.mp.PowerLaw(...)    # Power law
```

### Modify Redshifts
Edit `app.py` lines 55 and 69:
```python
redshift=0.5,              # Lens redshift
al.Galaxy(redshift=7.0)    # Source redshift
```

## 🐛 Troubleshooting

**"Build Failed" on Railway**
- Check Railway logs for specific error
- AutoLens build takes time, be patient
- Verify `requirements.txt` versions are compatible

**"Computing Lensing..." Hangs**
- Reduce `GRID_SHAPE` for faster computation
- Increase timeout in `Procfile`: `--timeout 180`
- Check Railway app logs

**Source Image Not Loading**
- Verify FITS file path is correct
- Check file format is valid FITS
- Try synthetic image first: `python create_source_image.py`

**Lensing Looks Wrong**
- Check your pixel scale matches your data
- Verify `pixel_scales=0.01` in `app.py` line 23
- Einstein radius might need adjustment for your image scale

## 📊 Parameter Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| **Center X/Y** | -2 to +2 | Moves lens position across field |
| **Einstein Radius** | 0.1 to 3.0 | Larger = stronger lensing (bigger rings) |
| **Axis Ratio** | 0.1 to 1.0 | Lower = more elliptical lens |
| **Angle** | 0° to 180° | Rotates elliptical lens |

**Tip**: Start with Einstein radius ~1.0, axis ratio ~0.9, then experiment!

## 🌟 Next Steps

1. **Add More Features**:
   - Multiple lens galaxies
   - Different mass profiles
   - Shear components
   - Light profile visualization

2. **Improve Performance**:
   - Cache common configurations
   - Pre-compute grid
   - Use async processing

3. **Enhanced UI**:
   - Save/load configurations
   - Animation of parameter sweeps
   - Download lensed images

## 📝 Quick Commands Reference

```bash
# Local testing
python app.py

# Create test image
python create_source_image.py

# Deploy updates
git add .
git commit -m "Update message"
git push

# Railway will auto-deploy on push!
```

## 🎉 You're All Set!

Your gravitational lensing simulator is ready to deploy. The interface is beautiful, the physics is correct, and it's all based on your original AutoLens code.

**Questions?** Check the full `README.md` for detailed documentation.

**Enjoy exploring Einstein rings! 🌌**
