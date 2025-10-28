# 3D Material Property Rendering Pipeline

A comprehensive Blender-based pipeline for rendering 3D objects with multiple lighting conditions and material property analysis. This pipeline generates high-quality renders, material maps, and transition videos for 3D objects.

## 🎬 Features

### **8 Lighting Conditions**
- **City** - Urban cityscape lighting
- **Night** - Night city lighting  
- **Studio** - Professional studio lighting
- **Sunset** - Golden hour sunset lighting
- **Sunrise** - Early morning sunrise lighting
- **Forest** - Natural forest environment
- **Courtyard** - Outdoor courtyard lighting
- **Interior** - Indoor lighting

### **5 Material Property Maps**
- **Normal Maps** - Surface normal information (OpenEXR format)
- **Albedo Maps** - Base color information
- **Specular Maps** - Specular reflection properties
- **Metallic Maps** - Metallic/rough material properties
- **Roughness Maps** - Surface roughness information

### **Comprehensive Video Outputs**
- **Individual Object Videos** - Each object with all 8 lighting conditions
- **Material Map Transition Videos** - Smooth transitions between material properties
- **Lighting Transition Videos** - Smooth transitions between lighting conditions
- **Grid Compositions** - Multiple objects in 3x3, 4x4, and 5x4 grids
- **Comprehensive Transition Videos** - Combined lighting and material transitions

## 🚀 Quick Start

### Prerequisites
- **Blender 3.6+** (Tested with 3.6.23)
- **Python 3.x** (for running scripts)
- **FFmpeg** (for video composition)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/3d-material-pipeline.git
   cd 3d-material-pipeline
   ```

2. **Set up Blender paths:**
   Edit `combined_pipeline.py` and update the Blender paths:
   ```python
   BLENDER_ROOT = r"path/to/your/blender"
   BLENDER_PATH = r"path/to/your/blender/blender.exe"
   ```

3. **Configure input/output directories:**
   ```python
   INPUT_DIR = r"path/to/your/3d/objects"
   OUTPUT_ROOT = r"path/to/output/directory"
   RESULTS_DIR = r"path/to/results/directory"
   ```

### Usage

**Run the complete pipeline:**
```bash
blender --background --python combined_pipeline.py
```

**Or use the batch file (Windows):**
```bash
run_pipeline.bat
```

## 📁 Project Structure

```
3d-material-pipeline/
├── combined_pipeline.py          # Main pipeline script
├── test_pipeline_8lightings.py   # Configuration test
├── test_pipeline_verification.py # Verification test
├── simple_verification.py        # Simple verification
├── run_pipeline.bat             # Windows batch runner
├── README.md                    # This file
└── output/                      # Generated output
    ├── object_name/
    │   ├── city/                # City lighting renders
    │   ├── night/               # Night lighting renders
    │   ├── studio/              # Studio lighting renders
    │   └── ...                  # Other lighting conditions
    ├── compositions/            # Video compositions
    │   ├── lighting_transition_3x3_9objects.mp4
    │   ├── lighting_grid_4x4_16objects.mp4
    │   └── {object}_material_maps_transition.mp4
    └── results/                 # Analysis results
        ├── summary.json
        └── detailed_results.json
```

## ⚙️ Configuration

### Render Settings
```python
RENDER_SETTINGS = {
    'frames': 60,              # Frames per animation
    'resolution_x': 1280,      # Output width
    'resolution_y': 720        # Output height
}
```

### Lighting Conditions
```python
ENVIRONMENT_MAPS = [
    'city', 'night', 'studio', 'sunset', 
    'sunrise', 'forest', 'courtyard', 'interior'
]
```

### Material Maps
```python
OUTPUT_SETTINGS = {
    'render_normal_maps': True,
    'render_albedo_maps': True,
    'render_specular_maps': True,
    'render_metallic_maps': True,
    'render_roughness_maps': True,
    'normal_map_format': 'OPEN_EXR',    # Preserves negative values
    'other_maps_format': 'OPEN_EXR'     # High precision
}
```

## 📊 Performance

### Rendering Statistics
- **19 objects** × **8 lighting conditions** × **5 material maps** × **60 frames**
- **Total frames**: 45,600
- **Estimated time**: ~63.3 hours (Blender Eevee)
- **Output size**: ~50GB+ (depending on resolution)

### Optimization Tips
1. **Reduce frames**: Set `frames: 30` for faster rendering
2. **Lower resolution**: Use `640x360` for preview renders
3. **Use Unreal Engine**: 150x faster than Blender (see `unreal_pipeline.py`)
4. **Parallel processing**: Run multiple objects simultaneously

## 🎥 Output Videos

### Individual Object Videos
- `{object_name}_city.mp4` - City lighting
- `{object_name}_night.mp4` - Night lighting
- `{object_name}_studio.mp4` - Studio lighting
- ... (8 lighting conditions total)

### Transition Videos
- `{object_name}_lighting_transition_8envs.mp4` - All 8 lighting conditions
- `{object_name}_material_maps_transition.mp4` - All 4 material maps
- `lighting_grid_3x3_9objects.mp4` - 3x3 object grid
- `lighting_grid_4x4_16objects.mp4` - 4x4 object grid

## 🔧 Advanced Features

### Custom HDR Support
Add your own HDR files to the custom directory:
```python
CUSTOM_HDR_DIR = r"path/to/custom/hdr/files"
```

### Fallback Lighting
The pipeline automatically uses fallback HDR files if the primary ones are missing.

### Material Analysis
Comprehensive material property analysis with CSV export for further processing.

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'bpy'"**
   - Run through Blender: `blender --background --python combined_pipeline.py`
   - Don't run directly with Python

2. **Missing HDR files**
   - Check Blender installation path
   - Verify HDR files exist in `studiolights/world/`
   - Pipeline will use fallback lighting

3. **Out of memory**
   - Reduce resolution in `RENDER_SETTINGS`
   - Process fewer objects at once
   - Close other applications

4. **Slow rendering**
   - Use Unreal Engine pipeline for 150x speed improvement
   - Reduce frame count
   - Lower resolution

### Performance Monitoring
```bash
# Check rendering progress
ls output_trial1/*/city/ | wc -l

# Monitor memory usage
tasklist | findstr blender
```

## 📈 Results Analysis

The pipeline generates comprehensive analysis files:

- **`summary.json`** - Overall statistics
- **`detailed_results.json`** - Per-object analysis
- **`comprehensive_material_verification.csv`** - Material property analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Blender Foundation** for the amazing 3D software
- **HDR Haven** for environment maps
- **OpenEXR** for high-precision image format

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration settings

---

**Happy Rendering! 🎬✨**
