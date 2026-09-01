# Image & Video Preprocessing Specification

This directory implements the image transformation, video frame sampling, and plant input validation components of the **Plant Disease AI** diagnostic pipeline.

## Overview & Architecture

The preprocessing module operates as the first stage of the vision architecture, converting raw input images or video streams into standardized PyTorch tensors or normalized NumPy arrays ready for CNN feature extraction and classification.

```
Raw Image / Video Stream
         │
         ▼
┌────────────────────────────────────────┐
│   ImageValidator                       │
│   - Check format, corruption, dimensions│
│   - HSV greenness check (not_a_plant)  │
└───────────────────┬────────────────────┘
                    │ Valid
                    ▼
┌────────────────────────────────────────┐
│   ImageTransformer / VideoExtractor    │
│   - Resize to 224x224 RGB               │
│   - Normalize (ImageNet mean & std)    │
│   - Optional Augmentations (Train mode)│
└───────────────────┬────────────────────┘
                    │ Tensors / Arrays
                    ▼
           CNN Vision Backbone
```

## Specifications

### 1. Image Transformations (`src/preprocessing/image_transforms.py`)
- **Target Spatial Resolution**: `224 x 224` pixels.
- **Color Format**: 3-channel RGB.
- **Normalization (ImageNet Standards)**:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`
  - Formula: $x_{norm} = \frac{x/255.0 - \text{mean}}{\text{std}}$
- **Training Augmentations**:
  - Random Horizontal & Vertical Flips ($p = 0.5$)
  - Random Rotation ($\pm 15^\circ$)
  - Color Jitter (Brightness: 0.1, Contrast: 0.1)

### 2. Video Frame Sampling (`src/preprocessing/video_extractor.py`)
- **Extraction Rates**: Configurable sampling interval (default: 1 frame per second or top $N$ quality frames).
- **Blur Filtering**: Laplacian variance thresholding ($\sigma^2 > 100.0$) to reject blurry frames caused by camera motion.
- **Exposure Check**: Filter dark ($< 20$ average intensity) or overexposed ($> 235$ average intensity) frames.
- **Frame Aggregation**: Combines multi-frame predictions via maximum confidence scoring or ensemble voting.

### 3. Input Validation (`src/preprocessing/image_validator.py`)
- **Corruption & Integrity Check**: Ensures file can be read and decoded into valid image channels.
- **Dimension Check**: Minimum input resolution of `32 x 32` pixels.
- **Foliage Detection (`NOT_A_PLANT`)**: Evaluates Hue-Saturation-Value (HSV) foliage coverage. If green/chlorophyll color spectrum coverage is below threshold ($< 5\%$), flags status as `PredictionStatus.NOT_A_PLANT`.

## Code Module Usage

```python
from src.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(target_size=(224, 224))

# Validate and process single image
is_valid, tensor_or_array, status_info = pipeline.process_image("path/to/leaf.jpg")

# Extract and process video frames
frames, frame_meta = pipeline.process_video("path/to/leaf_video.mp4", sample_fps=1.0)
```
