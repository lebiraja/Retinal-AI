# Validation Service

The **Validation Service** (`app/services/validation_service.py`) is a newly introduced, lightweight heuristic filter residing in the Backend API Gateway. 

## Purpose

The primary purpose of this service is to prevent invalid, non-retinal images (e.g., random photos, screenshots, or corrupted files) from being processed by the EfficientNet-B4 model. 

Deep learning models will often output highly confident (but ultimately meaningless) predictions even on pure noise or completely unrelated images. This pre-flight check saves GPU compute resources and ensures the system fails fast with a helpful, descriptive error to the user rather than returning a nonsensical "disease risk" for a picture of a cat.

## Heuristic Checks

The Validation Service evaluates an image based on three primary characteristics unique to retinal fundus photography. The checks are executed in `< 2 ms` per image. If any single check fails, the image is immediately rejected.

1. **Minimum Resolution**
   - **Rule**: Image must be at least `64x64` pixels.
   - **Reason**: Images smaller than this cannot realistically represent a recognizable fundus scan.

2. **Aspect Ratio**
   - **Rule**: The ratio of the longest side to the shortest side must be `≤ 1.6`.
   - **Reason**: Ophthalmoscope and fundus cameras output circular or near-square frames. Highly panoramic or extreme widescreen images are rejected.

3. **Corner Darkness (Vignette)**
   - **Rule**: The average brightness across the four corners of the image must be `≤ 100` (on a `0-255` scale).
   - **Reason**: Fundus cameras naturally produce a dark, circular vignette caused by the camera lens aperture. The corners of a true fundus image are almost entirely black.

4. **Red-Channel Dominance (Warm Tone)**
   - **Rule**: The mean intensity of the Red channel must be greater than the mean intensity of the Blue channel (`R > B`).
   - **Reason**: Retinal tissue and the interior of the eye are heavily dominated by red and orange hues due to blood vessels. Images with dominant cool colors (blues) are statistically highly unlikely to be retinal scans.

## Integration

The validation process occurs transparently during the `/predict` and `/predict-batch` endpoints in the Backend API.

```python
# Pseudo-code representation of the flow in the Backend API Gateway
@app.post("/predict")
async def predict_image(image: UploadFile):
    file_bytes = await image.read()
    
    # 1. Light-weight heuristic check
    validate_fundus_image(file_bytes, image.filename) # Raises 422 if failed
    
    # 2. Forward to GPU Model Service
    response = await forward_to_model_service(file_bytes)
    
    return response
```

If the validation fails, an HTTP `422 Unprocessable Entity` is returned to the client, detailing exactly which heuristic was breached (e.g., "aspect ratio too wide" or "corners too bright").
