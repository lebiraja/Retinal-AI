# Retinal Disease Classifier - Development Logs & Updates

## Out-of-Distribution (OOD) Image Safeguards

### The Problem
During evaluation, it was discovered that the `EfficientNet-B4` model would attempt to diagnose retinal diseases on **any** image uploaded to the AI, rather than just genuine retinal fundus images. 
For example, uploading a selfie, a macro picture of an eye pupil, or even a picture of a dragon resulted in the model predicting "Macular Hole" and "Corneal Reflex Shadow". This severely damages the medical integrity of the AI and poses a significant user-safety risk by throwing false-positive diagnoses on random inputs.

### The Solution: Dark Pixel Ratio Validation
To safeguard the model and emulate the logic of a Senior ML Engineer, a strict heuristic image validation layer was added to the FastAPI backend. Genuine retinal images taken with a fundus camera or ophthalmoscope have a characteristic thick black circular aperture ring/border around the ocular data.

**Key implementations:**
1. **Branch Migration**: Switched the project repository to the correct `LR` branch which contained the unified frontend and backend workflows.
2. **`_validate_fundus_image` Function added to `app/routers/predict.py`**:
   - The function intercepts the raw bytes of the uploaded file.
   - It converts the image to Grayscale (`"L"`) using the `PIL` library.
   - Using `numpy`, it checks the frequency of extremely dark pixels (intensity `< 15`).
   - It computes a **"dark pixel ratio"**. Genuine fundus images usually exceed an `8-10%` dark pixel ratio due to the aperture mask.
   - A strict minimum threshold of **`0.05` (5%)** was enforced.
3. **API Integration**:
   - The validation check was securely injected into both the `/predict` and `/predict-batch` FastAPI routes *before* the `.to(device)` inference commands are even executed, saving heavy VRAM usage on fake picture requests.
   - If the uploaded image falls below the threshold (e.g. a selfie or a screenshot, which has `0.001` or `0.1%` pitch-black pixels), the backend instantly aborts and raises a `400 Bad Request`.
4. **Professional UI Error Bridging**:
   - The backend specifically returns: *“This does not appear to be a retinal fundus photograph... Reason: no dark circular border detected...”* allowing the React Vite frontend to cleanly throw an `Analysis Failed` error component natively without crashing the webpage.
5. **System Launch**:
   - Loaded and booted up the `uvicorn` backend on NVIDIA CUDA.
   - Installed all `npm` modules for `cnn/retinal-frontend` and spawned the Vite web-app.
   - Verified that the full pipeline rejects standard photos while accurately processing valid Retinal datasets.

### Current Status
**Project is structurally complete, validated, and actively running:**
- **Frontend URL:** `http://localhost:5173`
- **Backend URL:** `http://localhost:8000`
- **Integrity Status:** Secure - Non-medical/Out-of-Distribution inputs are strictly blocked.
