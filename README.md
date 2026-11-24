<<<<<<< HEAD
# AXDS web system - Grad-CAM heat-map

## 1. Add the Grad-CAM model
Run the notebook cell `notebook_cell_export_gradcam.py` (after training) and copy the result into this folder:

    efficientnetb3_gradcam.tflite

It must come from the SAME training run as your classifier. The server then uses this single model for both the
prediction and the heat-map, so the label and the map always agree.
If the file is missing the API still works (using `efficientnetb3_covid19_model.tflite`) but returns `"heatmap": null`
and the page hides the Grad-CAM card.

## 2. Run
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000
Open `index.html` in a browser (or serve it with `python -m http.server 5500`).
Check the server: http://localhost:8000/health  ->  "heatmap": true

To point the page at another server, add before the main script: `<script>window.AXDS_API_URL='https://your-api'</script>`

## API
POST /predict (multipart field `file`) ->
    { prediction, confidence, all_scores, inference_ms,
      heatmap: { method, class, width, height, data[width*height] } | null }
`data` is row-major, values 0..1. The browser upsamples and colours it.

Optional env vars: GRADCAM_MODEL_PATH, CLASSIFIER_MODEL_PATH, MAX_UPLOAD_MB (default 20), ALLOWED_ORIGINS (default *).
=======
# AI-Powered-Chest-X-Ray-system---Web-App
>>>>>>> c43b58786d1443e9a195bf71240dc804b8ea9d36
