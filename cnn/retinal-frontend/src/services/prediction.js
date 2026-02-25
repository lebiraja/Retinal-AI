import api from './api';

/**
 * Transform raw backend PredictionResponse into frontend-friendly shape.
 *
 * Backend returns:
 *   predictions: { "DR": 0.92, "BRVO": 0.78 }         (Dict[str, float])
 *   detected_diseases: ["DR", "BRVO"]                   (List[str])
 *   detected_diseases_full: ["Diabetic Retinopathy", …] (List[str])
 *   top_prediction: "DR"
 *   confidence: 0.85
 *   risk_level: "HIGH"
 *   advisory: "..."
 *   elapsed_ms: 150.2
 *   threshold: 0.5
 *   disclaimer: "..."
 *
 * Frontend needs:
 *   predictions: [{ disease, fullName, confidence }]  (sorted desc)
 *   processing_time: seconds
 *   plus all other fields passed through
 */
function transformPrediction(raw) {
  const detectLabels = raw.detected_diseases || [];
  const detectFull = raw.detected_diseases_full || [];

  /* Convert dict → sorted array */
  const predictions = Object.entries(raw.predictions || {})
    .sort(([, a], [, b]) => b - a)
    .map(([label, confidence]) => {
      const idx = detectLabels.indexOf(label);
      const fullName = idx >= 0 ? detectFull[idx] : label;
      return { disease: label, fullName, confidence };
    });

  return {
    diseaseRisk: raw.disease_risk,
    predictions,
    numDetected: raw.num_detected,
    topPrediction: raw.top_prediction,
    confidence: raw.confidence,
    riskLevel: raw.risk_level,
    advisory: raw.advisory,
    elapsedMs: raw.elapsed_ms,
    threshold: raw.threshold,
    disclaimer: raw.disclaimer,
    processing_time: raw.elapsed_ms / 1000,
  };
}

/**
 * Send a fundus image to the prediction endpoint
 * @param {File} imageFile
 * @param {number} [threshold=0.5]
 * @returns {Promise<Object>} transformed prediction response
 */
export async function predictImage(imageFile, threshold = 0.5) {
  const formData = new FormData();
  formData.append('image', imageFile);

  const { data } = await api.post('/predict', formData, {
    params: { threshold },
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return transformPrediction(data);
}

/**
 * Check backend health
 * @returns {Promise<Object>}
 */
export async function checkHealth() {
  const { data } = await api.get('/health');
  return data;
}

/**
 * Get model info
 * @returns {Promise<Object>}
 */
export async function getModelInfo() {
  const { data } = await api.get('/info');
  return data;
}
