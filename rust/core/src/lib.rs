//! Global z-score anomaly flags for a univariate series.

/// Return 1.0 where |z| > threshold, else 0.0 (population mean/std).
pub fn detect_anomalies_zscore(values: &[f64], threshold: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt().max(1e-12);
    values
        .iter()
        .map(|v| {
            if ((v - mean) / std).abs() > threshold {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}
