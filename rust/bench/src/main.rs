use anomaly_detection_in_time_series_data_with_python_core::detect_anomalies_zscore;

fn main() {
    let v: Vec<f64> = (0..10000).map(|i| (i as f64 * 0.01).sin()).collect();
    for _ in 0..500 {
        let _ = detect_anomalies_zscore(&v, 3.0);
    }
}
