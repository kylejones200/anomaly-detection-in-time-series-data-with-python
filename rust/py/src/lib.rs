use anomaly_detection_in_time_series_data_with_python_core::detect_anomalies_zscore;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn detect_anomalies_zscore_py<'py>(py: Python<'py>, values: PyReadonlyArray1<f64>, threshold: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(detect_anomalies_zscore(values.as_slice()?, threshold).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (values, threshold, iterations=500))]
fn bench_kernel_py(values: PyReadonlyArray1<f64>, threshold: f64, iterations: usize) -> PyResult<f64> {
    let v = values.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = detect_anomalies_zscore(&v, threshold);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn anomaly_detection_in_time_series_data_with_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_anomalies_zscore_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
