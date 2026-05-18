# Anomaly Detection in Time Series Data with Python

This project demonstrates anomaly detection techniques for time series data.

## Business context

Anomaly detection identifies unusual patterns or outliers that deviate significantly from the expected behavior in a time series. These appraochs are commonly used in predictive maintenance, fraud detection, financial monitoring, and system health diagnostics. But these techniques can be used any time we have time series as a sanity check (does the income data make sense?)

<figcaption>Photo by <a class="markup--anchor markup--figure-anchor" rel="photo-creator noopener" target="_blank">Vivek Doshi</a> on <a class="markup--anchor markup--figure-anchor"

This article focuses on two anomaly detection techniques Isolaiton Forest and Autoencoders (deep learning).

## Article

Medium article: [Anomaly Detection in Time Series Data with Python](https://medium.com/gitconnected/anomaly-detection-in-time-series-data-with-python-5a15089636db)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Anomaly detection functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Detection method (isolation_forest, statistical)
- Model parameters (contamination, threshold)
- Output settings

## Anomaly Detection Methods

### Isolation Forest
- Unsupervised learning approach
- Handles multivariate data
- Effective for high-dimensional spaces

### Statistical Method (Z-score)
- Simple and interpretable
- Based on standard deviations
- Fast computation

## Caveats

- By default, generates synthetic time series with injected anomalies.
- Isolation Forest requires feature engineering (lagged features).
- Statistical method assumes normal distribution.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).