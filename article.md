# Anomaly Detection in Time Series Data with Python Anomaly detection identifies unusual patterns or outliers that deviate
significantly from the expected behavior in a time series. These...

### Anomaly Detection in Time Series Data with Python
Anomaly detection identifies unusual patterns or outliers that deviate
significantly from the expected behavior in a time series. These
appraochs are commonly used in predictive maintenance, fraud detection,
financial monitoring, and system health diagnostics. But these
techniques can be used any time we have time series as a sanity check
(does the income data make sense?)


<figcaption>Photo by <a
class="markup--anchor markup--figure-anchor"
rel="photo-creator noopener" target="_blank">Vivek Doshi</a> on <a
class="markup--anchor markup--figure-anchor"


This article focuses on two anomaly detection techniques Isolaiton
Forest and Autoencoders (deep learning).

#### What is Anomaly Detection?
Anomalies are weird things. We classify anomalies in time series into
three types. Point Anomalies are individual values that are
significantly different (e.g., a sudden spike in temperature).
Contextual anomalies are values that are unusual within a specific
context, such as seasonality (e.g., higher electricity usage at night).
Collective anomalies are a sequence of values that deviate as a group
(e.g., a long period of low readings).

The goal of anomaly detection is to identify and flag these
irregularities automatically. Once identified, we can choose to exclude
analomaties from our analysis or take some other action (like go fix a
sensor that isn't working as expected).

#### Algorithm: Isolation Forest
The Isolation Forest algorithm is an unsupervised anomaly detection
method, originally introduced by Amazon as Random Cut Forest. It works
by randomly partitioning the data into "trees" using random cuts to
maximize entropy. Anomalies are values where a cut off tree (basically
an isolated point) are very different than the overall values. RCF is
fast, scalable, and effective for high-dimensional and time series data.



This code is using isolation forest to compute anomaly scores for each
point. We are applying a threshold of "weird" at 5%. We can use
matplotlib to see these anomalies visually. It is not clear to me why
some of these points were identified as anomalies.

#### Autoencoders for Anomaly Detection
Autoencoders are deep learning models trained to reconstruct input data.
If the input contains anomalies, the reconstruction error will be higher
for these points because the model focuses on learning the "normal"
patterns of the data.

We train the autoencoder on normal (non-anomalous) data and we see how
much error is normal with our autoencoder. This represents "good." When
an autoencoder is applied to anomalous data, the error will be higher
because the anomalous value is "bad". We can use this to find anomalies
as points with reconstruction errors above a threshold.

Let's try this in python with Keras (based on tensorflow).



<figcaption>Number of anomalies detected: 27</figcaption>


The LSTM autoencoder learns to reconstruct "normal" time series data.
High reconstruction errors correspond to anomalies. We set a threshold
to identify points where reconstruction deviates significantly.

#### Which is better?
RCF is best for simple point anomalies where interpretability and speed
are priorities.

Autoencoders are best for detecting complex anomalies in sequential
data, such as collective and contextual outliers.

#### Next Steps
Anomaly detection is an automated approach for identifying unusual
behaviors (read weird values) in time series data. Random Cut Forest and
Autoencoders are two approaches to effectively detect both simple and
complex outliers and these are good tools for real-world applications
like predictive maintenance, fraud detection, and system monitoring.

#### Related Posts
This article is part of a series of posts on time series forecasting.
Here is the list of articles in the order they were designed to be read.

1.  [[Time Series for Business Analytics with
    Python](https://medium.com/@kylejones_47003/time-series-for-business-analytics-with-python-a92b30eecf62?source=your_stories_page-------------------------------------)]
2.  [[Time Series Visualization for Business Analysis with
    Python](https://medium.com/@kylejones_47003/time-series-visualization-for-business-analysis-with-python-5df695543d4a?source=your_stories_page-------------------------------------)]
3.  [[Patterns in Time Series for
    Forecasting](https://medium.com/@kylejones_47003/patterns-in-time-series-for-forecasting-8a0d3ad3b7f5?source=your_stories_page-------------------------------------)]
4.  [[Imputing Missing Values in Time Series Data for Business Analytics
    with
    Python](https://medium.com/@kylejones_47003/imputing-missing-values-in-time-series-data-for-business-analytics-with-python-b30a1ef6aaa6?source=your_stories_page-------------------------------------)]
5.  [[Measuring Error in Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd?source=your_stories_page-------------------------------------)]
6.  [[Univariate and Multivariate Time Series Analysis with
    Python](https://medium.com/@kylejones_47003/univariate-and-multivariate-time-series-analysis-with-python-b22c6ec8f133?source=your_stories_page-------------------------------------)]
7.  [[Feature Engineering for Time Series Forecasting in
    Python](https://medium.com/@kylejones_47003/feature-engineering-for-time-series-forecasting-in-python-7c469f69e260?source=your_stories_page-------------------------------------)]
8.  [[Anomaly Detection in Time Series Data with
    Python](https://medium.com/@kylejones_47003/anomaly-detection-in-time-series-data-with-python-5a15089636db?source=your_stories_page-------------------------------------)]
9.  [[Dickey-Fuller Test for Stationarity in Time Series with
    Python](https://medium.com/@kylejones_47003/dickey-fuller-test-for-stationarity-in-time-series-with-python-4e4bf1953eed?source=your_stories_page-------------------------------------)]
10. [[Using Classification Model for Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/using-classification-model-for-time-series-forecasting-with-python-d74a1021a5c4?source=your_stories_page-------------------------------------)]
11. [[Measuring Error in Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd?source=your_stories_page-------------------------------------)]
12. [[Physics-informed anomaly detection in a wind turbine using Python
    with an autoencoder
    transformer](https://medium.com/@kylejones_47003/physics-informed-anomaly-detection-in-a-wind-turbine-using-python-with-an-autoencoder-transformer-06eb68aeb0e8?source=your_stories_page-------------------------------------)]

This is another implemetion using data from the [US Electricity Data
Browser.](https://www.eia.gov/electricity/data/browser/)
