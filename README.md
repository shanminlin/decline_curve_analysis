# Decline-Curve-Analysis

In this project, we aim to develop a pipeline to fit the exponential, harmonic, hyperbolic and
hyperbolic-to-exponential decline curves to well production data and evaluate the
performance of the fit. 
To investigate the effect of outliers, we explored three methods for outlier detection, namely
one-class support vector machine, isolation forest and local outlier factor. All three methods
are able to detect significant outliers. Support vector machine and local outlier factor are
able to capture the shape of the training data closely. However, most recent declines are
often overlooked. On the other hand, isolation forest is able to capture the most recent
declines, which are more important in our forecast. Therefore,  we employed the isolation
forest algorithm in outlier filtering.

This project contains the following files:

- `DCA.py` - main script for curve fitting, prediction and evalution.
- `test.py` - script for unit tests.
