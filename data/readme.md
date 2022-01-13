# Data

This data folder provides the anomal time series, which are used to train our AnomalyGAN.

Please enter your own time series data to this folder to use our code.

In general, our code expects two numpy files, namely **data.npy** and **condInput.npy**.

## data.npy

This numpy file must contain the multi-variate time series.
The saved numpy array must have the following shape:

[#time-steps, #channels]


## condInput.npy

This numpy file must contain the information, at which time points the multi-variate time series of data.npy contains an anomaly.
The saved numpy array must have the following shape:

[#time-steps, 1]
