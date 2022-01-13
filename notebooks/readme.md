# Notebooks Test-Area

If you're not interested in performing the parameter sweeps yourself and only want to play around with our code / framework, we provide 2 notebooks in this directory.

Please install the [requirements.txt](https://omi-gitlab.e-technik.uni-ulm.de/aml/anomalygan/-/blob/master/Docker/AnomalyGAN/requirements.txt) via pip.

## AnomalyDetector

To test our anomaly detector, you can use the **cdnTrainDetector.ipynb**.
This notebook provides the necessary code to train a detector locally.
The output of the trained detector is stored within this folder.

## AnomalyGAN

To test our AnomalyGAN, you can use the **cdnAnomalyGAN.ipynb**.
If you want to evaluate the success by an AnomalyDetector, please provide a pre-trained detector.
