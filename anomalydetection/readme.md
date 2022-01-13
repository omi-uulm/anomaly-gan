# AnomalyDetector

This folder provides the code necessary to train our Anomaly Detector.

To illustrate a parameter search to find the best configuration of an Anomaly Detector, we include the configurations for the CDN data set, which can be utilized to create a WandB Sweep on your own.

For this task you have to have access to this data set. 
You can modify the configurations if you want to train an AnomalyDetector on a different data set.


## Code

The Code folder contains the python scripts, which include the Anomaly Detector.
We implemented a TCN as well as LSTM Dectector, but chose to only use TCN Detectors based on preliminary results.


## Pre-Trained Detectors

If you do not want to perform the sweep on your own, we provide our pre-trained detector in the **AnomalyGAN** folder.
You can use these detectors to evaluate the AnomalyGANs.

## Sweep

This folder includes the necessary .yaml file to start a Wandb Sweep.
For further information on sweeps we refer the reader to [Wandb Sweep](https://docs.wandb.ai/guides/sweeps).

For our hyperparameter search we includeded the following parameters:

- Learning Rate
- Window Size
- Batch Size
- Channels of TCN
- Number of TCN Layers
- Kernel Size
- TCN Dropout
- Loss Function

We trained the models for up to 500 epochs, with an early stopping included, if the validation auc-score does not improve over the last 150 epochs.
We modified the code to rerun each configuation 3 times. 
We noticed, that the performance of the AnomalyDetectors could vary based on the weights initalization and train/test/val split.
Therefore, we wanted to measure the stability of each configuration.

To create a wandb sweep you can run 

```
wandb sweep tcn_sweep.yaml
```

The output of this command can be as follows:

```
wandb: Creating sweep from: tcn_sweep.yaml
wandb: Created sweep with ID: rcg5qk3s
wandb: View sweep at: https://wandb.ai/alochner/anomalyDetector-CDN-Sweep/sweeps/rcg5qk3s
wandb: Run sweep agent with: wandb agent alochner/anomalyDetector-CDN-Sweep/rcg5qk3s
```

For running the Sweep we need to copy the string **alochner/anomalyDetector-CDN-Sweep/rcg5qk3s**

## Running Sweep

With the sweep created and the docker image built we can run the sweep.

To run the sweep you can run

```
./start_sweep.sh alochner/anomalyDetector-CDN-Sweep/rcg5qk3s
``` 

OR

```
docker run -d -it\
  --name AnomalyDetectorSweep \
  --mount type=bind,source="$(pwd)"/code/,target=/opt1/program \
  --mount type=bind,source="$(pwd)"/results/,target=/opt1/out \
  --mount type=bind,source="$(pwd)"/../data/,target=/opt1/data/ \
  anomalygan alochner/anomalyDetector-CDN-Sweep/rcg5qk3s
```

Modify this command to your requirements (GPU / CPU Usage).

If you have access to mulitple workers at the same time, you can run multiple containers to perform the sweep on mulitple agents in parallel.

You can monitor the sweep and the results on WandB.

**Note: As we implemented a random search, you need to stop the sweep manually.**

The trained models will be saved in the results folder.
The folder contains all WandB runs (= one tested configuration). Within each run, there will be 3 saved models, because of the 3 reruns.
You can chose the best model accordingly and move to training the AnomalyGAN.




