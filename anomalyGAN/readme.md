# AnomalyGAN

After training the AnomalyDetector on a data set, this folder can be utilized to train an AnomalyGAN on this data set.

## Code

The Code folder contains the python scripts, which include the AnomalyGAN.
We implemented a TCN as well as LSTM GAN, but chose to only use TCN GANs based on preliminary results.

## Pre-Trained Detectors

If you do not want to perform the sweep on your own, we provide our pre-trained detectors in the **pretrained-detectors** folder.
If you performed a wandb sweep for the AnomalyDetector yourself, you can include the detector in this folder.

## Sweep

This folder includes the necessary .yaml file to start a Wandb Sweep for our CDN data set.
For further information on sweeps we refer the reader to [Wandb Sweep](https://docs.wandb.ai/guides/sweeps).

For our hyperparameter search we includeded the following parameters:

- Alternate Training Setup (d_steps, g_steps)
- Kernel Size
- Batch Size
- TCN Dropout

We trained the models for up to 5000 epochs, with an early stopping included, if the auc-score of the generated time series does not improve over the last 1500 epochs. 

To create a wandb sweep you can run 

```
wandb sweep GAN_sweep.yaml
```

The output of this command can be as follows:

```
wandb: Creating sweep from: GAN_sweep.yaml
wandb: Created sweep with ID: rcg5qk3s
wandb: View sweep at: https://wandb.ai/alochner/CDNGANSweep/sweeps/rcg5qk3s
wandb: Run sweep agent with: wandb agent alochner/CDNGANSweep/rcg5qk3s
```

For running the Sweep we need to copy the string **alochner/CDNGANSweep/rcg5qk3s**

## Running Sweep

With the sweep created and the docker image built, we can run the sweep.


To run the sweep you can run:

```
./start_sweep.sh alochner/alochner/CDNGANSweep/rcg5qk3s
``` 

OR

```
docker run -d -it\
  --name GANSweep \
  --mount type=bind,source="$(pwd)"/code/,target=/opt1/program \
  --mount type=bind,source="$(pwd)"/results/,target=/opt1/out \
  --mount type=bind,source="$(pwd)"/../data/,target=/opt1/data/ \
  --mount type=bind,source="$(pwd)"/pretrained-detectors/,target=/opt1/pretrained-detectors/ \
  anomalygan alochner/alochner/CDNGANSweep/rcg5qk3s
``` 

If you have access to mulitple workers at the same time, you can run multiple containers to perform the sweep on mulitple agents in parallel.

You can monitor the sweep and the results on WandB.

**Note: As we implemented a random search, you need to stop the sweep manually.**

The trained models will be saved in the results folder.
The folder contains all WandB runs (= one tested configuration).
You can chose the best model and generate time series locally.




