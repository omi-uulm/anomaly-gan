from anomalyDetector_lib import *
from anomalyGAN import *
import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import sklearn
import wandb
import sys
import uuid

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    
    now = datetime.now() # current date and time
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wandb_projekt", type=str, default="default_value")


    ## DataSet Arguments
    parser.add_argument("--detector_path", type=str, default="./tcn_model")
    parser.add_argument("--sequence_length", type=int, default=256)


    ## TCN arguments

    parser.add_argument("--channels", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--tcn_dropout", type=float, default=0.2)

    ## GAN arguments

    parser.add_argument("--d_steps", type=int, default=1)
    parser.add_argument("--g_steps", type=int, default=1)

    
    args = parser.parse_args()

    aData, aMetaInformation, sampleAnomalyC = load_data(args.sequence_length)

    anomal_model_config = {
        "d_steps": args.d_steps,
        "g_steps": args.g_steps,
        "lr": args.lr,
        "wgan": False,
        "sampleC_fun": sampleAnomalyC,
        "optim": "ADAM",
        "batch_size": args.batch_size,
        "architecture": ["TCN", "TCN"],
        "Generator": {
            "channels": args.channels,
            "num_layers": args.num_layers,
            "kernel_size": args.kernel_size,
            "dropout": args.tcn_dropout        
        },
        "Discriminator": {
            "channels": args.channels,
            "num_layers": args.num_layers,
            "kernel_size": args.kernel_size,
            "dropout": args.tcn_dropout        
        },
        "scale": True,
        "num_fixed_noises": 200,
        "patience": 150
    }

    anomal_model_config["z_latent_dim"] = aData.shape[-1]
        
    training_data_config = {
        "num_data_points": args.sequence_length,
        "z_seq_length": args.sequence_length,
        "num_frequencies": 4,
        "num_anomalies": 0
    }

    training_data_config["multi_variates"] = aData.shape[-1]
               
    ### Load Pre-trained Anomaly Detector

    ad_training_data_config = json.load( open( f"{args.detector_path}/training_data_config.json" ) )
    ad_tcn_model_config = json.load( open( f"{args.detector_path}/model_config.json" ) )
    
    anomalyDetector = AnomalyDetector(ad_training_data_config,ad_tcn_model_config,None)
    
    pre_trained_model = torch.load(f"{args.detector_path}/best_model.pt", map_location=torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu"))

    anomalyDetector.model = pre_trained_model

    anomalyGAN = AnomalyGAN(training_data_config, anomal_model_config, aData,aMetaInformation,anomalyDetector)
    
    date_time = now.strftime("%Y_%d_%m/")
    
    save_path = "/opt/out/" + date_time + args.wandb_projekt

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory " , save_path ,  " Created ")
    else:    
        print("Directory " , save_path ,  " already exists")
    
    anomalyGAN.train(args.epochs,use_wandb=True, wandb_projekt=args.wandb_projekt, print_iterations=10,save_path=save_path, wandb_sweep=True)
    
    #save_model(anomalyGAN, save_path, anomal_model_config, training_data_config)


def save_model(ganModel,save_path, model_config, training_data_config):
    
        
    torch.save(ganModel.netD.state_dict(), f"{save_path}/discriminator.pt")
    
    torch.save(ganModel.netG.state_dict(), f"{save_path}/generator.pt")

    model_config.pop('sampleC_fun', None)
    
    with open(f"{save_path}/model_config.json", 'w') as file:
     file.write(json.dumps(model_config)) 
    
    with open(f"{save_path}/training_data_config.json", 'w') as file:
     file.write(json.dumps(training_data_config)) 
        
def extract_anomal_data(time_series, meta_information, length, skip_points=1):
    
    complete_length = time_series.shape[0]
    
    return_array = []
    
    return_meta_information = []
    
    for i in range(0, int((complete_length-length)/skip_points)):
        
        return_array.append(time_series[i:i+length])
        
        return_meta_information.append(meta_information[i:i+length])
        
        
    return np.array(return_array), np.array(return_meta_information)

def load_data(sequence_length=256):
    
    aCDN = np.load("../data/data.npy")
    anomalyInformation = np.load("../data/condInput.npy")

    def sampleAnomalyC(batch_size):
        return torch.Tensor(aMetaInformation[np.random.randint(0,aMetaInformation.shape[0],size=batch_size)])
    
    aData, aMetaInformation = extract_anomal_data(aCDN, anomalyInformation, sequence_length)

    return aData, aMetaInformation, sampleAnomalyC

if __name__ == "__main__":
    main()
