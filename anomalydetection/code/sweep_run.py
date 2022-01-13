from anomalyDetector_lib import *
import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import sklearn
import sys
import uuid
import wandb

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
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--modified_loss", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wandb_projekt", type=str, default="default_value")
    parser.add_argument("--architecture", type=str, default="TCN")
    parser.add_argument("--number_reruns", type=int, default=1)


    ## TCN arguments

    parser.add_argument("--channels", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--tcn_dropout", type=float, default=0.2)

    ## LSTM arguments

    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=str2bool, default="False")

    
    args = parser.parse_args()
    
    if args.architecture=="TCN":
        model_config = {
                "architecture": "TCN",
                "channels": args.channels,
                "num_layers": args.num_layers,
                "kernel_size": args.kernel_size,
                "dropout": args.tcn_dropout,
        }
    else:
        model_config={
                "architecture": "LSTM",
                "hidden_size": args.hidden_size,
                "dropout": args.lstm_dropout,
                "bidirectional": args.bidirectional
        }
        
    training_data_config = {
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_size": 0.9,
        "val_size": 0.05,
        "test_size": 0.05,
        "patience": args.patience,
        "modified_loss": args.modified_loss,
    }
    
    print(f"Number of Re-Runs: {args.number_reruns}")
    test_mse_array = []
    test_auc_score_array = []
    test_f1_array = []
    test_precision_array = []
    test_recall_array = []
    train_mse_array = []
    train_auc_array = []
    train_f1_array = []
    train_precision_array = []
    train_recall_array = []

    
    for i in range(args.number_reruns):
        print(f"Model Number: {i}")
        x,y,labels = load_data(args.window_size)

        training_data_config["multivariates"] = x.shape[-1]

        anomalyDetector = AnomalyDetector(training_data_config,model_config,(x,y,labels))

        anomalyDetector.create_model()
        anomalyDetector.create_datasets()

        date_time = now.strftime("%Y_%d_%m/")
    
        save_path = "/opt1/out/" + date_time + args.wandb_projekt

        
        test_mse, test_auc_score, test_f1, test_precision, test_recall, train_mse, train_auc_score, train_f1, train_precision, train_recall = anomalyDetector.train(wandb_sweep=True, save_path=save_path, save_postix=str(i))
        
        test_mse_array.append(test_mse)
        test_auc_score_array.append(test_auc_score)
        test_f1_array.append(test_f1)
        test_precision_array.append(test_precision)
        test_recall_array.append(test_recall)
        train_mse_array.append(train_mse)
        train_auc_array.append(train_auc_score)
        train_f1_array.append(train_f1)
        train_precision_array.append(train_precision)
        train_recall_array.append(train_recall)
    

    wandb.run.summary["test_loss"] = np.mean(test_mse_array)
    wandb.run.summary["test_auc"] = np.mean(test_auc_score_array)
    wandb.run.summary["test_f1"] = np.mean(test_f1_array)
    wandb.run.summary["test_precision"] = np.mean(test_precision_array)
    wandb.run.summary["test_recall"] = np.mean(test_recall_array)
    wandb.run.summary["train_loss"] = np.mean(train_mse_array)
    wandb.run.summary["train_auc"] = np.mean(train_auc_array)
    wandb.run.summary["train_f1"] = np.mean(train_f1_array)
    wandb.run.summary["train_precision"] = np.mean(train_precision_array)
    wandb.run.summary["train_recall"] = np.mean(train_recall_array)
    
    wandb.run.summary["test_loss_array"] = test_mse_array
    wandb.run.summary["test_auc_score_array"] = test_auc_score_array
    wandb.run.summary["test_f1_array"] = test_f1_array
    wandb.run.summary["test_precision_array"] = test_precision_array
    wandb.run.summary["test_recall_array"] = test_recall_array
    wandb.run.summary["train_loss_array"] = train_mse_array
    wandb.run.summary["train_auc_array"] = train_auc_array
    wandb.run.summary["train_f1_array"] = train_f1_array
    wandb.run.summary["train_precision_array"] = train_precision_array
    wandb.run.summary["train_recall_array"] = train_recall_array

    wandb.log({"test_loss": np.mean(test_mse_array),\
               "test_auc": np.mean(test_auc_score_array),\
               "test_f1": np.mean(test_f1_array),\
               "test_precision": np.mean(test_precision_array),\
               "test_recall": np.mean(test_recall_array),\
               "train_loss": np.mean(train_mse_array),\
               "train_auc": np.mean(train_auc_array),\
               "train_f1": np.mean(train_f1_array),\
               "train_precision": np.mean(train_precision_array),\
               "train_recall": np.mean(train_recall_array),\
               "test_loss_array": test_mse_array,\
               "test_auc_score_array": test_auc_score_array,\
               "test_f1_array": test_f1_array,\
               "test_precision_array": test_precision_array,\
               "test_recall_array": test_recall_array,\
               "train_loss_array": train_mse_array,\
               "train_auc_array": train_auc_array,\
               "train_f1_array": train_f1_array,\
               "train_precision_array": train_precision_array,\
               "train_recall_array": train_recall_array})

    print("Finished")
    return
        
def apply_sliding_window(X,aMetaInformation,window_size):
    
    
    return_x = np.zeros(shape=[X.shape[0]-window_size,window_size,X.shape[1]])
    return_y = np.zeros(shape=[X.shape[0]-window_size,X.shape[1]])
    return_label = np.zeros(shape=[X.shape[0]-window_size,1])
    
    for i in range(0, int(X.shape[0]-window_size)):
        return_x[i] = X[i:i+window_size]
        return_y[i] = X[i+window_size]
        return_label[i] = aMetaInformation[i+window_size]
        
    return return_x.reshape(-1,window_size,X.shape[1]), return_y.reshape(-1,X.shape[1]), return_label.reshape(-1,1)
     

def load_data(window_size):
    
    dataInput = np.load("../data/data.npy")
    condInput = np.load("../data/condInput.npy")
    
    x,y,labels = apply_sliding_window(dataInput, condInput, window_size=window_size)
    
    return x,y,labels


if __name__ == "__main__":

    with wandb.init() as run:
        main()