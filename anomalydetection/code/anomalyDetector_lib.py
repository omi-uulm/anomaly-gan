import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
import json
import os
import wandb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score,auc, roc_curve

from lstm_lib import *
from tcn_lib import * 

class AnomalyDetector:
    def __init__(self, training_data_config,model_config, data,seed=None):
        
        self.training_data_config = training_data_config
        self.model_config = model_config
        
        # Data Parameters
        self.window_size = training_data_config["window_size"]
        self.multivariates = training_data_config["multivariates"]
        
        
        
        
        # Optimizer Parameters
        self.batch_size = training_data_config["batch_size"]
        self.lr = training_data_config["lr"]
        self.epochs = training_data_config["epochs"]
        self.patience = training_data_config["patience"]
        
        if "modified_loss" in training_data_config:
            loss = nn.MSELoss(reduction = "none")
            
            if training_data_config["modified_loss"] == "minmax":
                self.train_error_func = lambda output,y,label: torch.mul(label*-2+1,loss(output,y)).mean()
                
            elif training_data_config["modified_loss"] == "0":
                self.train_error_func = lambda output,y,label: torch.mul(label*-1+1,loss(output,y)).mean()
            else:
                loss = nn.MSELoss()
            
                self.train_error_func = lambda output,y,label: loss(output,y)
                
        else: 
            loss = nn.MSELoss()
            
            self.train_error_func = lambda output,y,label: loss(output,y)
        
        
        # Training Parameters
        
        self.train_size = training_data_config["train_size"]
        self.test_size = training_data_config["test_size"]
        self.val_size = training_data_config["val_size"]
        
        assert (self.train_size + self.test_size + self.val_size == 1)
        
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.data = data
        
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        print(f"Device: {self.device}")
        
    def create_model(self):
        
        if self.model_config["architecture"] == "LSTM":
            
            self.model = LSTMAnomalyDetector(input_size=self.multivariates,hidden_size=self.model_config["hidden_size"],dropout=self.model_config["dropout"],output_size=self.multivariates,bidirectional=self.model_config["bidirectional"]).to(self.device)
        
        else:
            self.model = TCNAnomalyDetector(input_size=self.multivariates,input_length=self.window_size, output_size=self.multivariates, channels=self.model_config["channels"], num_layers=self.model_config["num_layers"], kernel_size=self.model_config["kernel_size"],dropout=self.model_config["dropout"]).to(self.device)
    
    
    def create_datasets(self):
    
        X,y,labels = self.data
        
        X_tensor = torch.Tensor(X).to(self.device)
        self.X_tensor = X_tensor
        y_tensor = torch.Tensor(y).to(self.device)
        self.y_tensor = y_tensor
        labels_tensor = torch.Tensor(labels).to(self.device)
        self.labels_tensor = labels_tensor
        
        my_dataset = TensorDataset(X_tensor,y_tensor,labels_tensor)
        
        train_size = int(self.train_size * len(my_dataset))
        val_size = int(self.val_size * len(my_dataset))
        test_size = len(my_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, val_size, test_size])
        

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset),shuffle=False)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset),shuffle=False)
        
    def train(self,use_wandb=False, wandb_projekt=None, wandb_run_name=None, wandb_sweep=False, save_path="./", save_postix=None):
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
                
        self.model.threshold = None
        
        
        
        self.training_losses_history = []
        self.val_losses_history = []
        
        self.use_wandb = use_wandb
        
        self.save_path = save_path

        if use_wandb and not wandb_sweep:
            os.environ["WANDB_API_KEY"] = json.load("../../Docker/wandbkey.json")['APIKey']
            wandb.init(config={"training_conf": self.training_data_config}, project=wandb_projekt)
            if wandb_run_name:
                wandb.run.name = wandb_run_name
            print("wandb init")
            
            self.save_path = save_path + "/" + wandb.run.name

        

        if save_postix:
            self.save_path = self.save_path + "/" + save_postix

        if self.save_path: 
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                print("Directory " , self.save_path ,  " Created ")
            else:    
                print("Directory " , self.save_path ,  " already exists")

        earlyStopping = EarlyStopping(patience=self.patience,verbose=True, delta=0, path=(self.save_path + "/best_model.pt") if self.save_path else None)

        

        
        
        for epoch in range(self.epochs):
            
            self.model.train()
            train_losses = []
            for x,y,label in iter(self.train_dataloader):
                
                self.model.zero_grad()
                output = self.model(x)
                
                loss = self.train_error_func(output,y,label)
                train_losses.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()
                
            

            errors = []
            val_labels = []
            val_losses = []


            """for val_x,val_y,val_label in iter(self.val_dataloader):

                val_pred = self.model(val_x)"""
            
                
                #error = np.sqrt(np.mean((val_y.detach().cpu().numpy() - val_pred.detach().cpu().numpy())**2,axis=-1))

                #errors.append(error)
                #val_labels.append(val_label.detach().cpu.numpy())
                #val_losses.append(self.loss(val_pred, val_y).detach().cpu.numpy())
                

            self.model.eval()
            val_x,val_y,val_label = next(iter(self.val_dataloader))
            val_pred = self.model(val_x)

            mse = np.mean((val_y.detach().cpu().numpy() - val_pred.detach().cpu().numpy())**2,axis=-1)
            
            rmse = np.sqrt(mse)
            
            auc_score = self.calculate_auc_scores(rmse, val_label.detach().cpu().numpy())
          
            
            train_loss = np.mean(train_losses)
            val_loss=np.mean(mse)
            
            self.training_losses_history.append(train_loss)
            self.val_losses_history.append(val_loss)

            
            earlyStopping(auc_score, self.model, minimize=False)
            
            if use_wandb and not wandb_sweep:
                wandb.log({"avg_train_loss": train_loss, "val_loss": val_loss, "val_auc": auc_score})
                    
            
            print(f"Epoch {epoch}: Train-Loss: {train_loss}, Val-Loss: {val_loss}, Val-AUC: {auc_score}")
            
            if earlyStopping.early_stop:
                print("Training Finished")
                break
            
        
        ### Load Best Model
        
        if save_path:
        
            self.model = torch.load(self.save_path + "/best_model.pt")
              
        
        self.model.eval()
        
        
        
        
        ## Calculate Dynamic Threshold based on Train Set
        
        train_f1, train_precision, train_recall, train_auc_score,train_mse = self.eval_data_set(self.X_tensor[:10000], self.y_tensor[:10000], self.labels_tensor[:10000])
        
        ## Eval Test Set
        
        test_x, test_y,test_label = next(iter(self.test_dataloader))
        test_f1, test_precision, test_recall, test_auc_score,test_mse = self.eval_data_set(test_x, test_y, test_label)
    
        
        if use_wandb and not wandb_sweep:
            wandb.run.summary["test_loss"] = np.mean(test_mse)
            wandb.run.summary["test_auc"] = test_auc_score
            wandb.run.summary["test_f1"] = test_f1
            wandb.run.summary["test_precision"] = test_precision
            wandb.run.summary["test_recall"] = test_recall
            wandb.run.summary["train_loss"] = np.mean(train_mse)
            wandb.run.summary["train_auc"] = train_auc_score
            wandb.run.summary["train_f1"] = train_f1
            wandb.run.summary["train_precision"] = train_precision
            wandb.run.summary["train_recall"] = train_recall
            wandb.log({"test_loss": np.mean(test_mse), "test_auc": test_auc_score,
                      "test_f1": test_f1, "test_precision": test_precision, "test_recall": test_recall,
                       "train_loss": np.mean(train_mse), "train_auc": train_auc_score,
                      "train_f1": train_f1, "train_precision": train_precision, "train_recall": train_recall,
                      })
        
        print(f"Train Loss: {np.mean(train_mse)}, AUC: {train_auc_score}, F1: {train_f1}, Precision: {train_precision}, Recall: {train_recall}")
        print(f"Test Loss: {np.mean(test_mse)}, AUC: {test_auc_score}, F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall}")
        
        if save_path:
        
            torch.save(self.model, self.save_path + "/best_model.pt")

            with open(f'{self.save_path}/training_data_config.json', 'w', encoding='utf-8') as f:
                json.dump(self.training_data_config, f, ensure_ascii=False, indent=4)

            with open(f'{self.save_path}/model_config.json', 'w', encoding='utf-8') as f:
                json.dump(self.model_config, f, ensure_ascii=False, indent=4)
        
        return np.mean(test_mse), test_auc_score, test_f1, test_precision, test_recall, np.mean(train_mse), train_auc_score, train_f1, train_precision, train_recall,
     
    def eval_data_set(self, X,y,labels):
        
                  
        pred = self.model(X)

        mse = np.mean((y.detach().cpu().numpy() - pred.detach().cpu().numpy())**2,axis=-1)

        rmse = np.sqrt(mse)
        
        f1,precision,recall = self.calculate_eval_scores(rmse,labels.detach().cpu().numpy())
        
        auc_score = self.calculate_auc_scores(rmse, labels.detach().cpu().numpy(),print=False)
        
        return f1, precision, recall, auc_score,mse
    """
    def eval_data_set(self, X,y,labels,batchsize=None):
        
        if not batchsize:
            batchsize = X.shape[0]
        
        dataset = TensorDataset(X,y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,shuffle=False)
        
        mse = []
        rmse = []
        for x,y in dataloader:
        
            pred = self.model(x)

            mse_ = np.mean((y.detach().cpu().numpy() - pred.detach().cpu().numpy())**2,axis=-1)
            
            rmse_ = np.sqrt(mse_)
            mse.extend(mse_)

            rmse.extend(rmse_)
            
        rmse = np.array(rmse).reshape(labels.shape)
        mse = np.array(mse).reshape(labels.shape)
        
        print(rmse.shape)
        print(mse.shape)
        
        f1,precision,recall = self.calculate_eval_scores(rmse,labels.detach().cpu().numpy())
        
        auc_score = self.calculate_auc_scores(rmse, labels.detach().cpu().numpy(),print=False)
        
        return f1, precision, recall, auc_score,mse"""
        
    
    def calculate_eval_scores(self, pred_errors, labels,print_figure=True):
        
        if not self.model.threshold:
            ## Calculate Threshhold if not already done   
            median = float(np.median(pred_errors))
            stdev = float(np.std(pred_errors))
            threshold_array = np.linspace(median, median + (10 * stdev), num=100)

            precision = []
            recall = []
            f1_score = []

            for t in threshold_array:
                true_positives = np.multiply((pred_errors > t),labels.reshape(-1,)).sum()
                false_positives = np.multiply((pred_errors > t), ((pred_errors-1)**2).reshape(-1,)).sum()

                false_negatives = np.multiply((pred_errors < t),labels.reshape(-1,)).sum()

                if (true_positives + false_positives > 0):
                    precision.append(true_positives / (true_positives + false_positives))
                else:
                    precision.append(0)

                if (true_positives + false_negatives) != 0:
                    recall.append(true_positives / (true_positives + false_negatives))
                else:
                    recall.append(0)
                if (precision[-1] + recall[-1]) != 0:
                    f1_score.append(2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
                else:
                    f1_score.append(0)

            if print_figure:
                plt.figure()
                plt.plot(threshold_array,f1_score,label="f1 Score")
                plt.plot(threshold_array,precision,label="Precision")
                plt.plot(threshold_array,recall,label="Recall")

                plt.legend()
        
            argmax = np.argmax(f1_score)
            print(f"Set threshold to {threshold_array[argmax]}")
            self.model.threshold = threshold_array[argmax]
            return f1_score[argmax], precision[argmax], recall[argmax]
        
        
        else:
            print(f"Using Threshold == {self.model.threshold}")
            true_positives = np.multiply((pred_errors > self.model.threshold),labels.reshape(-1,)).sum()
            false_positives = np.multiply((pred_errors > self.model.threshold), ((pred_errors-1)**2).reshape(-1,)).sum()

            false_negatives = np.multiply((pred_errors < self.model.threshold),labels.reshape(-1,)).sum()

            if (true_positives + false_positives > 0):
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0

            if (true_positives + false_negatives) != 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0
            if (precision + recall) != 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0 
            return f1_score, precision, recall
        
        
        
        
    def calculate_auc_scores(self,pred_erros, labels,print=False):
        
        
        if print:
            plt.figure()
        
            _ = plt.hist(pred_erros[labels[:,0] == 1],density=True,label="Anomaly Scores")
            _ = plt.hist(pred_erros[labels[:,0] == 0],density=True,label="Normal Scores")
            plt.legend()

        return roc_auc_score(labels,pred_erros)
            
            
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, minimize=True):

        
        score = -val_loss if minimize else val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        if self.path:
            torch.save(model, self.path)
        self.val_loss_min = val_loss    
        
        