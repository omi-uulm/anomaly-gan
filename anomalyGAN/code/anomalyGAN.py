from tcnlib import *
from torch.utils.data import TensorDataset, DataLoader
import torch
from scipy.fft import fft, fftfreq
import torch.nn as nn
from torch.distributions import normal
import copy
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML
import os
import json
import wandb

import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128


class AnomalyGAN:
    
    
    def __init__(self, training_data_config, model_config, data, cond_data, anomalyDetector=None):
        
        training_data_config = training_data_config.copy()
        model_config = model_config.copy()
        
        
        self.num_data_points = training_data_config["num_data_points"]
        self.z_seq_length = training_data_config["z_seq_length"]
        self.num_anomalies = training_data_config["num_anomalies"]
        self.num_frequencies = training_data_config["num_frequencies"]
        
        self.multi_variates = training_data_config["multi_variates"]
        
        self.anomaly_detector=anomalyDetector
        if self.anomaly_detector:
            self.anomaly_detector_window_size = self.anomaly_detector.window_size
        
       
        self.wgan = model_config["wgan"] in ["wgan-cp", "wgan-gp"]
        
        if self.wgan:
            self.wgan_type = self.wgan
            
            if self.wgan_type == "wgan-cp":
                self.weight_cliping_limit = model_config["wgan_cl"]
            else:
                self.gp_weight = model_config["gp_weight"]
        else:
            self.wgan_type = None
        
        
        self.batch_size = model_config["batch_size"]
        self.d_config = model_config["Discriminator"]
        self.g_config = model_config["Generator"]
        self.lr = model_config["lr"]
        
        self.d_steps = model_config["d_steps"]
        self.scale = model_config["scale"]
        self.z_latent_dim = model_config["z_latent_dim"]
        self.num_fixed_noises = model_config["num_fixed_noises"]
        
        if "patience" in model_config:
            self.patience = model_config["patience"]
        else:
            self.patience = 100000
        
        if "sampleC_fun" in model_config:
            self.sampleC_fun = model_config["sampleC_fun"]
        else: 
            self.sampleC_fun = None
            
        if "cond_size" in model_config:
            self.cond_size = model_config["cond_size"]
        else: 
            self.cond_size = 1
        
        self.g_steps = model_config["g_steps"]
        
        
        
        self.optim = model_config["optim"]
        
        self.model_config = model_config
        
        self.training_data_config = training_data_config
                                               
        self.data = data
        
        self.cond_data = cond_data
        
        self.conditional = cond_data is not None
        
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
               
        self.netG = TCNGenerator(input_size=self.z_latent_dim + self.cond_size, channels=self.g_config["channels"], num_layers=self.g_config["num_layers"],output_size=self.num_data_points, kernel_size=self.g_config["kernel_size"], dropout=self.g_config["dropout"], input_length=self.num_data_points, multi_variate=self.multi_variates, output_function=torch.nn.Sigmoid() if self.scale else None).to(self.device)
        
        """self.netD = TCNGenerator(input_size=self.z_latent_dim + (1 if self.conditional else 0), channels=self.g_config["channels"], num_layers=self.g_config["num_layers"],output_size=self.num_data_points, kernel_size=self.g_config["kernel_size"], dropout=self.g_config["dropout"], input_length=self.num_data_points, multi_variate=self.multi_variates, output_function=torch.nn.Sigmoid()).to(self.device)"""
        
        self.netD = TCNDiscriminator(input_size=self.multi_variates + self.cond_size, input_length=self.num_data_points,channels=self.d_config["channels"], num_layers=self.d_config["num_layers"], kernel_size=self.d_config["kernel_size"], dropout=self.d_config["dropout"], num_classes=1, wgan=self.wgan).to(self.device)
        
        
    def _gradient_penalty(self, real_data, generated_data):

        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Pass interpolated data through Critic
        prob_interpolated = self.netD((interpolated, None))

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device), create_graph=True,
                               retain_graph=True)[0]
        # Gradients have shape (batch_size, num_channels, series length),
        # here we flatten to take the norm per example for every batch
        
        #print(gradients.shape)
        gradients = gradients.reshape(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of the
        # square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        
        
    def train(self, num_epochs=10, print_iterations=500, seed=None, use_wandb=False, wandb_projekt=None, wandb_run_name=None, unrolled_steps=0, save_path="./", wandb_sweep=False):
        
        self.unrolled_steps = unrolled_steps
        
        

        if seed:
            torch.manual_seed(seed)
            
        self.use_wandb = use_wandb
        self.save_path = save_path
        if use_wandb:
            if not wandb_sweep:
                os.environ["WANDB_API_KEY"] = json.load(open("../../Docker/wandbkey.json"))['APIKey']
            
            wandb.init(config={"model_conf": self.model_config, "training_conf": self.training_data_config}, project=wandb_projekt)
            if wandb_run_name:
                wandb.run.name = wandb_run_name
                wandb.run.save()
            print("wandb init")
            self.save_path = save_path + "/" + wandb.run.name


        
        if self.save_path:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                print("Directory " , self.save_path ,  " Created ")
            else:    
                print("Directory " , self.save_path ,  " already exists")
        
        self.earlyStopping = GANEarlyStopping(patience=self.patience,verbose=True, delta=0, path=(self.save_path) if self.save_path else None)
    
        print(self.device)
        
        data_tensor = torch.Tensor(self.data).to(self.device)

        if self.conditional:
        
            cond_data_tensor = torch.Tensor(self.cond_data).to(self.device)

            my_dataset = TensorDataset(data_tensor,cond_data_tensor)
        
        else:
            
            my_dataset = TensorDataset(data_tensor)

        self.dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=self.batch_size,shuffle=True)

        return self.train_loop(num_epochs,self.device,self.lr,self.d_steps, print_iterations, num_fixed_noises=self.num_fixed_noises)
    
    
    def evaluate_with_anomaly_detector(self, generated, label, window_size,batchsize=10000):
        
        
        num_samples, seq_length, mvariates = generated.shape
        
        
        anomaly_detector_X = torch.zeros(num_samples, seq_length - window_size, window_size, mvariates).to(self.device)
    
        anomaly_detector_Y = torch.zeros(num_samples, seq_length - window_size, mvariates).to(self.device)
        
        anomaly_detector_label = torch.zeros(num_samples, seq_length - window_size, 1).to(self.device)
        
        for i in range(0, int(seq_length - window_size)):
            anomaly_detector_X[:,i] = generated[:,i:i+window_size]
            anomaly_detector_Y[:,i] = generated[:,i+window_size]
            anomaly_detector_label[:,i] = label[:,i+window_size]
        
        x,y,labels = anomaly_detector_X.reshape(-1,window_size, mvariates), anomaly_detector_Y.reshape(-1, mvariates), anomaly_detector_label.reshape(-1,1)
        
        dataset = TensorDataset(x,y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,shuffle=False)
               
        mse = []
        rmse = []
        for x,y in dataloader:
        
            pred = self.anomaly_detector.model(x)

            mse_ = np.mean((y.detach().cpu().numpy() - pred.detach().cpu().numpy())**2,axis=-1)
            
            rmse_ = np.sqrt(mse_)
            mse.extend(mse_)

            rmse.extend(rmse_)
            
        rmse = np.array(rmse)#.reshape(labels.shape)
        mse = np.array(mse)#.reshape(labels.shape)
                
        f1,precision,recall = self.anomaly_detector.calculate_eval_scores(rmse,labels.detach().cpu().numpy(),print_figure=False)
        
        auc_score = self.anomaly_detector.calculate_auc_scores(rmse, labels.detach().cpu().numpy(),print=False)
        
        
        return f1, precision, recall, auc_score,np.mean(mse)
            
        
        
    
    
    def d_unrolled_loop(self, noise, C, real_label, fake_label,criterion, optimizerD):
        # 1. Train D on real+fake
        self.netD.zero_grad()

        if self.wgan:
            criterion = lambda output, label: torch.mul(output,label).mean(0).view(1)
            if self.wgan_type=="wgan-cp":
                for p in self.netD.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
            #  1A: Train D on real
        
        data = next(iter(self.dataloader))
        
        real_cpu = data[0].to(self.device)#.unsqueeze(2)

        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)


        output = self.netD((real_cpu, data[1] if self.conditional else None)).view(-1)
        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()


        # Generate fake image batch with G


        with torch.no_grad():
            fake = self.netG((noise,C))
        label.fill_(fake_label)




        # Calculate D's loss on the all-fake batch

        output = self.netD((fake.detach(), C)).view(-1)
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        
        if self.wgan_type == "wgan-gp":
            gradient_penalty = self._gradient_penalty(real_cpu, fake)

            gradient_penalty.backward()

        optimizerD.step()


    def train_discriminator(self, data, optimizerD, device, real_label,criterion, fake_label, noise, C):

        if self.wgan:
            criterion = lambda output, label: torch.mul(output,label).mean(0).view(1)
            if self.wgan_type=="wgan-cp":
                for p in self.netD.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
        self.netD.zero_grad()
        real_cpu = data[0].to(device)#.unsqueeze(2)

        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)


        self.netD.zero_grad()

        output = self.netD((real_cpu, data[1] if self.conditional else None)).view(-1)
        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()


        # Generate fake image batch with G


        with torch.no_grad():
            fake = self.netG((noise,C))
        label.fill_(fake_label)




        # Calculate D's loss on the all-fake batch
        
        output = self.netD((fake.detach(), C)).view(-1)
        errD_fake = criterion(output, label)
        

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        
        D_G_z1 = output.mean().item()
        
        if self.wgan_type == "wgan-gp":
        
            gradient_penalty = self._gradient_penalty(real_cpu, fake)

            gradient_penalty.backward()
        
        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D

        optimizerD.step()

        return errD, D_G_z1, D_x

    def train_generator(self, real_label, fake_label, noise, C, criterion, optimizerG):

        self.netG.zero_grad()
        label = torch.full((self.batch_size,), real_label, dtype=torch.float, device=self.device)
        label.fill_(real_label)  # fake labels are real for generator cost
        #label = torch.cat((torch.full((b_size,1), real_label, dtype=torch.float, device=device),torch.full((b_size,1), fake_label, dtype=torch.float, device=device)), dim=1) 
        # Since we just updated D, perform another forward pass of all-fake batch through D
        
        if self.wgan:
            criterion = lambda output, label: torch.mul(output,label).mean(0).view(1)
        
        if self.unrolled_steps > 0:
            backup = copy.deepcopy(self.netD)
            for i in range(self.unrolled_steps):
                self.d_unrolled_loop(noise=noise, C=C, real_label=real_label, fake_label=fake_label, criterion=criterion, optimizerD=copy.deepcopy(self.optimizerD))
        

        fake = self.netG((noise,C))

        output = self.netD((fake,C)).view(-1)
        errG = criterion(output, label)

        # Calculate G's loss based on this output

        # Calculate gradients for G
        errG.backward()

        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        if self.unrolled_steps > 0:
            self.netD.load_state_dict(copy.deepcopy(backup.state_dict())) 
            del backup
        
        return errG, D_G_z2

    def train_loop(self,num_epochs,device,lr,d_steps, print_iterations, num_fixed_noises=10):
        
        

        num_anomalies = self.num_anomalies

        img_list = []
        evaluation_list = []
        fft_values = []
        eval_values = []
        G_losses = []
        D_losses = []
        iters = 0




        fixed_noise = torch.randn(num_fixed_noises, self.z_seq_length, self.z_latent_dim, device=device)
        
        self.fixed_noise = fixed_noise
        
        C = self.sample_C(num_fixed_noises)
        
        fixed_noise_label = C if self.conditional else None

        self.fixed_noise_label = fixed_noise_label

        # Initialize BCELoss function
        criterion = nn.BCELoss()
        #criterion = F.nll_loss

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = -1 if self.wgan else 0
        
        if self.optim == "RMSprop":
            optimizerD = torch.optim.RMSprop(self.netD.parameters(), lr=lr)
            optimizerG = torch.optim.RMSprop(self.netG.parameters(), lr=lr)
            
        else: 
            # Setup Adam optimizers for both G and D
            optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
            #optimizerD = optim.RMSprop(self.netD.parameters(), lr=d_lr)
            optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))

        self.optimizerD = optimizerD
        
        
        self.reference_metric = self.evaluation_pipeline(self.data[np.random.randint(0,self.data.shape[0],size=500),:,0], num_frequencies=self.num_frequencies)
        

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader

            print(f"Epoch {epoch}")
            for i in range((len(self.dataloader) // self.d_steps)+1):

                b_size = self.batch_size

                noise = torch.randn(b_size, self.z_seq_length, self.z_latent_dim, device=device)

                C = self.sample_C(b_size)

                
                for k in range(self.d_steps):

                    data = next(iter(self.dataloader))

                    errD, D_G_z1, D_x = self.train_discriminator(data, optimizerD, device, real_label, criterion, fake_label, noise, C if self.conditional else None)

                    iters += 1



                for k in range(self.g_steps):

                    errG, D_G_z2 = self.train_generator(real_label, fake_label, noise, C if self.conditional else None, criterion, optimizerG)

            if epoch % print_iterations == 0:

                self.log_metrics(fixed_noise, fixed_noise_label, epoch, num_epochs, errD, errG, D_x, D_G_z1, D_G_z2, img_list)
                if self.earlyStopping.early_stop:
                    print("Training Finished")
                    break
            
            



        self.img_list = np.transpose(img_list,(1,0,2,3))
        
        if self.save_path:
            self.save_configs(self.save_path)
        
        
    def save_configs(self,save_path):
        print("Saving Configs")
        
        self.model_config.pop('sampleC_fun', None)
    
        with open(f"{save_path}/model_config.json", 'w') as file:
         file.write(json.dumps(self.model_config)) 

        with open(f"{save_path}/training_data_config.json", 'w') as file:
         file.write(json.dumps(self.training_data_config)) 
        
        np.save(f"{save_path}/fixed_noise.npy",self.fixed_noise.detach().cpu().numpy())
        
        np.save(f"{save_path}/fixed_noise_label.npy",self.fixed_noise_label.detach().cpu().numpy())
        
        
    def sample_C(self,b_size):
        
        if self.sampleC_fun:
            
            return self.sampleC_fun(b_size).to(self.device)
        
        else:
        
            C = torch.zeros(size=(b_size,self.num_data_points)).to(self.device)
                                    # locations
            labels = torch.randint(0,self.num_data_points,(b_size,self.num_anomalies))

            C[torch.repeat_interleave(torch.arange(b_size).unsqueeze(dim=1), self.num_anomalies,dim=1), labels] = 1
        
        return C
        


    def log_metrics(self, fixed_noise, fixed_noise_label, epoch, num_epochs, errD, errG, D_x, D_G_z1, D_G_z2, img_list, num_frequencies=1):

        #print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        #                  % (epoch, num_epochs,
        #                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        logging_dict = {}
            
        #with torch.no_grad():

        generated = self.netG((fixed_noise.view(-1,self.z_seq_length,self.z_latent_dim),fixed_noise_label))
        
        fake = generated.detach().cpu()
        
        if self.anomaly_detector:
            f1, precision, recall, auc_score,mse = self.evaluate_with_anomaly_detector(generated, fixed_noise_label,self.anomaly_detector_window_size)
            
            self.earlyStopping(auc_score, self.netG, self.netD, minimize=False)
        
            
            logging_dict["f1"] = f1
            logging_dict["precision"] = precision
            logging_dict["recall"] = recall
            logging_dict["auc_score"] = auc_score
            logging_dict["mse"] = mse
            
        ### FFT Evaluation
        
        eval_dict = self.evaluation_pipeline(fake.numpy()[:,:,0],self.num_frequencies)
        
        ref_values = self.reference_metric["fft"]
        eval_values = eval_dict["fft"]
        
        ref_value = np.mean(ref_values,axis=0)
        eval_value = np.mean(eval_values,axis=0)
        
        difference = (ref_value-eval_value)**2
        
        logging_dict["FFT-Difference"] = np.sum(difference)          
                                       
        if self.use_wandb:
            fig = self.visualize_training_wandb(fake[:20])

            logging_dict["chart"] = wandb.Image(fig)

            plt.close(fig)

            wandb.log(logging_dict)
            
        print(logging_dict)

        img_list.append(fake.numpy())
                     
    def evaluation_pipeline(self,time_series, remove_0_bin=True, num_frequencies=1):
    
        return_dict = dict() 
        
        a, n = self.fft_evaluation(time_series.shape[1], 1/time_series.shape[1], time_series, remove_0_bin=True)
        
        eval_values = n[np.repeat(np.arange(n.shape[0]),num_frequencies).reshape(-1,num_frequencies),np.argsort(-n)[:,:num_frequencies]]
                     
        return_dict["fft"] = np.array(eval_values)

        return return_dict
                     
    
    
        
        
    def fft_evaluation(self,N,T,time_series, use_softmax=False, remove_0_bin=True):
        
    
        yf = fft(time_series)
        xf = fftfreq(N, T)[:N//2]

        abs_val = 2.0/N * np.abs(yf[:,0:N//2])

        if remove_0_bin:
            abs_val = np.concatenate((np.zeros(shape=[abs_val.shape[0],1]), abs_val[:,1:]),axis=1)

        if use_softmax:
            normalized = softmax(abs_val)
        else:
            summe = np.sum(abs_val,axis=1)

            indices = ~(summe == 0.0)

            summe = summe[indices]


            summe_array = np.stack([[summe[i]] * (N//2)  for i in range(summe.shape[0])],axis=0)

            normalized = abs_val[indices] / summe_array


        return abs_val[indices], normalized
            
    def visualize_training_wandb(self, fake):


        num_samples,num_data_points,num_multi_variates = fake.shape
        fig, axs = plt.subplots(num_samples,1,figsize=(20,2*num_samples),squeeze=False)
        
        colors = ["b", "g", "r"]
        
        for i in range(num_samples):
            for j,color in zip(range(self.multi_variates),colors):
                axs[i%num_samples][int(i/num_samples)].plot(fake[i,:,j],c=color)
                if self.cond_size > 1:
                
                    
                    axs[i%num_samples][int(i/num_samples)].scatter(np.where(self.fixed_noise_label[i,:,j].detach().cpu()==1)[0],np.ones(shape=[np.where(self.fixed_noise_label[i,:,j].detach().cpu() == 1)[0].shape[0]]), c=color)
                    
                
            if self.cond_size == 1:
                
                axs[i%num_samples][int(i/num_samples)].scatter(np.arange(0,num_data_points),self.fixed_noise_label[i].detach().cpu())

        return fig

    def visualize_training(self,mnist=False,num_samples=10):


        img_list = self.img_list

        fixed_noise_label = self.fixed_noise_label

        num_data_points = self.num_data_points


        fig, axs = plt.subplots(len(img_list),1,figsize=(20,2*num_samples),squeeze=False)


        xdata, ydata = [], []
        ln_list = [None for i in range(img_list.shape[0]*self.multi_variates)]


        if self.conditional:
            scatter_list = [axs[i%num_samples][int(i/num_samples)].scatter(np.arange(0,num_data_points),fixed_noise_label[i].detach().cpu())
                for i in range(num_samples)]
            """[
                axs[i%num_samples][int(i/num_samples)].scatter(np.where(fixed_noise_label.detach().cpu()==1)[1].reshape(num_samples,-1)[i],[0]*self.num_anomalies)
                for i in range(img_list.shape[0]) for j in range(self.multi_variates)
            ]"""
                
                
        
        ln_list = [
            axs[i%num_samples][int(i/num_samples)].plot(img_list[i,0,:,j])[0]
            for i in range(num_samples) for j in range(self.multi_variates)
        ]
        
        
        
        

        """for i,im in enumerate(img_list):
                ln_list[i] = axs[i%num_samples][int(i/num_samples)].plot(img_list[i,0,:,0])[0]"""
                #[axs[i%num_samples][int(i/num_samples)].plot(img_list[i,0,:,j])[0] for j in range(self.multi_variates)]

        def init():
            for i in range(num_samples):
                if self.conditional:
                    axs[i%num_samples][int(i/num_samples)].scatter(np.arange(0,num_data_points),fixed_noise_label[i].detach().cpu())
                axs[i%num_samples][int(i/num_samples)].set_xlim(0, num_data_points)
                    #axs[i%num_samples][int(i/num_samples)].set_ylim(0,1)
            return ln_list[0],

        def update(frame):
            #xdata.append(frame)
            #ydata.append(np.sin(frame))
            for index,ln in enumerate(ln_list):  

                i = index // self.multi_variates
                j = index % self.multi_variates
                #for j in range(self.multi_variates):
                ln.set_data(np.arange(0,num_data_points), img_list[i,frame,:,j])
                
                if self.conditional:
                    """
                    x = np.where(fixed_noise_label.detach().cpu()==1)[1].reshape(num_samples,-1)[i]
                    y = np.array([0]*self.num_anomalies)
                    data = np.hstack((x[:,np.newaxis], y[:, np.newaxis]))
                    scatter_list[index].set_offsets(data)
                    """
                #scatter_list[index].set_offsets(tmp)
                #axs[i%num_samples][int(i/num_samples)].plot(img_list[i,frame])
                #axs[i%num_samples][int(i/num_samples)].set_ylim(np.min(img_list[i][frame]), np.max(img_list[i][frame]))
            return ln_list[0],

        ani = FuncAnimation(fig, update, frames=np.arange(0,len(img_list[0])),
                            init_func=init, blit=True)



        plt.tight_layout()
        plt.close(fig)
        return HTML(ani.to_jshtml())
    
class GANEarlyStopping:
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
    def __call__(self, val_loss, generator, discriminator, minimize=True):

        
        score = -val_loss if minimize else val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, generator, discriminator)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, generator, discriminator)
            self.counter = 0

    def save_checkpoint(self, val_loss, generator, discriminator):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        if self.path:
            torch.save(generator.state_dict(), self.path + "/best_generator.pt")
            torch.save(discriminator.state_dict(), self.path + "/best_discriminator.pt")
        self.val_loss_min = val_loss    
        