# import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
import wandb
import numpy as np




def preprocess_df_for_training_torch(df,tr_range,**kwargs):
    train_df = df.loc[(df['parameter']<max(tr_range))&(df['parameter']>min(tr_range))]
    from sklearn.model_selection import train_test_split
    if(kwargs.get('test_size')):
        size = kwargs['test_size']
    else:
        size = 0.2
    
    train, validation = train_test_split(train_df, test_size=size, shuffle=True)
   
    return  CustomESDataset(train),\
            CustomESDataset(validation)

def preprocess_df_for_training_torch_bound_validation(df,tr_range,**kwargs):
    train_df = df.loc[(df['parameter']<max(tr_range))&(df['parameter']>min(tr_range))]
    
    if(kwargs.get('reversed')):
        train_df.sort_values('parameter',inplace=True,ascending=False)
    else:
        train_df.sort_values('parameter',inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    # from sklearn.model_selection import train_test_split
    if(kwargs.get('test_size')):
        size = kwargs['test_size']
    else:
        size = 0.2
    
    n_test_samples = int(size * len(train_df['parameter']))
    validation = train_df.iloc[-n_test_samples:]
    train = train_df.iloc[0:len(train_df['parameter'])-n_test_samples]
    
    return  CustomESDataset(train),\
            CustomESDataset(validation)

class CustomESDataset(Dataset):
    ''' Class for the dataset:
        the initialization divides the input DF in .features, .labels and .parameter, 
        the __getitem__ method returns a dictionary for a given index
    '''
    def __init__(self, df):
        self.features = torch.FloatTensor(df.drop(['parameter'],axis=1).values)
        self.parameter = df['parameter']

    def __len__(self):
        return len(self.parameter)

    def __getitem__(self, idx):
        es_eigenvalues = self.features[idx]
        param_value = self.parameter.iloc[idx]

        return {"es":es_eigenvalues,"parameter":param_value}

def custom_collate(dictionary):
    return dictionary



def train_loss2(x,x_hat):
    return torch.sqrt(torch.sum((x - x_hat)**2))

# def train_loss2_log(x,x_hat):
#     return torch.sqrt(torch.sum((-torch.log10(x**2) + torch.log10(x_hat**2))**2))

def log_cosh_loss(x,x_hat):
        ey_t = x - x_hat
        return torch.sum(torch.log(torch.cosh(ey_t + 1e-12)))

def compute_batch_reconstruction_loss(x,xhat):
    loss = nn.MSELoss(reduction='sum')
    losses_array = []
    for i in range(len(x)):
        losses_array.append(np.sqrt(loss(x[i,:],xhat[i,:]).item()))
    return np.array(losses_array)

def training(epochs:int, dataloader_tr, dataloader_val, model, loss_fn, optimizer, scheduler, device):
    ### Initialize wandb
    # wandb.init(project='ae', entity='cerbero94')
    # wandb.run
    # wandb.watch(model, log_freq=5)

    size = len(dataloader_tr.dataset)
    training_loss = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        counter = 0
        for batch, dictionary in enumerate(dataloader_tr):
            # print(batch,dictionary)
            features = torch.stack([v["es"] for v in dictionary])
            X = features.to(device)

            # Compute prediction error
            Xhat = model(X)
            loss = loss_fn(X, Xhat)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch % 5 == 0:
            counter += len(X)
            loss, current = loss.item(), counter
            epoch_loss += loss
            training_loss += loss
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # scheduler.step()
        
        epoch_loss /= len(dataloader_tr)
        X_val = torch.stack([v["es"] for v in next(iter(dataloader_val))])
        X_hat_val = model(X_val)
        validation_loss = loss_fn(X_val,X_hat_val)

        scheduler.step()

        lr_current = optimizer.state_dict()['param_groups'][0]['lr']
        # wandb.log({'tr_loss':epoch_loss, 'val_loss':validation_loss})
        print(f'EPOCH {epoch};\tloss: {epoch_loss};\tval_loss: {validation_loss};\tlr: {lr_current}')
    training_loss /= size   

def training_gan(epochs:int, dataloader_tr, dataloader_val, netD, netG, loss1, loss2, 
                optD, optG, scheduler_G, scheduler_D, Lambda=0.05, Epsilon=0.9,device='cpu',**kwargs):
    
    std_config={'lambda':Lambda, 'epsilon':Epsilon, \
                'lrD':[param['lr'] for param in optD.param_groups][0],\
                'lrG':[param['lr'] for param in optG.param_groups][0]}
    if kwargs.get('config'):
        wandb_additional_config = kwargs['config']
        complete_config = std_config | wandb_additional_config
    else: complete_config = std_config
    ### Initialize wandb
    wandb.init(project='gan', entity='cerbero94', config=complete_config)
    wandb.run
    wandb.watch(netD, log_freq=50)
    wandb.watch(netG, log_freq=50)
    
    real_label = 1.
    fake_label = 0.
    iters = 0
    print(f"Starting Training Loop with Lambda = {Lambda} and Epsilon = {Epsilon} ...")
    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader_tr):
            # Format batch
            features = torch.stack([v["es"] for v in data])
            X = features.to(device)
            b_size = X.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            Xhat = netG(X)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Forward pass real batch through D
            output = netD(X).view(-1)
            # fill real labels
            label.fill_(real_label)
            # Calculate loss on all-real batch
            errD_real = loss1(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake image batch with G
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(Xhat.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = loss1(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = (errD_real + errD_fake)*Lambda
            # Update D
            optD.step()
                
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # fake labels are real for generator cost
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(Xhat).view(-1)
            # Calculate G's loss based on this output
            errG_D = Lambda*loss1(output, label)
            errG_G = (Epsilon)*loss2(X,Xhat) 
            # print(errG_G)
            errG = errG_D + errG_G
            # Calculate gradients for G
            netG.zero_grad()
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optG.step()
            
            # Output training stats
            # if i % 17 == 0:
            #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #             % (epoch, epochs, i, len(dataloader_train),
            #                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            iters += 1
        
        X_val = torch.stack([v["es"] for v in next(iter(dataloader_val))])
        X_hat_val = netG(X_val)
        validation_loss = loss2(X_val,X_hat_val)
        wandb.log({'loss_D':errD.item(), 'loss_G_D_tr':errG_D.item(), 'loss_G_G_tr':errG_G.item(), \
                   'loss_G_val':validation_loss, 'D_x':D_x, 'D_G_z1':D_G_z1, 'D_G_z2':D_G_z2})
        lr_current_G = optG.state_dict()['param_groups'][0]['lr']
        lr_current_D = optD.state_dict()['param_groups'][0]['lr']
        # print(f'epoch: {epoch},\tlr_G: {lr_current_G},\tlr_D: {lr_current_D}')
        print(f'epoch: {epoch},\tval_loss: {validation_loss},\ttr_loss: {errG_G.item()}')
        # scheduler_D.step()
        scheduler_G.step()
        
        ### some thresholds can be fixed on the validation and the training set
        ### so the training is stopped below them
        if kwargs.get('br_loss'):
            br_loss = kwargs['br_loss']
        else:
            br_loss = 1e-6
        
        if kwargs.get('br_loss_train'):
            br_loss_train = kwargs['br_loss_train']
        else:
            br_loss_train = 1e-6

        if(validation_loss<br_loss and errG_G.item()<br_loss_train):
            print(f'Training broken because of threshold val_loss = {br_loss}')
            break
        
    print("Finished Training Loop...")

