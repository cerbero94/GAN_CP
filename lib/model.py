"""GAN for phase transition detection
"""
import os

from collections import OrderedDict
from lib.networks import AE, discriminator
from tqdm import tqdm
from lib.loss import l2_loss, l2_loss_batch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import numpy as np
from lib.visualizer import Visualizer
import wandb
import pandas as pd

class BaseModel():
    """ Base Model for phase transition detection
    """
    def __init__(self, opt, dataloader):
        
        # Initalize variables.
        self.opt = opt
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        # Initialize wandb logger
        ### to be changed
        if (self.opt.isTrain):
            wandb.init(project='gan', entity='cerbero94', config=opt)

    def set_input(self, input:torch.Tensor):
        """ Set input

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input = input
    
    ### TODO: I think this should go in the specific model, or better it should only report self.losses which is an ordered dictionary
    ### containing all the losses below.
    ##
    def get_losses(self):
        """ Reports the losses of the model.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        return self.losses

    ##
    def get_img(self):
        """ Returns current img (input and reconstruted features).

        Returns:
            [reals, fakes]
        """

        reals = self.input.data
        fakes = self.fake.data

        return reals, fakes

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pt' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pt' % (weight_dir))

    ##
    def load_weights(self):
        """Load netG and netD weights.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights','netG.pt')
        # if not os.path.exists(weight_dir): os.makedirs(weight_dir)
        pretrained_dict = torch.load(weight_dir)#['state_dict']
        
        try:
            self.netg.load_state_dict(pretrained_dict)
        except IOError:
            raise IOError("netG weights not found")
        print('   Loaded weights.')

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0

        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize
            
            batch_input_features = torch.stack([v["es"] for v in data])

            self.set_input(batch_input_features)
            self.optimize_params()

            # if self.total_steps % self.opt.print_freq == 0:
                
            # if self.total_steps % self.opt.save_image_freq == 0:
            #     reals, fakes, fixed = self.get_current_images()
            #     self.visualizer.save_current_images(self.epoch, reals, fakes)
            #     if self.opt.display:
            #         self.visualizer.display_current_images(reals, fakes)
        
        # Update the scheduler
        self.scheduler_G.step()
        self.scheduler_D.step()
        # Validate each epoch
        self.validate()

        if self.opt.display:
            errors = self.get_losses()
            self.losses.update({'lr_G':self.optimizer_G.state_dict()['param_groups'][0]['lr'],\
                                'lr_D':self.optimizer_D.state_dict()['param_groups'][0]['lr']})
            wandb.log(errors)
        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0
        
        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        # Start the logger
        wandb.run

        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()

            if(self.losses['loss_G_val']<self.opt.val_thr and self.losses['loss_G_G_tr']<self.opt.tr_thr):
                print('>> Target thresholds for losses achieved:')
                print('Training reconstruction loss : %1.5f\tValidation reconstruction loss : %1.5f'\
                    %(self.losses['loss_G_G_tr'],self.losses['loss_G_val']))
                # self.test()
                break
        self.save_weights(self.epoch)
        self.test()        
        print(f">> Training model {self.name}.[Done]")
    
    def validate(self):
        """ Evaluate GAN model on validation set.
        """
        X_val = torch.stack([v["es"] for v in next(iter(self.dataloader['validation']))])
        with torch.no_grad():
            X_hat_val = self.netg(X_val)
            val_loss = l2_loss(X_val,X_hat_val)

        self.losses.update({'loss_G_val':val_loss.item()})

    def test(self):
        """ Evaluate GAN model on the whole test dataset.
        """
        
        self.opt.phase = 'test'

        if(not self.opt.isTrain):
            self.load_weights()
        
        ### TODO: this depends on the model actually!!! For models with 2 control variables it does not work. 
        ### maybe must migrate in the specific model and not the GAN generic one
        import matplotlib.pyplot as plt
        X_test = torch.stack([v["es"] for v in next(iter(self.dataloader['test']))])
        X_train = torch.stack([v["es"] for v in next(iter(self.dataloader['train']))])
        X_validation = torch.stack([v["es"] for v in next(iter(self.dataloader['validation']))])
        # print(X_test.size())
        # print(X_train.size())
        # print(len([v["parameter"] for v in next(iter(self.dataloader['test']))]))
        with torch.no_grad():
            X_hat_test = self.netg(X_test)
            X_hat_train = self.netg(X_train)
            X_hat_validation = self.netg(X_validation)
            # TODO: not exactly right: it should be the error on the whole training set and not on a random batch
            test_loss = 100*l2_loss_batch(X_test,X_hat_test)
            noise_loss = 100*l2_loss_batch(X_train,X_hat_train)
            noise_loss_mean = np.mean(noise_loss)
            noise_loss_sum = np.sum(noise_loss)
            loss_validation_sum = np.sum(100*l2_loss_batch(X_validation,X_hat_validation))
        print(f'>> Training loss: {noise_loss_sum} %\t Validation loss: {loss_validation_sum} %')
        parameter = [v["parameter"] for v in next(iter(self.dataloader['test']))]

        # if(self.opt.isTrain):
            # export anomaly detection results
        plt.figure(figsize=(8,6))
        plt.xticks(fontsize=21)
        plt.yticks(fontsize=21)
        plt.xlabel('parameter',fontsize=22)
        plt.plot(parameter,test_loss-noise_loss_mean,'o-',ms=4,color='#E57439')
        plt.grid()
        plt.axvspan(min(self.opt.training_reg), max(self.opt.training_reg), facecolor='yellow', alpha=0.15)
        plt.axvspan(min(self.opt.validation_reg), min(self.opt.validation_reg), facecolor='green', alpha=0.15)

        plot_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        plt.savefig('%s/loss_anomaly_detection.png' % (plot_dir),dpi=300)

        # export the .csv file containing the curve of the results
        performance = pd.DataFrame(data={'parameter':parameter,'loss-noise':test_loss-noise_loss_mean})
        performance.to_csv('%s/loss_anomaly_detection.csv' % (plot_dir))
        print(f">> Evaluation of model {self.opt.name}.[Done]")
        print(f">> results exported to {plot_dir}/loss_anomaly_detection.csv")
        # else:
        #     return parameter, test_loss-noise_loss
   
        ### TODO: export a summary of the performances as in wandb (maybe taking from the history)

##
class GAN_ES_1D(BaseModel):
    """GAN class for critical points detection using 1D bipartite ES 
    """

    @property
    def name(self): return 'GAN_ES_1D'

    def __init__(self, opt, dataloader):
        super(GAN_ES_1D, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = AE(self.opt).to(self.device)
        self.netd = discriminator(self.opt).to(self.device)
        
        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pt'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pt'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pt'))['state_dict'])
            print("\tDone.\n")

        self.l_rec = l2_loss
        self.l_adv = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            ### TODO change the frequences according to opt
            wandb.watch(self.netg, log_freq=50)
            wandb.watch(self.netd, log_freq=50)
            # models in training mode
            self.netg.train()
            self.netd.train()
            # definition of optimizers
            self.optimizer_G = optim.Adam(self.netg.parameters(), lr=self.opt.G_lr)
            self.optimizer_D = optim.Adam(self.netd.parameters(), lr=self.opt.D_lr)
            # definition of the schedulers
            self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, self.opt.niter, eta_min=0, last_epoch=-1, verbose=False)
            self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, self.opt.niter, eta_min=0, last_epoch=-1, verbose=False)

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.reconstruction = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real = self.netd(self.input)
        self.pred_fake = self.netd(self.reconstruction.detach())

    ### TODO: check if in this phase the adv loss can be used as l2 for the labels:
    ### in this case, the aim of the discriminator is to minimize l2(netD(input),netD(reconstruction))
    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        real = torch.ones (size=(self.input.size(0),1), dtype=torch.float32, device=self.device)
        self.D_G_z2 = self.netd(self.reconstruction).mean().item()
        self.err_g_adv = self.l_adv(self.netd(self.reconstruction), real)
        self.err_g_rec = self.l_rec(self.input, self.reconstruction)
        self.errG_D = self.err_g_adv * self.opt.Lambda
        self.errG_G = self.err_g_rec * self.opt.Epsilon
        self.err_g = self.errG_D + \
                     self.errG_G 
        self.err_g.backward(retain_graph=True)
         

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss & Backward-Pass
        real = torch.ones (size=(self.input.size(0),1), dtype=torch.float32, device=self.device)
        fake = torch.zeros (size=(self.input.size(0),1), dtype=torch.float32, device=self.device)

        self.err_d_real = self.l_adv(self.pred_real, real)
        self.err_d_real.backward()
        self.err_d_fake = self.l_adv(self.pred_fake, fake)
        self.err_d_fake.backward()
        self.errD = self.err_d_real + self.err_d_fake
        self.D_x = self.pred_real.mean().item()
        self.D_G_z1 = self.pred_fake.mean().item()

    def update_losses(self):
        self.losses = OrderedDict([
            ('loss_D',self.errD.item()), 
            ('loss_G_D_tr',self.errG_D.item()), 
            ('loss_G_G_tr',self.errG_G.item()),
            ('D_x',self.D_x), 
            ('D_G_z1',self.D_G_z1), 
            ('D_G_z2',self.D_G_z2)])
    
    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        self.optimizer_D.zero_grad()
        self.backward_d()
        self.optimizer_D.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.optimizer_G.zero_grad()
        self.backward_g()
        self.optimizer_G.step()

        # Update the losses every step
        self.update_losses()