""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset_type', default='ES_1D', help='type of dataset (e.g. one-dimensional ES)')
        self.parser.add_argument('--dataset', default='BH_32_BKT', help='dataset in ./data folder')
        self.parser.add_argument('--batchsize', type=int, default=24, help='input batch size')
        self.parser.add_argument('--isize', type=int, default=64, help='input ES size.')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--device', type=str, default='cpu', help='Device: gpu | cpu')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='gan_es_1D', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--display', action='store_true', help='Show the final loss profile')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--wandb_account', default='your_wandb_account', help='see https://docs.wandb.ai')
        self.parser.add_argument('--wandb_proj', default='your_wandb_project', help='see https://docs.wandb.ai')

        ##
        # Train
        self.parser.add_argument('--training_reg',nargs=2 ,type=float, default=None, help='bounds of the training region in the parameter space (use 2 floats separated with a space and no quotes)')
        self.parser.add_argument('--validation_reg',nargs=2 ,type=float, default=None, help='frequency of showing training results on console (use 2 floats separated with a space and no quotes)')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, validation, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=250, help='number of epochs to train for')
        self.parser.add_argument('--G_lr', type=float, default=3e-3, help='initial learning rate of Generator for adam')
        self.parser.add_argument('--D_lr', type=float, default=3e-2, help='initial learning rate of Discriminator for adam')
        self.parser.add_argument('--Lambda', type=float, default=0.1, help='Adversarial loss weight')
        self.parser.add_argument('--Epsilon', type=float, default=10, help='Reconstruction loss weight')
        self.parser.add_argument('--val_thr', type=float, default=1.5e-2, help='Target threshold for reconstruction validation loss')
        self.parser.add_argument('--tr_thr', type=float, default=1e-3, help='Target threshold for reconstruction training loss')
        self.isTrain = False
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        # str_ids = self.opt.gpu_ids.split(',')
        # self.opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         self.opt.gpu_ids.append(id)

        # set gpu ids
        # if self.opt.device == 'gpu':
        #     torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        else:
            self.opt.name = "%s/%s" % (self.opt.name, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
