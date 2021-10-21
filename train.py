"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
#### MINE BELOW
from torch.utils.data import DataLoader, random_split, Dataset
from lib.functions_classes_torch import *

from lib.data import load_data
from lib.model import GAN_ES_1D

##
def train():
    """ Training
    """
    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader = load_data(opt)

    ##
    # LOAD MODEL
    model = GAN_ES_1D(opt, dataloader)
    ##
    # TRAIN MODEL
    model.train()

# ### TODO
# def evaluate():
#     """ Evaluation of already trained models
#     """
#     ##
#     # ARGUMENTS
#     opt = Options().parse()
#     # LOAD MODEL
#     model = GAN_ES_1D(opt, dataloader_train_32)
#     # TRAIN MODEL
#     model.evaluate()


if __name__ == '__main__':
    train()
