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
from torch.utils.data import DataLoader, random_split, Dataset
from lib.functions_classes_torch import *

from lib.data import load_data
from lib.model import GAN_ES_1D 

##
def evaluate():
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
    # TEST MODEL
    return model.test()

if __name__ == '__main__':
    evaluate()