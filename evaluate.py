"""
EVALUATE 

. Example: Run the following command from the terminal.
    
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from torch.utils.data import DataLoader, random_split, Dataset

from lib.data import load_data
from lib.model import GAN_ES_1D 

##
def evaluate():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    opt.isTrain = False

    ##
    # LOAD DATA
    dataloader = load_data(opt)

    ##
    # LOAD MODEL
    model = GAN_ES_1D(opt, dataloader)
    model.load_weights()

    ##
    # TEST MODEL
    return model.test()

if __name__ == '__main__':
    evaluate()