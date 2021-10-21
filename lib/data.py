"""
LOAD DATA from file.
"""

##
import os
import torch
import numpy as np
import pandas as pd

##
class ES_1D_single_param_Dataset(torch.utils.data.Dataset):
    ''' Class for the dataset:
        the initialization divides the input DF in .features and .parameter, 
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

##
def custom_collate(dictionary):
    return dictionary

##
def load_data(opt):
    if opt.dataset_type in ['ES_1D']:
        splits = ['train','validation','test']
        shuffle = {'train':True,'validation':False,'test':False}
        dataset = {}
        
        df = pd.read_csv(f'./data/{opt.dataset}.csv')
        df = df.iloc[:,:opt.isize+1]

        dataset['train'] = ES_1D_single_param_Dataset( df.loc[(df['parameter']<max(opt.training_reg))&(df['parameter']>min(opt.training_reg))] )
        dataset['validation'] = ES_1D_single_param_Dataset( df.loc[(df['parameter']<max(opt.validation_reg))&(df['parameter']>min(opt.validation_reg))] )
        dataset['test'] = ES_1D_single_param_Dataset( df )
        b_size = {'train':opt.batchsize,'validation':len(dataset['validation']),'test':len(dataset['test'])}

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=b_size[x],
                                                     shuffle=shuffle[x],
                                                     collate_fn=custom_collate)
                                                     #num_workers=int(opt.workers),
                                                     #drop_last=drop_last_batch[x],
                                                     #worker_init_fn=(None if opt.manualseed == -1
                                                     #else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
