import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index
import gc
# get options
opt = BaseOptions().parse()

def eval(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    test_dataset = EvalDataset_LR_v2(opt, phase='test')
   
    

    projection_mode = test_dataset.projection_mode

   
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create net
    netG = SuRSNet(opt, projection_mode).to(device=cuda)
    
    print('Using Network: ', netG.name)
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
     
    with torch.no_grad():
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))
        set_eval()
        test_dataset.is_train = False


           
                
        if not opt.no_gen_mesh:
            
            print('generate mesh (test) ...')
            for gen_idx in tqdm(range(len(test_dataset))):
                #print(len(test_dataset))
                test_data = test_dataset[gen_idx]
                #print(test_data['name'][0])
                save_path = '%s/%s/%s.obj' % (
                    opt.results_path, opt.name,  test_data['name'][0])
                gen_mesh(opt, netG, cuda, test_data, save_path)
            



if __name__ == '__main__':
    eval(opt)
