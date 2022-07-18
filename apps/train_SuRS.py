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

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = TrainDataset_LR_v2(opt, phase='train')
    test_dataset = TrainDataset_LR_v2(opt, phase='test')
    
    

    projection_mode = train_dataset.projection_mode

    # create data loader

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create net
    netG = SuRSNet(opt, projection_mode).to(device=cuda)
    if opt.optimizer == 'SGD':
        optimizerG = torch.optim.SGD(netG.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
        #kwargs = {'momentum': args.momentum}
    elif opt.optimizer == 'ADAM':
        optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate, betas=(opt.beta1, opt.beta2),eps=opt.epsilon, weight_decay=opt.weight_decay)
        #kwargs = {
        ##    'betas': (args.beta1, args.beta2),
        #    'eps': args.epsilon
        #}
    elif opt.optimizer == 'RMSprop':
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=opt.weight_decay)
        #kwargs = {'eps': args.epsilon}
    elif opt.optimizer == 'AMSgrad':
        optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate, betas=(opt.beta1, opt.beta2),eps=opt.epsilon, weight_decay=opt.weight_decay,amsgrad=True)
        #kwargs = {
        #    'betas': (args.beta1, args.beta2),
        #    'eps': args.epsilon
            #'amsgrad'=args.ams
        #}
    lr = opt.learning_rate
    print('Using Network: ', netG.name)
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    if opt.continue_train==0:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        netG.load_state_dict(torch.load(model_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    weight_sr=opt.srweight
    weight_mlp1=opt.mlp1
    weight_mlp2=opt.mlp2
    weight_disp=opt.dispweight
    

    # training/fin qua tutto uguale
    if opt.continue_train!=0:
        start_epoch = 0
    else: 
        start_epoch=max(opt.resume_epoch,0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()
        gc.collect()
        set_train()
        iter_data_time = time.time()

        

        



        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve the data
            image_tensor_lr = train_data['img_LR'].to(device=cuda)
            image_tensor_hr = train_data['img_HR'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor_lr = train_data['samples_LR'].to(device=cuda)
            sample_tensor_hr = train_data['samples_HR'].to(device=cuda)
            #print(image_tensor_lr.shape, image_tensor_hr.shape)

            #print(image_tensor_lr.shape, image_tensor_hr.shape)
            image_tensor_hr,image_tensor_lr, calib_tensor = reshape_multiview_tensors(image_tensor_hr,image_tensor_lr, calib_tensor)
            #print(image_tensor_lr.shape, image_tensor_hr.shape)

            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)

            label_tensor_hr = train_data['labels_HR'].to(device=cuda)
            label_tensor_lr = train_data['labels_disp'].to(device=cuda)
            #arrivato qua a modificare
            res_hr, error,res_lr = netG.forward(image_tensor_lr,image_tensor_hr, sample_tensor_lr,sample_tensor_hr, calib_tensor,labels_lr=label_tensor_lr, labels_hr=label_tensor_hr)
            #error=0
            #error=weight_sr*error_SR+weight_mlp1*error_mlp1+weight_mlp2*error_mlp2+error_disp*weight_disp
            optimizerG.zero_grad()
            error.backward()
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            #if train_idx % opt.freq_plot == 0:
                #print(
                #    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | Err_SR: {5:.06f} | Err_mlp1: {6:.06f} | Err_mlp2: {7:.06f}| Err_disp: {8:.06f}  | LR: {9:.06f} | Sigma: {10:.02f} | dataT: {11:.05f} | netT: {12:.05f} | ETA: {13:02d}:{14:02d}'.format(
                 #       opt.name, epoch, train_idx, len(train_data_loader), error.item(), error_SR.item(), error_mlp1.item(), error_mlp2.item(), error_disp.item(), lr, opt.sigma,
                 #                                                           iter_start_time - iter_data_time,
                #                                                            iter_net_time - iter_start_time, int(eta // 60),
                 #       int(eta - 60 * (eta // 60))))

            if train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            if train_idx % opt.freq_save_ply == 0:
                save_path = '%s/%s/%spred.ply' % (opt.results_path, opt.name,epoch)
                r = res_hr[0].cpu()
                
                points = sample_tensor_hr[0].transpose(0, 1).cpu()
                print('points',sample_tensor_hr[0].shape)
                save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())
                save_path2 = '%s/%s/%spred_gt.ply' % (opt.results_path, opt.name,epoch)
                #print(label_tensor_hr.shape,sample_tensor[0].shape)

                r2 = label_tensor_hr[0].cpu()
                #print("ok")
                points2 = sample_tensor_hr[0].transpose(0, 1).cpu()
                save_samples_truncted_prob(save_path2, points2.detach().numpy(), r2.detach().numpy())
                #print("ok")
                save_path3 = '%s/%s/%spred_lr.ply' % (opt.results_path, opt.name,epoch)
                r3 = label_tensor_lr[0].cpu()
                points3 = sample_tensor_lr[0].transpose(0, 1).cpu()
                save_samples_truncted_prob(save_path3, points3.detach().numpy(), r3.detach().numpy())
            
            iter_data_time = time.time()

            gc.collect()
        torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)
       
        #### test
        with torch.no_grad():
            set_eval()

            
                
                
            if not opt.no_gen_mesh:
                ev=0
                print('generate mesh (test) ...')
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    test_data = test_dataset[ev]
                    #print(test_data['name'])
                    save_path = '%s/%s/test_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, test_data['name'])
                    gen_mesh(opt, netG, cuda, test_data, save_path) #da modificare
                    ev+=1

                print('generate mesh (train) ...')
                train_dataset.is_train = False
                ev2=0
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    train_data = train_dataset[ev2]
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])
                    gen_mesh(opt, netG, cuda, train_data, save_path)
                    ev2+=1
                train_dataset.is_train = True


if __name__ == '__main__':
    train(opt)
