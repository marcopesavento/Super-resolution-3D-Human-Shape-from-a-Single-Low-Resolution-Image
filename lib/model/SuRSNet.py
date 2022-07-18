'''
MIT License

Copyright (c) 2022 Marco Pesavento, Marco Volino and Adrian Hilton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


import torch
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
from .BaseSuRSNet import BaseSuRSNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
#from .PifuSR_v2 import *
from .SuRSSR_v3 import *
from .SurfaceClassifier import SurfaceClassifier


class SuRSNet(BaseSuRSNet):
    

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(SuRSNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion_rec = nn.L1Loss().to(self.device)
        self.criterion_BCE = nn.BCELoss().to(self.device)
        self.criterion_BCElog=nn.BCEWithLogitsLoss().to(self.device)

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter_lr = HGFilter(opt.num_stack_lr, opt.hg_depth, 256, opt.hg_dim, 
                                     opt.norm, 'low_res', False)
        self.image_filter_hr = HGFilter(opt.num_stack_hr, opt.hg_depth, 64, opt.hg_dim, 
                                     opt.norm, 'high_res', False)
        self.super_resolution=SuRSSR_v3(opt)

        self.mlp_lr = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim_lr,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            res_layers=self.opt.mlp_res_layers_lr,
            last_op=nn.Sigmoid())
        self.mlp_hr = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim_hr,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            res_layers=self.opt.mlp_res_layers_hr,
            last_op=nn.Sigmoid())


        
        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.im_SR = None
        self.feature_lr = None 
        self.feature_hr = None
        self.point_local_feature=None
        self.point_local_feature_hr=None
        self.intermediate_preds_list_lr = []
        self.intermediate_preds_list_hr = []
        self.z_feat=None    
        #self.in_img=None
        #self.tmpx_local_feature=None
        init_net(self)

    def filter_lr(self, images):
        '''
        Filter the input images
        store all intermediate features.
        '''
        
        self.im_feat_list_lr = self.image_filter_lr(images)
       
        if not self.training:
            self.im_feat_list_lr = [self.im_feat_list_lr[-1]]
    
    def filter_hr(self, images):
        '''
        Filter the input images
        store all intermediate features.
        
        '''
        
        self.im_feat_list_hr = self.image_filter_hr(images)
        
        if not self.training:
            self.im_feat_list_hr = [self.im_feat_list_hr[-1]]
        
    def super_res(self, images):
        
        
        self.im_SR, self.feature_lr, self.feature_hr = self.super_resolution(images)
       
        return self.im_SR, self.feature_lr, self.feature_hr

    def query_mr(self, points, calibs, transforms=None, labels=None):
       
       
        if labels is not None:
            
            self.labels_lr = labels
        xyz = self.projection(points, calibs, transforms)

        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        z_feat = self.normalizer(z, calibs=calibs)
        

        self.intermediate_preds_list_lr = []
        self.point_local_feat=[]
    
        for im_feat in range(0,len(self.im_feat_list_lr)): 
            
            point_local_feat_list = [self.index(self.im_feat_list_lr[im_feat], xy),self.index(self.im_feat_list_hr[0],xy)]#take points in the image
           
            point_local_feat_list = [torch.cat(point_local_feat_list, 1),z_feat]
            self.point_local_feat.append(torch.cat(point_local_feat_list, 1))
            
            pred = in_img[:,None].float() * self.mlp_lr(self.point_local_feat[im_feat]) 
            self.intermediate_preds_list_lr.append(pred)

        self.preds_lr = self.intermediate_preds_list_lr[-1]

    def query_sr(self, points, calibs, transforms=None, labels=None):
        
    
        if labels is not None:
            self.labels_hr = labels
        
        xyz = self.projection(points, calibs, transforms)
        
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        
        z_feat = self.normalizer(z, calibs=calibs)
        self.intermediate_preds_list_hr = []
        for im_feat in range(0,len(self.im_feat_list_lr)):
            point_local_feat_list = [self.index(self.im_feat_list_lr[im_feat], xy),self.index(self.im_feat_list_hr[0],xy)]#take points in the image
            
            point_local_feat_list = [torch.cat(point_local_feat_list, 1),z_feat]

            point_local_feat_list = [torch.cat(point_local_feat_list, 1),self.intermediate_preds_list_lr[im_feat]]
            self.point_local_feat_hr=torch.cat(point_local_feat_list, 1)

            pred = in_img[:,None].float() * self.mlp_hr(self.point_local_feat_hr)
            
            self.intermediate_preds_list_hr.append(pred)

        self.preds_hr = self.intermediate_preds_list_hr[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error_lr(self):
        '''
        '''
        error = 0
        for preds in self.intermediate_preds_list_lr:
            error += self.error_term(preds, self.labels_lr) 
        error /= len(self.intermediate_preds_list_lr)
        
        return error
    
    def get_error_hr(self):
        '''
        '''
        error = 0
        for preds in self.intermediate_preds_list_hr:
            error += self.error_term(preds, self.labels_hr)
        error /= len(self.intermediate_preds_list_hr)
        
        return error
    
    def get_errorSR(self,image_SR,images_hr):
        '''
        '''

        self.images_HR=images_hr
       
       
        error=self.criterion_rec(image_SR,images_hr)
        
        
        return error
    
    def get_error_disp_1(self):
        '''
        '''
       
        disp_gt=self.labels_hr-self.labels_lr
        disp_fake=self.intermediate_preds_list_hr[-1]-self.intermediate_preds_list_lr[-1]
        error = self.error_term(disp_gt, disp_fake)
        
        return error
    
    

    def forward(self, images_lr, images_hr, points_lr,points_hr, calibs, transforms=None, labels_lr=None, labels_hr=None):
        # Get image feature
       
        img_SR,feature_lr,feature_hr=self.super_res(images_lr)
       
        
        self.filter_lr(feature_lr)#128x128x(512/256)
        self.filter_hr(feature_hr)#512x512x (128/64)
        
        self.query_mr(points=points_hr, calibs=calibs, transforms=transforms, labels=labels_hr)
        self.query_sr(points=points_lr, calibs=calibs,transforms=transforms, labels=labels_lr)

        
        # get the prediction
        res_hr, res_lr = self.get_preds()

        # get the error
        error_mlp1 = self.get_error_lr()
        error_mlp2 = self.get_error_hr()
        error_SR = self.get_errorSR(img_SR,images_hr)
        disp=self.opt.disp_error
            
        error_disp=self.get_error_disp_1()
        

        error=self.opt.mlp1*error_mlp1+self.opt.mlp2*error_mlp2+self.opt.srweight*error_SR+self.opt.dispweight*error_disp
        return res_hr, error, res_lr
