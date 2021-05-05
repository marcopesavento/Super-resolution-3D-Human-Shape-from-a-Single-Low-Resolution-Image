import torch
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
#from .PifuSR_v2 import *
from .PifuSR_v3 import *


class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hgpifu'
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion_rec = nn.L1Loss().to(self.device)

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter_lr = HGFilter(opt.num_stack_lr, opt.hg_depth, 256, opt.hg_dim, 
                                     opt.norm, 'low_res', False)
        self.image_filter_hr = HGFilter(opt.num_stack_hr, opt.hg_depth, 64, opt.hg_dim, 
                                     opt.norm, 'high_res', False)
        self.super_resolution=PifuSR_v3(opt)

        self.mlp_lr = MLP(
            filter_channels=self.opt.mlp_dim_lr,
            num_views=self.num_views,
            res_layers=self.opt.mlp_res_layers_lr,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid())
        self.mlp_hr = MLP(
            filter_channels=self.opt.mlp_dim_hr,
            num_views=self.num_views,
            res_layers=self.opt.mlp_res_layers_hr,
            norm=self.opt.mlp_norm,
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
        self.in_img=None
        self.tmpx_local_feature=None
        init_net(self)

    def filter_lr(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        ##print(images.shape)
        self.im_feat_list_lr, self.tmpx_lr, self.normx_lr = self.image_filter_lr(images)
        ##print(self.im_feat_list[0].shape) # 4x 256,128,128 4 e' il numero di stack della hourglass network
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list_lr = [self.im_feat_list_lr[-1]]
    
    def filter_hr(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        ##print(images.shape)
        self.im_feat_list_hr, self.tmpx_hr, self.normx_hr = self.image_filter_hr(images)
        ##print(self.im_feat_list[0].shape) # 4x 256,128,128 4 e' il numero di stack della hourglass network
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list_hr = [self.im_feat_list_hr[-1]]
        
    def super_res(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        ##print(images.shape)
        self.im_SR, self.feature_lr, self.feature_hr = self.super_resolution(images)
        ##print(self.im_feat_list[0].shape) # 4x 256,128,128 4 e' il numero di stack della hourglass network
        # If it is not in training, only produce the last im_feat
        #if not self.training:
        #    self.im_feat_list = [self.im_feat_list[-1]]
        return self.im_SR, self.feature_lr, self.feature_hr

    def query_lr(self, points, calibs, transforms=None, labels_lr=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels_lr is not None:
            ##print("ok")
            #print(labels_lr.shape)
            self.labels_lr = labels_lr
        ###print(calibs) #calibs e' ok ma da dove vengono fuori i punti?????
        xyz = self.projection(points, calibs, transforms)
        
        xy = xyz[:, :2, :]
        ##print(xy)
        z = xyz[:, 2:3, :]
        #print(xyz.shape)

        self.in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        #print(self.in_img.shape)
        self.z_feat = self.normalizer(z, calibs=calibs)
        #print(self.z_feat.shape)
        if self.opt.skip_hourglass:
            #rint("ok")
            tmpx_local_feature_lr = self.index(self.tmpx_lr, xy)
            tmpx_local_feature_hr = self.index(self.tmpx_hr, xy)
            self.tmpx_local_feature=torch.cat((tmpx_local_feature_lr,tmpx_local_feature_hr),1)


        self.intermediate_preds_list_lr = []
        self.point_local_feat=[]
        ##print(len(self.im_feat_list_hr))
        ##print(len(self.im_feat_list))#e' qua che devo cambiare perche voglio prenderlo da ambedue e concatenarli
        for im_feat in range(0,len(self.im_feat_list_lr)): #no perche per l'hr ne ho solo uno
            # [B, Feat_i + z, N]

            point_local_feat_list = [self.index(self.im_feat_list_lr[im_feat], xy),self.index(self.im_feat_list_hr[0],xy)]#take points in the image
            #point_local_feat_list_hr = [, xy), z_feat]#take points in the image

            if self.opt.skip_hourglass:
                point_local_feat_list.append(self.tmpx_local_feature)

            point_local_feat_list = [torch.cat(point_local_feat_list, 1),self.z_feat]
            self.point_local_feat.append(torch.cat(point_local_feat_list, 1))
            #print(self.point_local_feat.shape)

            # out of image plane is always set to 0
            pred = self.in_img[:,None].float() * self.mlp_lr(self.point_local_feat[im_feat]) #non so se va lo [0]!!!!!!
            #torch.set_#printoptions(threshold=10000)
            ##print(pred)
            self.intermediate_preds_list_lr.append(pred)

        self.preds_lr = self.intermediate_preds_list_lr[-1] #qua da capire perche la faccio solo all'ultimo e non a tutti, potrei fare la hr con tutti e non solo con l'ultimo alla fine

    def query_hr(self, transforms=None, labels_hr=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels_hr is not None:
            self.labels_hr = labels_hr
        ###print(calibs) #calibs e' ok ma da dove vengono fuori i punti?????

        ##print(z_feat.shape)
        #if self.opt.skip_hourglass:
            ##print("ok")
            #tmpx_local_feature_lr = self.index(self.tmpx_lr, xy)
            #tmpx_local_feature_hr = self.index(self.tmpx_hr, xy)
            #tmpx_local_feature_hr=torch.cat((self.tmpx_local_feature_lr,tmpx_local_feature_hr),1)


        self.intermediate_preds_list_hr = []
        ##print(len(self.im_feat_list))#e' qua che devo cambiare perche voglio prenderlo da ambedue e concatenarli
        for im_feat in range(0,len(self.im_feat_list_lr)):
            # [B, Feat_i + z, N]

            #point_local_feat_list = [self.index(self.im_feat_list_lr[im_feat], xy),self.index(self.im_feat_list_hr[im_feat],xy)]#take points in the image
            #point_local_feat_list_hr = [, xy), z_feat]#take points in the image

            if self.opt.skip_hourglass:
                point_local_feat_list=[]
                point_local_feat_list.append(self.tmpx_local_feature)

            point_local_feat_list = [self.point_local_feat[im_feat],self.intermediate_preds_list_lr[im_feat]]
            self.point_local_feat_hr=torch.cat(point_local_feat_list, 1)
            #print(self.point_local_feat.shape)

            # out of image plane is always set to 0
            pred = self.in_img[:,None].float() * self.mlp_hr(self.point_local_feat_hr)#non so se va lo zero
            #torch.set_#printoptions(threshold=10000)
            ##print(pred)
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
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list_lr:
            error += self.error_term(preds, self.labels_lr) #cambio qua
        error /= len(self.intermediate_preds_list_lr)
        
        return error
    
    def get_error_hr(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list_hr:
            error += self.error_term(preds, self.labels_hr)
        error /= len(self.intermediate_preds_list_hr)
        
        return error
    
    def get_errorSR(self,image_SR,images_hr):
        '''
        Hourglass has its own intermediate supervision scheme
        '''

        self.images_HR=images_hr
       
        #error=self.error_term(self.im_SR,self.images_HR)
        #print("img_SR",self.im_SR)
        #print("img_HR",images_hr)
        error=self.criterion_rec(image_SR,images_hr)
        #for preds in self.intermediate_preds_list:
         #   error += self.error_term(preds, self.labels)
        #error /= len(self.intermediate_preds_list)
        
        return error

    def forward(self, images_lr, images_hr, points, calibs, transforms=None, labels_lr=None, labels_hr=None):
        # Get image feature
        ##print(images.shape) modifico aggiungendo la sr network
        #self.images_hr=images_hr
        img_SR,feature_lr,feature_hr=self.super_res(images_lr)
        ##print(feature_hr,feature_lr)
        
        self.filter_lr(feature_lr)#128x128x(512/256)#error
        self.filter_hr(feature_hr)#512x512x (128/64)#out of memory
        
        # Phase 2: point query application primo mlp
        self.query_lr(points=points, calibs=calibs, transforms=transforms, labels_lr=labels_lr)
        self.query_hr(transforms=transforms, labels_hr=labels_hr)

        
        # get the prediction
        res_hr, res_lr = self.get_preds()
        #uso questi e do input al secondo mlp

        # get the error
        error_mlp1 = self.get_error_lr()
        error_mlp2 = self.get_error_hr()
        error_SR = self.get_errorSR(img_SR,images_hr)

        error=self.opt.mlp1*error_mlp1+self.opt.mlp2*error_mlp2+self.opt.srweight*error_SR

        return res_hr, error, error_mlp1, error_mlp2, error_SR, res_lr