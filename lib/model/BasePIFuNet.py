import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import index, orthogonal, perspective

class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        """
        :param projection_mode:
        Either orthogonal or perspective.
        It will call the corresponding function for projection.
        :param error_term:
        nn Loss between the predicted [B, Res, N] and the label [B, Res, N]
        """
        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.error_term = error_term

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective
        self.im_SR=None
        self.preds_lr = None
        self.preds_hr = None
        self.labels_hr = None
        self.labels_lr = None
        self.feature_lr=None
        self.feature_hr=None

    def forward(self, points, images_lr, images_hr, calibs, transforms=None):
        '''
        :param points: [B, 3, N] world space coordinates of points
        :param images: [B, C, H, W] input images
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :return: [B, Res, N] predictions for each point
        '''
        self.super_res(images_lr)
        self.filter_lr(self.feature_lr)
        self.filter_hr(self.feature_hr)
        self.query_lr(points, calibs, transforms)
        self.query_hr(transforms)
        return self.get_preds()

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        None

    def super_res(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        return self.im_SR, self.feature_lr, self.feature_hr

    def query(self, points, calibs, transforms=None, labels=None):
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
        None

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.preds_hr

    def get_error_hr(self):
        '''
        Get the network loss from the last query
        :return: loss term
        '''
        return self.error_term(self.preds_hr, self.labels_hr)
    
    def get_error_lr(self):
        '''
        Get the network loss from the last query
        :return: loss term
        '''
        return self.error_term(self.preds_lr, self.labels_lr)

