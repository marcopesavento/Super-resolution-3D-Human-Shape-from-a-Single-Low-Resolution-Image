import torch
import numpy as np
from .mesh_util import *
from .sample_util import *
from .geometry import *
import cv2
from PIL import Image
from tqdm import tqdm
import imageio
import numpy as np

from torchvision.utils import save_image

def reshape_multiview_tensors(image_tensor_hr,image_tensor_lr, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor_hr = image_tensor_hr.view(
        image_tensor_hr.shape[0] * image_tensor_hr.shape[1],
        image_tensor_hr.shape[2],
        image_tensor_hr.shape[3],
        image_tensor_hr.shape[4]
    )
    image_tensor_lr = image_tensor_lr.view(
        image_tensor_lr.shape[0] * image_tensor_lr.shape[1],
        image_tensor_lr.shape[2],
        image_tensor_lr.shape[3],
        image_tensor_lr.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor_hr, image_tensor_lr, calib_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor

def gen_mesh(opt, net, cuda, data, save_path, use_octree=True):
    image_tensor = data['img_LR'].to(device=cuda)
        
    
    img_SR,feature_lr, feature_hr=net.super_res(image_tensor)
    net.filter_hr(feature_hr) 
    net.filter_lr(feature_lr)

    b_min = data['b_min']
    b_max = data['b_max']
    projection_matrix = np.identity(4)*2
    projection_matrix[1, 1] = -2
    projection_matrix[3,3] = 1
    calib = torch.Tensor(projection_matrix).float().unsqueeze(0)
    calib_tensor=calib.to(device=cuda)
    



    verts_hr, faces_hr, _, _,verts_lr, faces_lr, _, _ = reconstruction(
        opt,net, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree,num_samples=opt.num_samples)

    save_path_hr = save_path[:-4] + '_HR.obj'
    
    
    
        
    
    save_obj_mesh(save_path_hr, verts_hr, faces_hr)

    save_path_lr = save_path[:-4] + '_LR.obj'
    
    save_obj_mesh(save_path_lr, verts_lr, faces_lr)



def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr



