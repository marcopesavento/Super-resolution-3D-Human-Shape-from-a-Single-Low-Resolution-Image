from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging



import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

log = logging.getLogger('trimesh')
log.setLevel(40)

def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):
    ih, iw = img_in.shape[:2]

    p = scale if multi_scale else 1
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'): #no non lo aggiugne
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]


def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f))#da modificare per il nome

    return meshs

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


class TrainDataset_LR_v2(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        self.scale_pifu = self.opt.scale_pifu

        self.B_MIN = np.array(opt.b_min, dtype=float)
        self.B_MAX = np.array(opt.b_max, dtype=float)

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0,360,1))
        self.pitch_list = [0]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self):
        all_subjects = os.listdir(self.RENDER)
        
        var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        
        pitch = self.pitch_list[pid]

        # The ids are an even distribution of num_views around view_id
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        calib_list = []
        render_list_LR = []
        mask_list_LR = []
        render_list_HR = []
        mask_list_HR = []
        extrinsic_list = []

        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            if not os.path.isfile(render_path):
                render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.png' % (vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))
            if not os.path.isfile(mask_path):
                mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask_HR = Image.open(mask_path).convert('L')
            render_HR = Image.open(render_path).convert('RGB')
      
         

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render_HR = ImageOps.expand(render_HR, pad_size, fill=0)
                mask_HR = ImageOps.expand(mask_HR, pad_size, fill=0)

                w, h = render_HR.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render_HR = transforms.RandomHorizontalFlip(p=1.0)(render_HR)
                    mask_HR = transforms.RandomHorizontalFlip(p=1.0)(mask_HR)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render_HR = render_HR.resize((w, h), Image.BILINEAR)
                    mask_HR = mask_HR.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render_HR = render_HR.crop((x1, y1, x1 + tw, y1 + th))
                mask_HR = mask_HR.crop((x1, y1, x1 + tw, y1 + th))

                render_HR = self.aug_trans(render_HR)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render_HR = render_HR.filter(blur)
                

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()



            #low resolution tensor
            
            mask_LR = mask_HR.resize((x //2 for x in mask_HR.size), Image.NEAREST)

            render_LR = render_HR.resize((x //2 for x in render_HR.size), Image.BICUBIC)



            

            mask_LR = transforms.ToTensor()(mask_LR).float()
           
            render_LR = self.to_tensor(render_LR).float()
         
            render_LR = mask_LR.expand_as(render_LR) * render_LR

            #high resolution tensor

            mask_HR = transforms.ToTensor()(mask_HR).float()
            
            render_HR = self.to_tensor(render_HR).float()
            render_HR = mask_HR.expand_as(render_HR) * render_HR
            
            render_list_HR.append(render_HR)
            mask_list_HR.append(mask_HR)
            render_list_LR.append(render_LR)
            mask_list_LR.append(mask_LR)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
        return {
            'img_LR': torch.stack(render_list_LR, dim=0),
            'img_HR': torch.stack(render_list_HR, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0)
        }

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        name_HR=subject[0]+'_HR.obj'
        name_LR=subject[0]+'_LR.obj'
        mesh_HR = self.mesh_dic[name_HR]
        mesh_LR= self.mesh_dic[name_LR]
        #generation of 3D points
        surface_points, _ = trimesh.sample.sample_surface(mesh_HR, 4 * self.num_sample_inout) #24000 points sampled on the meshes
        threed_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)

        print(name_HR)
        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        threed_points = np.concatenate([threed_points, random_points], 0)
        np.random.shuffle(threed_points)
        #sampling on HR surface
        inside_HR = mesh_HR.contains(threed_points)
        inside_points_HR = threed_points[inside_HR]
        outside_points_HR = threed_points[np.logical_not(inside_HR)]

        
        

        #sampling on LR surface
        inside_LR = mesh_LR.contains(threed_points)
        inside_points_LR = threed_points[inside_LR]
        outside_points_LR = threed_points[np.logical_not(inside_LR)]

        
        nin_LR = inside_points_LR.shape[0]
        inside_points_LR = inside_points_LR[
                        :self.num_sample_inout // 2] if nin_LR > self.num_sample_inout // 2 else inside_points_LR
        outside_points_LR = outside_points_LR[
                         :self.num_sample_inout // 2] if nin_LR > self.num_sample_inout // 2 else outside_points_LR[
                                                                                         :(self.num_sample_inout - nin_LR)]
        nin_HR = inside_points_HR.shape[0]
        inside_points_HR_new = inside_points_HR[
                        :self.num_sample_inout // 2] if nin_HR > self.num_sample_inout // 2 else inside_points_HR
        outside_points_HR_new = outside_points_HR[
                         :self.num_sample_inout // 2] if nin_HR > self.num_sample_inout // 2 else outside_points_HR[
                                                                                         :(self.num_sample_inout - nin_HR)]
        
        label_disp_inside=np.ones((1,self.num_sample_inout//2))
        label_disp_outside=np.zeros((1,self.num_sample_inout//2))
  #         
       
        #classifying LR x points with respect to HR shape
        for i in range(inside_points_LR.shape[0]):
            if inside_points_LR[i] in outside_points_HR:
                label_disp_inside[0][i]=0
                
            #
            if outside_points_LR[i] in inside_points_HR:
                label_disp_outside[0][i]=1
                
            #
        
         
        label_disp=np.concatenate([label_disp_inside, label_disp_outside], 1)                                            
        samples_HR = np.concatenate([inside_points_HR_new, outside_points_HR_new], 0).T
        samples_LR = np.concatenate([inside_points_LR, outside_points_LR], 0).T
        
        labels = np.concatenate([np.ones((1, inside_points_HR_new.shape[0])), np.zeros((1, outside_points_HR_new.shape[0]))], 1)
        
        samples_HR = torch.Tensor(samples_HR).float()
        samples_LR = torch.Tensor(samples_LR).float()
        labels = torch.Tensor(labels).float()
        label_disp=torch.Tensor(label_disp).float()
        
        del mesh_HR
        del mesh_LR

        return {
            'samples_HR': samples_HR,
            'samples_LR': samples_LR,
            'labels_HR': labels,
            'labels_disp': label_disp
        }


    def get_color_sampling(self, subject, yid, pid=0):
    
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

        # Segmentation mask for the uv render.
        # [H, W] bool
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0
        # UV render. each pixel is the color of the point.
        # [H, W, 3] 0 ~ 1 float
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0

        # Normal render. each pixel is the surface normal of the point.
        # [H, W, 3] -1 ~ 1 float
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0
        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]

        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
            surface_points = surface_points[sample_list].T
            surface_colors = surface_colors[sample_list].T
            surface_normal = surface_normal[sample_list].T

        # Samples are around the true surface with an offset
        normal = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() \
                  + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal
    
        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {
            'color_samples': samples,
            'rgbs': rgbs_color
        }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        subject = os.path.splitext(self.subjects[sid])
        res = {
            'name': subject,
            'mesh_path_HR': os.path.join(self.OBJ, subject[0] + '_HR.obj'),
            'mesh_path_LR': os.path.join(self.OBJ, subject[0] + '_LR.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject[0], num_views=self.num_views, yid=yid, pid=pid,
                                        random_sample=self.opt.random_multiview)
        res.update(render_data) #add images and masks
        
        
        if self.opt.num_sample_inout:
           
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)
        
        

        if self.num_sample_color:
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res
       

    def __getitem__(self, index):
        return self.get_item(index)
