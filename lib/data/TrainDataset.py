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
#le immagini hanno lo stesso nome delle meshes. alle meshes aggiungo un _HR e un _LR
log = logging.getLogger('trimesh')
log.setLevel(40)

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


class TrainDataset(Dataset):
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

        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])

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
            #transforms.Resize(self.load_size), qua modificato ma potrebbe essere sbagliato
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
            ##print(render_path)
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

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
      
            


            #low resolution tensor

            








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

            mask_LR = mask_HR.resize((x - (x % 2) for x in mask_HR.size), Image.BICUBIC)

            render_LR = render_HR.resize((x - (x % 2) for x in render_HR.size))

            mask_LR = mask_HR.resize((x //2 for x in mask_LR.size), Image.NEAREST)

            render_LR = render_HR.resize((x //2 for x in render_LR.size), Image.BICUBIC)

            mask_LR = transforms.ToTensor()(mask_LR).float()
           

            render_LR = self.to_tensor(render_LR)
         
            render_LR = mask_LR.expand_as(render_LR) * render_LR

            

            #high resolution tensor

            mask_HR = transforms.ToTensor()(mask_HR).float()
            
            render_HR = self.to_tensor(render_HR)
            render_HR = mask_HR.expand_as(render_HR) * render_HR
            #mask_LR = transforms.Resize(self.load_size)(mask_LR)
            
            render_list_HR.append(render_HR)
            mask_list_HR.append(mask_HR)
            render_list_LR.append(render_LR)
            mask_list_LR.append(mask_LR)
            ###print(len(render_list_HR),len(mask_list_LR),len(render_list_LR),len(mask_list_HR))
            calib_list.append(calib)
            ###print(len(calib_list))
            extrinsic_list.append(extrinsic)
            #print(render_list_LR[0].shape,render_list_HR[0].shape)
        return {
            'img_LR': torch.stack(render_list_LR, dim=0),
            'img_HR': torch.stack(render_list_HR, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask_LR': torch.stack(mask_list_LR, dim=0),
            'mask_HR': torch.stack(mask_list_HR, dim=0)
        }

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        name_HR=subject[0]+'_HR.obj'
        name_LR=subject[0]+'_LR.obj'
        ##print(self.mesh_dic)
        mesh_HR = self.mesh_dic[name_HR]
        mesh_LR= self.mesh_dic[name_LR]
        ##print(name_HR,name_LR)
        surface_points, _ = trimesh.sample.sample_surface(mesh_HR, 4 * self.num_sample_inout) #20000 points sampled on the meshes
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)

        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)
        ###print("ok1")
        inside_HR = mesh_HR.contains(sample_points) #non mi lascia
        ###print("ok2")
        inside_points_HR = sample_points[inside_HR]
        outside_points_HR = sample_points[np.logical_not(inside_HR)]
        
        inside_LR = mesh_LR.contains(sample_points)
        inside_points_LR = sample_points[inside_LR]
        outside_points_LR = sample_points[np.logical_not(inside_LR)]
        ###print("ok3")
        nin_LR = inside_points_LR.shape[0]
        inside_points_LR = inside_points_LR[
                        :self.num_sample_inout // 2] if nin_LR > self.num_sample_inout // 2 else inside_points_LR
        outside_points_LR = outside_points_LR[
                         :self.num_sample_inout // 2] if nin_LR > self.num_sample_inout // 2 else outside_points_LR[
                                                                                              :(self.num_sample_inout - nin_LR)]
        nin_HR = inside_points_HR.shape[0]
        inside_points_HR = inside_points_HR[
                        :self.num_sample_inout // 2] if nin_HR > self.num_sample_inout // 2 else inside_points_HR
        outside_points_HR = outside_points_HR[
                         :self.num_sample_inout // 2] if nin_HR > self.num_sample_inout // 2 else outside_points_HR[
                                                                                               :(self.num_sample_inout - nin_HR)]
        ###print("ok5")                                                     
        samples = np.concatenate([inside_points_HR, outside_points_HR], 0).T
        labels_HR = np.concatenate([np.ones((1, inside_points_HR.shape[0])), np.zeros((1, outside_points_HR.shape[0]))], 1)
        labels_LR = np.concatenate([np.ones((1, inside_points_LR.shape[0])), np.zeros((1, outside_points_LR.shape[0]))], 1)

        # save_samples_truncted_prob('out.ply', samples.T, labels.T)
        # exit()
        ###print("ok6")
        samples = torch.Tensor(samples).float()
        labels_LR = torch.Tensor(labels_LR).float()
        labels_HR = torch.Tensor(labels_HR).float()
        
        del mesh_HR
        del mesh_LR

        return {
            'samples': samples,
            'labels_HR': labels_HR,
            'labels_LR': labels_LR
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

        # name of the subject 'rp_xxxx_xxx'
        subject = os.path.splitext(self.subjects[sid])
        ###print(subject)
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
        ###print(res)
        render_data = self.get_render(subject[0], num_views=self.num_views, yid=yid, pid=pid,
                                        random_sample=self.opt.random_multiview)
        res.update(render_data) #add images and masks
        
        
        #qua devo mettere low resolution e high resolution! occhio che low resolution mi interessa solo groundtruth
        #mi conviene associare l'object mesh direttamente as input data quindi a subject!
        if self.opt.num_sample_inout:
           
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)
        
        # img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        # rot = render_data['calib'][0,:3, :3]
        # trans = render_data['calib'][0,:3, 3:4]
        # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        # for p in pts:
        #     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        if self.num_sample_color:
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        ###print(res)
        return res
        # except Exception as e:
        #     ##print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        ##print(index)
        return self.get_item(index)