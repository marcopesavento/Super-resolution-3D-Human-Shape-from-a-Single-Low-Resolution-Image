from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure


def reconstruction(opt,net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=50000, transform=None):
    '''
    
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query_mr(samples, calib_tensor)
        net.query_sr(samples, calib_tensor)
        pred_hr, pred_lr = net.get_preds()
        return pred_hr.detach().cpu().numpy(), pred_lr.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf_hr,sdf_lr = eval_grid_octree(opt,coords, eval_func, num_samples=num_samples)
    else:
        sdf_hr,sdf_lr = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    
    count=0
    
    verts_hr, faces_hr, normals_hr, values_hr = measure.marching_cubes_lewiner(sdf_hr, 0.5)
    # transform verts into world coordinate system
    verts_hr = np.matmul(mat[:3, :3], verts_hr.T) + mat[:3, 3:4]
    verts_hr = verts_hr.T

    verts_lr, faces_lr, normals_lr, values_lr = measure.marching_cubes_lewiner(sdf_lr, 0.5)
    # transform verts into world coordinate system
    verts_lr = np.matmul(mat[:3, :3], verts_lr.T) + mat[:3, 3:4]
    verts_lr = verts_lr.T
    return verts_hr, faces_hr, normals_hr, values_hr,verts_lr, faces_lr, normals_lr, values_lr



def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
