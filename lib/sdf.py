import numpy as np


def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)

    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
  
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.shape[1]
    sdf_lr = np.zeros(num_pts)
    sdf_hr=np.zeros(num_pts)

    num_batches = num_pts // num_samples
    
    for i in range(num_batches):
        sdf_hr[i * num_samples:i * num_samples + num_samples],sdf_lr[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    
    if num_pts % num_samples:
        
        sdf_hr[num_batches * num_samples:],sdf_lr[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])
    print(sdf_hr.shape,sdf_lr.shape)
    return sdf_hr, sdf_lr


def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf_hr, sdf_lr = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf_hr.reshape(resolution),sdf_lr.reshape(resolution)


def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf_lr = np.zeros(resolution)
    sdf_hr = np.zeros(resolution)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        #print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf_hr[test_mask],sdf_lr[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf_hr[x, y, z]
                    v1 = sdf_hr[x, y, z + reso]
                    v2 = sdf_hr[x, y + reso, z]
                    v3 = sdf_hr[x, y + reso, z + reso]
                    v4 = sdf_hr[x + reso, y, z]
                    v5 = sdf_hr[x + reso, y, z + reso]
                    v6 = sdf_hr[x + reso, y + reso, z]
                    v7 = sdf_hr[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf_hr[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
                    
                    v0_lr = sdf_lr[x, y, z]
                    v1_lr = sdf_lr[x, y, z + reso]
                    v2_lr = sdf_lr[x, y + reso, z]
                    v3_lr = sdf_lr[x, y + reso, z + reso]
                    v4_lr = sdf_lr[x + reso, y, z]
                    v5_lr = sdf_lr[x + reso, y, z + reso]
                    v6_lr = sdf_lr[x + reso, y + reso, z]
                    v7_lr = sdf_lr[x + reso, y + reso, z + reso]
                    v_lr = np.array([v0_lr, v1_lr, v2_lr, v3_lr, v4_lr, v5_lr, v6_lr, v7_lr])
                    v_min_lr = v_lr.min()
                    v_max_lr = v_lr.max()
                    # this cell is all the same
                    if (v_max_lr - v_min_lr) < threshold:
                        sdf_lr[x:x + reso, y:y + reso, z:z + reso] = (v_max_lr + v_min_lr) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

    return sdf_hr.reshape(resolution), sdf_lr.reshape(resolution)
