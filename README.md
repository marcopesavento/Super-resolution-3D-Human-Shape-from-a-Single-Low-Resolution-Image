Pytorch implementation of the paper Super-resolution 3D Human Shape from a Single Low-resolution Image accepted at ECCV 2022.
Note that the code of this repo is heavily based on [PIFU](https://shunsukesaito.github.io/PIFu/). We thank the authors for their great job!

[Project Page](https://marcopesavento.github.io/SuRS/)

## Contents
- [Requirements and dependencies](#requirements-and-dependencies)
- [Dataset creation](#datasets)
- [Train](#train)
- [Test](#test)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Requirements and dependencies
* python 3.7
**packages**
* pytorch >= 1.3.1
* torchvision >= 0.4.0
* pillow >= 6.2.1
* tqdm >= 4.38.0
* future >= 0.18.2
* tensorboard >= 2.0.1
* scikit-learn >= 0.21.3
* kornia >= 0.1.4.post2
* numpy >= 1.17.4
* pandas >= 0.25.3
* googledrivedownloader >= 0.4
* opencv-python >= 4.1.1.26

**dev-packages**
* ipython >= 7.9.0
* pylint >= 2.4.4
* autopep8 >= 1.4.4
* flake8 >= 3.7.9
* jupyterlab >= 1.2.3


## Dataset creation

1. Download [T-Human2.0](https://github.com/ytrock/THuman2.0-Dataset) 
2. Process the .obj file to make the mesh watertight with the [Fast Winding Number](https://www.dgp.toronto.edu/projects/fast-winding-numbers/) algorithm
3. Render the training dataset following [PIFU](https://shunsukesaito.github.io/PIFu/)
4. For Testing set,create two folders and named "mask_final" the folder that contains the mask of the image and "image_final" the folder that contains the RGB input images.

## Train

```shell 
$ python train_SuRS.py --freq_save_ply 25 --residual --dataroot {path_to_input_data} --results_path {path_to_outdir} --random_flip --random_trans --random_scale --num_samples 50000 --threshold 0.05 --b_min -0.5 -0.5 -0.5 --b_max 0.5 0.5 0.5 --sigma 0.06 --resolution 512 --loadSize {input_image_size * 2} 
```

## Test
```shell
$ python eval_SuRS.py --freq_save_ply 25 --residual --dataroot {path_to_input_data} --loadSize {input_image_size * 2} --results_path   {path_to_outdir} --num_samples 50000 --threshold 0.05 --num_threads 6 --resolution 512 --load_netG_checkpoint_path {path_to_checkpoints}/netG_epoch_12 --b_min -0.5 -0.5 -0.5 --b_max 0.5 0.5 0.5
```
# add instruction to train and test SuRS (code is already uploaded)
# add environment.yml
