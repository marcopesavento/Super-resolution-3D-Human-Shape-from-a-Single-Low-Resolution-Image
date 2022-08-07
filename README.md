Pytorch implementation of the paper Super-resolution 3D Human Shape from a Single Low-resolution Image accepted at ECCV 2022.
Note that the code of this repo is heavily based on [PIFU](https://shunsukesaito.github.io/PIFu/). We thank the authors for their great job!

[Project Page](https://marcopesavento.github.io/SuRS/)

## Contents
- [Environment](#Environment)
- [Dataset creation](#datasets)
- [Train](#train)
- [Test](#test)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Environment
Create a conda environment from environment.yml file:
```shell
conda env create -f environment.yml
```
The first line of the yml file sets the new environment's name.


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

## Citation
If you find the code useful in your research, please consider citing the paper.

```
ARXIVE when submitted

}
```
## Contacts

If you meet any problems, please describe them in issues or contact:

* Marco Pesavento: m.pesavento@surrey.ac.uk

## Acknowledgments

This research was supported by UKRI EPSRC Platform Grant EP/P022529/1.

