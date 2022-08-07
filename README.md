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


## DATASET CREATION

[Download [T-Human2.0] (https://github.com/ytrock/THuman2.0-Dataset) and process the .obj file to make the mesh watertight with the [Fast Winding Number] algorithm(https://www.dgp.toronto.edu/projects/fast-winding-numbers/)]
[Render the training dataset following [PIFU](https://shunsukesaito.github.io/PIFu/)]

# add instruction to train and test SuRS (code is already uploaded)
# add environment.yml
