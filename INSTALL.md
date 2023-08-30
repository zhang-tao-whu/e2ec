### Set up the python environment

```
conda create -n e2ec python=3.7
conda activate e2ec

# install pytorch, the cuda version is 11.1
# You can also install other versions of cuda and pytorch, but please make sure # that the pytorch cuda is consistent with the system cuda

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install Cython==0.28.2
pip install -r requirements.txt
```

### Compile cuda extensions 

```
ROOT=/path/to/e2ec
cd $ROOT/network/backbone/DCNv2-master
# please check your cuda version and modify the cuda version in the command
export CUDA_HOME="/usr/local/cuda-11.1"
bash ./make.sh
```

Maybe you will encounter some build errors. You can choose a plan :

1.  You can look for another implementation of DCN-V2 and compiled successfully.
2.  You can set `cfg.model.use_dcn` as `False`. This may result in a slight drop in accuracy.
3.  You can install **mmcv**, and replace  352 line of `network/backbone/dla.py` as `from mmcv.ops import ModulatedDeformConv2dPack as DCN`, replace the `deformable_groups` in 353 line as `deform_groups`.

### Set up datasets

#### Cityscapes

1. Download the Cityscapes dataset (leftImg8bit\_trainvaltest.zip) from the official [website](https://www.cityscapes-dataset.com/downloads/).

2. Download the processed annotation file [cityscapes_anno.tar.gz](https://drive.google.com/file/d/1hj1um8EE8SuJQhEWvmI-d8rkJe-AEVpi/view?usp=sharing).

3. Organize the dataset as the following structure:
    ```
    ├── /path/to/cityscapes
    │   ├── annotations
    │   ├── coco_ann
    │   ├── leftImg8bit
    │   ├── gtFine
    ```
    
4. Create a soft link:
    ```
    ROOT=/path/to/e2ec
    cd $ROOT/data
    ln -s /path/to/cityscapes cityscapes
    ```

#### Kitti

1. Download the Kitti dataset from the official [website](http://www.cvlibs.net/download.php?file=data_object_image_2.zip).
2. Download the annotation file `instances_train.json` and `instances_val.json` from [Kins](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset).
3. Organize the dataset as the following structure:

	├── /path/to/kitti
	│   ├── testing
	│   │   ├── image_2
	│   │   ├── instance_val.json
	│   ├── training
	│   │   ├── image_2
	│   │   ├── instance_train.json
   ```
4. Create a soft link:
   ```
    ROOT=/path/to/e2ec
    cd $ROOT/data
    ln -s /path/to/kitti kitti

#### Sbd

1. Download the Sbd dataset at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EV2P-6J0s-hClwW8uZy1ZXYBPU0XwR7Ch7EBGOG2vfACGQ?e=wpyE2M).
2. Create a soft link:
    ```
    ROOT=/path/to/e2ec
    cd $ROOT/data
    ln -s /path/to/sbd sbd
    ```

#### COCO

1. Download the Sbd dataset at [here](https://cocodataset.org/#download).

2. Organize the dataset as the following structure:

   ```
   ├── /path/to/coco
   │   ├── annotations
   │   │   ├── instances_train2017.json
   │   │   ├── instances_val2017.json
   │   │   ├── image_info_test-dev2017.json
   │   ├── train2017
   │   ├── val2017
   │   ├── test2017
   ```

3. Create a soft link:

   ```
   ROOT=/path/to/e2ec
   cd $ROOT/data
   ln -s /path/to/coco coco
   ```
