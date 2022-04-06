# E2EC: An End-to-End Contour-based Method for High-Quality High-Speed Instance Segmentation

![city](city_demo.png)

> [**E2EC: An End-to-End Contour-based Method for High-Quality High-Speed Instance Segmentation**](https://arxiv.org/pdf/2203.04074.pdf)  
> Tao Zhang, Shiqing Wei, Shunping Ji  
> CVPR 2022

Any questions or discussions are welcomed!

## Installation

Please see [INSTALL.md](INSTALL.md).

## Performances

We re-tested the speed on a single RTX3090.

|    Dtataset    |  AP  |  Image size   |  FPS  |
| :------------: |:----:| :-----------: | :---: |
|    SBD val     | 59.2 |    512×512    | 59.60 |
| COCO test-dev  | 33.8 | original size | 35.25 |
|    KINS val    | 34.0 |   768×2496    | 12.39 |
| Cityscapes val | 34.0 |   1216×2432   | 8.58  |

The accuracy and inference speed of the contours at different stages on SBD val set. We also re-tested the speed on a single RTX3090.

| stage |  init  | coarse | final | final-dml |
| :---: | :----: | :----: | :---: | :-------: |
|  AP   |  51.4  |  55.9  | 58.8  |   59.2    |
|  FPS  | 101.73 | 91.35  | 67.48 |   59.6    |

The accuracy and inference speed of the contours at different stages on coco val set.

| stage | init  | coarse | final | final-dml |
| :---: | :---: | :----: | :---: | :-------: |
|  AP   | 27.8  |  31.6  | 33.5  |   33.6    |
|  FPS  | 80.97 | 72.81  | 42.55 |   35.25   |

## Testing

### Testing on COCO

1. Download the pretrained model [here](http://gpcv.whu.edu.cn/member/ZhangTao/model.zip) or [Baiduyun](https://pan.baidu.com/s/1EgEmSFXLVe7dKpqeBG9G3g)(password is `e2ec`).

2. Prepared the COCO dataset according to the [INSTALL.md](INSTALL.md).

3. Test:

   ```
   # testing segmentation accuracy on coco val set
   python test.py coco --checkpoint /path/to/model_coco.pth --with_nms True
   # testing detection accuracy on coco val set
   python test.py coco --checkpoint /path/to/model_coco.pth --with_nms True --eval bbox
   # testing the speed
   python test.py coco --checkpoint /path/to/model_coco.pth --with_nms True --type speed
   # testing the contours of specified stage(init/coarse/final/final-dml)
   python test.py coco --checkpoint /path/to/model_coco.pth --with_nms True --stage coarse
   # testing on coco test-dev set, run and submit data/result/results.json
   python test.py coco --checkpoint /path/to/model_coco.pth --with_nms True --dataset coco_test
   ```

### Testing on SBD

1. Download the pretrained model [here](http://gpcv.whu.edu.cn/member/ZhangTao/model.zip) or [Baiduyun](https://pan.baidu.com/s/1o4gkzs7H_Wyf2Iyoj3xYOA )(password is `e2ec`).

2. Prepared the SBD dataset according to the [INSTALL.md](INSTALL.md).

3. Test:

   ```
   # testing segmentation accuracy on SBD
   python test.py sbd --checkpoint /path/to/model_sbd.pth
   # testing detection accuracy on SBD
   python test.py sbd --checkpoint /path/to/model_sbd.pth --eval bbox
   # testing the speed
   python test.py sbd --checkpoint /path/to/model_sbd.pth --type speed
   # testing the contours of specified stage(init/coarse/final/final-dml)
   python test.py sbd --checkpoint /path/to/model_sbd.pth --stage coarse
   ```

### Testing on KINS

1. Download the pretrained model [here](http://gpcv.whu.edu.cn/member/ZhangTao/model.zip) or [Baiduyun](https://pan.baidu.com/s/1vvXat_jH9d1D5E61ULmYAg)(password is `e2ec`).

2. Prepared the KINS dataset according to the [INSTALL.md](INSTALL.md).

3. Test:

   Maybe you will find some troules, such as `object of type <class 'numpy.float64'> cannot be safely interpreted as an integer`. Please modify the  `/path/to/site-packages/pycocotools/cooceval.py`. Replace `np.round((0.95 - .5) / .05) ` in lines 506 and 507 with `int(np.round((0.95 - .5) / .05))`.

   ```
   # testing segmentation accuracy on KINS
   python test.py kitti --checkpoint /path/to/model_kitti.pth
   # testing detection accuracy on KINS
   python test.py kitti --checkpoint /path/to/model_kitti.pth --eval bbox
   # testing the speed
   python test.py kitti --checkpoint /path/to/model_kitti.pth --type speed
   # testing the contours of specified stage(init/coarse/final/final-dml)
   python test.py kitti --checkpoint /path/to/model_kitti.pth --stage coarse
   ```

### Testing on Cityscapes 

1. Download the pretrained model [here](http://gpcv.whu.edu.cn/member/ZhangTao/model.zip) or [Baiduyun](https://pan.baidu.com/s/1_qrNoviwWdcSDK8LjWDUdA)(password is `e2ec`).

2. Prepared the KINS dataset according to the [INSTALL.md](INSTALL.md).

3. Test:

   We will soon release the code for e2ec with multi component detection. Currently only supported for testing e2ec performance on cityscapes dataset.

   ```
   # testing segmentation accuracy on Cityscapes with coco evaluator
   python test.py cityscapesCoco --checkpoint /path/to/model_cityscapes.pth
   # with cityscapes official evaluator
   python test.py cityscapes --checkpoint /path/to/model_cityscapes.pth
   # testing the detection accuracy
   python test.py cityscapesCoco \
   --checkpoint /path/to/model_cityscapes.pth --eval bbox
   # testing the speed
   python test.py cityscapesCoco \
   --checkpoint /path/to/model_cityscapes.pth --type speed
   # testing the contours of specified stage(init/coarse/final/final-dml)
   python test.py cityscapesCoco \
   --checkpoint /path/to/model_cityscapes.pth --stage coarse
   # testing on test set, run and submit the result file
   python test.py cityscapes --checkpoint /path/to/model_cityscapes.pth \
   --dataset cityscapes_test
   ```

### Evaluate boundary AP

1. Install the Boundary IOU API according [boundary iou](http://github.com/bowenc0221/boundary-iou-api).

2. Testing segmentation accuracy with coco evaluator.

3. Using offline evaluation pipeline.

   ```
   python /path/to/boundary_iou_api/tools/coco_instance_evaluation.py \
       --gt-json-file /path/to/annotation_file.json \
       --dt-json-file data/result/result.json \
       --iou-type boundary
   ```

## Visualization

1. Download the [pretrained model](http://gpcv.whu.edu.cn/member/ZhangTao/model.zip).

2. Visualize:

   ```
   # inference and visualize the images with coco pretrained model
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True
   # you can using other pretrained model, such as cityscapes 
   python visualize.py cityscapesCoco /path/to/images \
   --checkpoint /path/to/model_cityscapes.pth
   # if you want to save the visualisation, please specify --output_dir
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True \
   --output_dir /path/to/output_dir
   # visualize the results at different stage
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True --stage coarse
   # you can reset the score threshold, default is 0.3
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True --ct_score 0.1
   # if you want to filter some of the jaggedness caused by dml 
   # please using post_process
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True \
   --with_post_process True
   ```

## Training

We have only released the code for single GPU training, multi GPU training with ddp will be released soon.

### Training on SBD

```
python train_net.py sbd --bs $batch_size
# if you do not want to use dinamic matching loss (significantly improves 
# contour detail but introduces jaggedness), please set --dml as False
python train_net.py sbd --bs $batch_size --dml False
```

### Training on KINS

```
python train_net.py kitti --bs $batch_size
```

### Training on Cityscapes

```
python train_net.py cityscapesCoco --bs $batch_size
```

### Training on COCO

In fact it is possible to achieve the same accuracy without training so many epochs.

```
# first to train with adam
python train_net.py coco --bs $batch_size
# then finetune with sgd
python train_net.py coco_finetune --bs $batch_size \
--type finetune --checkpoint data/model/139.pth
```

### Training on the other dataset

If the annotations is in coco style:

1. Add dataset information to `dataset/info.py`.

2. Modify the `configs/coco.py`, reset the `train.dataset` , `model.heads['ct_hm']` and `test.dataset`. Maybe you also need to change the `train.epochs`, `train.optimizer['milestones']` and so on.

3. Train the network.

   ```
   python train_net.py coco --bs $batch_size
   ```

If  the annotations is not in coco style:

1. Prepare `dataset/train/your_dataset.py` and `dataset/test/your_dataset.py` by referring to `dataset/train/base.py` and `dataset/test/base.py`.

2. Prepare `evaluator/your_dataset/snake.py` by referring to `evaluator/coco/snake.py`.

3. Prepare `configs/your_dataset.py` and by referring to `configs/base.py`.

4. Train the network.

   ```
   python train_net.py your_dataset --bs $batch_size
   ```

## Citation

If you find this project helpful for your research, please consider citing using BibTeX below:

```
@article{zhang2022e2ec,
  title={E2EC: An End-to-End Contour-based Method for High-Quality High-Speed Instance Segmentation},
  author={Zhang, Tao and Wei, Shiqing and Ji, Shunping},
  journal={arXiv preprint arXiv:2203.04074},
  year={2022}
}
```

## Acknowledgement

Code is largely based on [Deep Snake](https://github.com/zju3dv/snake). Thanks for their wonderful works.
