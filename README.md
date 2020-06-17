# Trainig Script for Space Time Memory Network

This repository contains the reimplemented training code for [Space Time Memory Network](http://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html). We implement the data interface for DAVIS16, DAVIS17 and Youtube-VOS, the result of hybrid training on Youtube-VOS and DAVIS17 can be at most 69.7 J&F score on DAVIS17 validation set.

## Required Package
- torch
- python-opencv
- pillow
- yaml
- imgaug
- easydict
- progress

## Data Organization

To run the training and testing code, we require the following data organization format
```
${ROOT}--
        |--${DATASET1}
        |--${DATASET2}
        ...
```
The `ROOT` folder can be set in `libs/dataset/data.py`, which contains all datasets to be used. Each sub-directory ${DATASET} should be the name of one specific dataset (e.g. DAVIS17 or Youtube-VOS) and contain all video and annotation data.

### Youtbe-VOS Organization
To run the training script on youtube-vos dataset, please ensure the data is organized as following format
```
Youtube-VOS
      |----train
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json
      |----valid
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json 
```
Where `JPEGImages` and `Annotations` contain the frames and annotation masks of each video.

### DAVIS Organization

To run the training script on davis16/17 dataset, please ensure the data is organized as following format
```
DAVIS16(DAVIS17)
      |----JPEGImages
      |----Annotations
      |----data
      |------|-----db_info.yaml
```
Where `JPEGImages` and `Annotations` contain the 480p frames and annotation masks of each video.

## Training and Testing
To train the STM network, run following command
```python3
python3 train.py --gpu ${GPU}
```
To test the STM network, run following command
```python3
python3 test.py
```
The test results will be saved as indexed png file at `${ROOT}/${output}/${valset}`.

Additionally, you can modify some setting parameters in `options.py` to change training configuration.
