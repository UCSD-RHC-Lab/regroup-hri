
# REGROUP-HRI 🤖

REGROUP: A Robot-Centric Group Detection and Tracking System

REGROUP is a new system that enables robots to detect and track groups of people from an ego-centric perspective using a crowd-aware, tracking-by-detection approach (see Figures 1-2).

[[Paper]](https://dl.acm.org/doi/abs/10.5555/3523760.3523816) [[Video]](https://youtu.be/mD-uUtgVgxo)



<p align="center"> 
<img src="./images/regroup.gif" width="70%"> 
<i>

<b>Figure 1:</b> This shows REGROUP running on an RGB video stream, as captured from a mobile  robot. 

</i> 
</p>

Contents: 
- [Introduction](#introduction) 
- [REGROUP Overview](#overview)
- [Installation](#installation) 
- [Code Structure Overview](#code_structure_overview) 
- [Dataset Setup](docs/dataset_setup.md) 
- [Usage](#usage) 
- [Experiments](docs/experiments.md)

## Introduction

To facilitate the Human-Robot Interaction field's transition from dyadic to group interaction, new methods are needed for robots to sense and understand team behavior. We introduce the Robot-Centric Group Detection and Tracking System (REGROUP), a new method that enables robots to detect and track groups of people from an ego-centric perspective using a crowd-aware, tracking-by-detection approach. Our system employs a novel technique that leverages person re-identification deep learning features to address the group data association problem. REGROUP is robust to real-world vision challenges such as occlusion, camera egomotion, shadow, and varying lighting illuminations. Also, it runs in real-time on real-world data. 

You can use this bibtex if you would like to cite this work ([Taylor](https://www.angeliquemtaylor.com/) and [Riek](https://cseweb.ucsd.edu/~lriek), 2022): 

``` 
@article{taylor_2022, 
author = {Taylor, A. and Riek, L.D.}, 
title = {REGROUP: A Robot-Centric Group Detection and Tracking System}, 
journal = {In Proc. of the 17th Annual ACM/IEEE Conference on Human Robot Interaction (HRI).}, 
year = {2022}
}
```

## REGROUP Overview



<p align="center"> 
<img src="images/regroup_arch.png" width=860></img> 
</p>

<b>Figure 2:</b> Given an image sequence, REGROUP extracts pedestrian patches, extract appearance descriptors using a CNN. Then, it uses these feature vectors to track pedestrians. Then, REGROUP's group detector uses these tracks to detect groups, then track them using group detections and our crowd indication feature (CIF), which enables REGROUP to handle high levels of occlusion.

We use the pedestrian tracker from [Deep Sort](https://github.com/nwojke/deep_sort).

## Installation 

### Install REGROUP

You can install the dependencies with `pip`: 
``` 
git clone https://github.com/UCSD-RHC-Lab/regroup-hri.git 
cd regroup-hri 
pip install opencv-python
pip install numpy
```

Additional libraries to install:

- [Tensorflow](https://www.tensorflow.org/install/pip#virtual-environment-install)

### Prerequisites (Pedestrian Detection with YOLO)

Download [YOLOv3]( https://github.com/pjreddie/darknet) weights, configuration file, and class name files: 


```
mkdir regoup/darknet
mkdir regroup/darknet/model 
cd regroup/darknet/model

wget https://opencv-tutorial.readthedocs.io/en/latest/_downloads/549b18ea691a01b06e888f9bb6b35900/yolo1.py

wget https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg

wget https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names
```

### Prerequisites (Install Person Re-Identification Convolutional Neural Network models)

Download [network weights](https://drive.google.com/file/d/1iyHztyVMBqjwyFbZ__I4TAcnfRJ4x_RJ/view?usp=sharing) and add the contents of the folder to the regroup/resources/networks folder.


## Code Structure Overview

`regroup-hri` contains: 

`regroup/`: stores files to run REGROUP
- `regroup`:
- `darknet/`: contains YOLO pedestrian detection system from [here](https://github.com/pjreddie/darknet).
- `resources/`: store person re-identification Convolutional Neural Network models from [Deep SORT](https://github.com/nwojke/deep_sort)
- `tools/`: stores helper scripts to generate pedestrian detections
- `application_util/`: stores group tracking visualization code

`data/`: stores the data set discussed in the [paper](https://cseweb.ucsd.edu/~lriek/)
- `test`: stores the testing dataset

- `images/`: images on GitHub README

## Dataset Setup

The input data for REGROUP is stored in the regroup/data folder (See example image in Figure 3). REGROUP requires pedestrian detections as input. The dataset should be formatted as follows:

`data/`: stores the data set discussed in the [paper](https://dl.acm.org/doi/abs/10.5555/3523760.3523816)
- `test`: stores the testing dataset
- `[sequence]/`: stores small dataset with custom sequence name (user-defined) 
- `rgb/`: folder that stores RGB data
- `det/`: folder that stores pedestrian detection files

```
mkdir data
mkdir data/test
mkdir data/test/group-01/rgb
mkdir data/test/group-01/det
```

Here, [sequence] is group-01. 

<p align="center"> 
<img src="images/00307.png" width=860></img> 
</p>

<b>Figure 3:</b> Example input RGB image for REGROUP.


The pedestrian detection files are formatted as (see Figure 3):

`<image frame ID>, <pedestrian track ID>, <top-right-x-coordinate>, <top-right-y-coordinate>, <width>, <height>, -1, -1, -1, -1`

<p align="center"> 
<img src="./images/pedestrian_detections.png" width="70%"> 
<i>

<b>Figure 4:</b> Example pedestrian detections from data/test/[sequence]/det/det.txt

## Usage

### Running REGROUP on a video sample. The REGROUP system outputs group tracks as bounding box coordinates in a video consistent with the [Multiple Object Tracking literature](https://motchallenge.net/). It uses the same format as pedestrian detections (see Figure 4 for an example):



The basic structure of commands is the following: 

`python regroup-pre-stored-dets.py --sequence_dir=<sequence_dir> --display=<display> --img_output_path=<img_output_path>` 

where `<sequence_dir>` is the dataset and group directory, `<display>` indicates whether the tracker displays the video as the tracker runs, `<img_output_path>` is the path to save the image data that shows group tracks.

After the dataset is set up, run the tracker on a set of images (e.g., <sequence_dir> shown below) as follows:

`python regroup-pre-stored-dets.py --sequence_dir=data/test/[sequence] --display=1 --output_file=data/test/[sequence].txt` 



Additional parameters include:
- detection_file: path to pedestrian detections
- distance_metric: distance metric for data association (default="cosine")
- output_file: path to the tracking output file. This file will contain the tracking results on completion
- min_confidence: detection confidence threshold. Disregard all detections that have a confidence lower than this value (default=-1)
- min_detection_height: threshold on the detection bounding box height. Detections with height smaller than this value are disregarded (default=40)
- cif: crowd indication feature threshold (default=4)
- nms_max_overlap: non-maxima suppression threshold: maximum detection overlap (default=0.7)
- max_cosine_distance: Gating threshold for cosine distance metric object appearance (default=0.9)
nn_budget: maximum size of the appearance descriptors gallery. If None, no budget is enforced (default=100)
- display: show intermediate tracking results (default=0)
- img_output_path: path to write images
- group_detector: group detection method (default='regroup')
- model: path to freezed tensorflow inference graph protocol buffer, which is used to extract features from [person re-identification Convolutional Neural Networks](https://github.com/nwojke/deep_sort) (default='regroup/resources/networks/mars-small128.pb')
- gp_dist: ground plane distance threshold \kappa (default=20)
- h_ratio: height ratio threshold \alpha [0,1] (default=0.8)
- cost_metric: cost_metric = {0-appearance, 1-motion, 2-both}  (default=2)

## Generate Pedestrian Detections

REGROUP requires pedestrian detections as input to predict group detections. These are instructions to run the pedestrian detection on a video stream and on pre-stored RGB image sequences that have been extracted from a video sequence. Be sure to store RGB data in the regroup/data/test/[sequence]>/rgb directory.

To generate pedestrian detections on a video stream using OpenCV Library, provide the path to the video and output pedestrian detection filename:
``` 
cd regroup/darknet/ 
python generate_ped_detections_video.py --sequence_dir=<../data/test/[sequence]> --video_path=<../data/test/[sequence].mp4> --output_filename=<../data/test/[sequence]/det/det.txt> 
```

To run YOLO on pre-stored image data using OpenCV Library, provide the path to the directory where the image data is stored and output pedestrian detection filename:
``` 
cd regroup/darknet/
python generate_ped_detections_image.py --sequence_dir=<../data/test/[sequence]> --video_path=<path to video>
```
## Different Ways of Deploying REGROUP

REGROUP can be run either on a pre-recorded video stream or on pre-stored pedestrian detections.

REGROUP requires pedestrian detections as input to predict group detections. There is no need to regenerate pedestrian detections each time to REGROUP. After collecting pedestrian detections and storing them in the correct format, you can run REGROUP using the detections directly for computer vision benchmarking. It is useful to test vision systems offline, particularly using preprocessed pedestrian detections which are conveniently provided in this repository. For instance, REGROUP performs non-maximum suppression on bounding boxes and it removes 'bad' bounding boxes automatically (see parameters from the Usage Section).

### Run on video stream


To run REGROUP on a video stream using OpenCV Library, provide the sequence directory and path to the video:
``` 
cd regroup/
python regroup-yolo-video-input.py --sequence_dir=<../data/test/[sequence]> --video_path=<../data/test/[sequence]/[sequence].mp4> 
```

regroup-yolo-video-input.py will generate an image sequence from the video at video_path and store the images in the sequence_dir.

REGROUP assumes a 640X360 image resolution with a frame rate of at least 30 frames per second, so be sure to pre-process the data to be consistent with this.


### Run on pre-stored pedestrian detections

To REGROUP offline with pre-stored pedestrian detections, you need to download YOLO first. See [Installation Section](#installation) for details. Store the RGB images in the regroup/data/test/[sequence]/rgb directory, and store pre-stored pedestrian detections in a file located at data/test/[sequence]/det/det.txt



``` 
cd regroup/
python regroup-yolo.py --sequence_dir=<../data/test/[sequence]> 
```

### Alternate Group Detectors


We compared REGROUP's group detector to four state-of-the-art group detectors. We will provide code for comparative methods to enable HRI researchers to benchmark their vision systems for the group perception problem domain in the near future.

### Additional comments

The code in this repository generates the results found in our paper (see Figures 5-7). 

Results for group detection experiments: 

<p align="center"> 
<img src="images/detection.png" width=460></img> 
</p>




<b>Figure 5:</b>  Results for group detection and tracking ablation experiments: 

<p align="center"> 
<img src="images/tracking.png" width=860></img> 
</p>


<b>Figure 6:</b> Group Detection and Tracking Results. We report Multiple Object Tracking Accuracy (MOTA &#8593;), Multiple Object Tracking Precision (MOTP &#8593;), Mostly Tracked Targets (MT &#8593;), Mostly Lost Targets (ML &#8595;), False Positives (FP &#8595;), False Negatives (FN &#8595;), Total Number of ID Switches (IDsw &#8595;), and end-to-end computation time in seconds per image (t(s)) where &#8593; means higher is better and &#8595; means lower is better.

<p align="center"> 
<img src="images/visual_results.png" width=860></img> 
</p>



<b>Figure 7:</b> Visual results from ablation experiments using REGROUP with NCuts and Self-Tuning-SC.

## Further Issues and questions ❓ 

If you have issues or questions, don't hesitate to contact [Angelique Taylor](https://www.angeliquemtaylor.com/) at amt062@eng.ucsd.edu, or amt298@cornell.edu.

