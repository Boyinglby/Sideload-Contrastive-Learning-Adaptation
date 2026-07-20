> Code release for the paper:  
> **Enhancing Vehicle Detection under Adverse Weather Conditions with Contrastive Learning**  

---
<img width="2350" height="1316" alt="overview" src="https://github.com/Boyinglby/NVD-sideload_YOLO/blob/main/overview.png" />

## Table of Contents

- Overview
- Pretraining
- Finetuning
- Evaluation

---

## Overview
This repository contains the implementation of our paper, which introduces a sideload-ICL-adaption framework that incorperating contrastive learning pretraining strategy with lightweight YOLO11n object detector. 
We provide code for training, evaluation, and reproducing the main results. Our proposed framework consists of two stages: pretraining and finetuning, both stages use the Nordic Vehicle Dataset(https://nvd.ltu-ai.dev/). The
description of NVD dataset, how to **Setup**, **Data Visualization and Preparation** and **Set up Logging** are well documented in the NVD repo: https://github.com/Amirhossein-Nayebi/Nordic-Vehicle-Dataset, please follow the NVD repo for installation, data preparation and set up loggoing.
## Pretraining 
- Download the Unannotated Videos from https://nvd.ltu-ai.dev/
- As described as **Prepare data** in NVD repo to extract the unannotated frames from the video, store the frames in a folder and name the folder as "dataset_unanotated/frames"
- Conduct the pretraining by run: `python feature_map_CL.py --frame-dir dataset_unanotated/frames --exclude-prefix "frame2022-12-04 Bjenberg"`
- After pretraining, the pretrained side CNN model will be save as 'patchCL_yolo_backbone_best.pt'.
## Modifications to Ultralytics YOLO 
This project includes custom modifications to the official Ultralytics YOLO library to support sideloading a pretrained CNN during training. 
An easy way to implement the custom modifications is to first 
`pip install ultralytics` and replace the files in the table to our modified ones (files in the Ultraylytics folder).

| Example File path| Function/Class|Description of change|
| ------------- | ------------- |------------- |
| lib/python3.10/site-packages/ultralytics/nn/modules/conv.py  |Add, LearnableWei, Conv0, SEblock|Inplement fusion blocks|
| lib/python3.10/site-packages/ultralytics/nn/tasks.py  | parse_model  |Enable it to parse the new fusion blocks|
- Now you can sideload the pretrained CNN with Yolo11n by run: `python load_pretrainweights_yolobackbone_SE.py`
## Finetuning 
- Download the annotated Videos from https://nvd.ltu-ai.dev/
- As described as 'Prepare data' in NVD repo, extract the annotated frames from the video and make the train/val/test split
- Finetune the sideload model on the annoated data by run: `python train.py [--epochs EPOCHS] [--yolo_model YOLO_MODEL] [--batch BATCH-SIZE] [--aug] [--freeze FROZEN-LAYERS] [--seed SEED]`
## Evaluation
Following the description of **Test** in NVD repo to evaluate the model.

---


