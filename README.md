# HuBMAP - Hacking the kidney

The project developed for the Computer Vision Project Work. It contains some approaches for the "HuBMAP - Hacking the Kidney" competition hosted on [Kaggle](https://www.kaggle.com/c/hubmap-kidney-segmentation).

The objective is detecting functional tissue units (FTUs) across different tissue preparation pipelines. An FTU is defined as a “three-dimensional block of cells centered around a capillary, such that each cell in this block is within diffusion distance from any other cell in the same block” (de Bono, 2013). The proposed deep learning segmentation models are trained to identify the glomeruli in the annotated PAS stained microscopy data. 

## Setup

Create a Python 3 virtual environment, activate it and run the command

```
pip install -r requirements.txt
```

Data can be found in the competition website. Models were trained on Google Colab and tested on Kaggle.

## Approaches

Notebooks contain the presentation to the proposed methods
- Prerequisites, contains a EDA and data preparation 
- Baseline, presents results using U-Net 
- DeepLab, proposes DeepLabV3
- HookNet, proposes HookNet

