# Blurry Image Detection

The goal of the project is to build and deploy Machine Learning model that predicts if image (scan/photo of receipt or invoice) is blurred. 

Imagine following use case: company is applying OCR to customers' receipts, OCR operation is costly, so it would be nice to check if image quality is suitable for processing.


## Prerequisites

This is python (ver 3.11) project. The following packages should be present:
- gradio
- matplotlib
- numpy
- pillow
- tensorflow
- zipfile

(Optional) Docker should be installed if you want to deploy resulting model as docker image.    


## Project structure

### dataset
Folder contains zip file with blurred and non-blur images. The dataset is based on [Receipt Image Dataset](https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/), 
some images were artificially blurred (I used batch image processing utility of [IrfanView](https://www.irfanview.com/) application).

### model
Contains the source code  
- [notebook.ipynb](model/notebook.ipynb) - notebook used for EDA and hyperparameter tuning. (I used Google Colab to run it).
- [train.py](model/train.py) - python script that builds model and serializes it as `.tflite` file.

### deployment_local
Contains `app.py` script to launch [Gradio](https://www.gradio.app/) application that performs blur detection.

### deployment_cloud
Instructions how to deploy model on [Hugging Face](https://huggingface.co/) cloud platform. 

### deployment_docker
Folder contains files necessary to deploy model as Docker image.

### test
Instructions how to use/test predictions using dockerized model. 

