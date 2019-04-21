![imageailogo](http://imageai.org/xlogo2.png.pagespeed.ic.0XXrMSSEh2.png "ImageAI")

[Overview](#overview)
[System Requirements](#requirements)
[Dependencies](#dependencies)
[Experiments](#experiments)
[imageai.ipynb](#imageai)
[imageaimodel.ipynb](#imageaimodel)
[imageaivideo.ipynb](#imageaivideo)
[neuralnetwork.ipynb](#neuralnetwork)
[Documentation](#documentation) 

<a name="overview"/>
# Overview
Created by Moses Olafenwa and John Olafenwa, *ImageAI 2.0.2* serves as a gateway into Deep Learning using Machine Learning algorithms for image prediction, object detection, video detection, custom image prediction, and video object tracking.  This library features pre-trained models of traditional algorithms including RetinaNet, YOLOv3, TinyYOLOv3, SqueezeNet, ResNet50, InceptionV3, DenseNet121, and others.

<a name="requirements"/>
## System Requirements
Google Colaboratory’s virtual environment was used in these experiments which consists of free GPU (NVIDIA Tesla K80). It is assumed that the user has basic understanding of Google Colaboratory. The dependencies configured for the project include the following:

<a name="dependencies"/>
### Installing Dependencies
- Python 3
- TensorFlow
- NumPy
- SciPy
- OpenCV
- Pillow
- Matplotlib
- h5p5
- Keras
- ImageAI 2.0.2

```
!pip3 install tensorflow
!pip3 install numpy
!pip3 install scipy
!pip3 install opencv-python
!pip3 install pillow
!pip3 install matplotlib
!pip3 install h5py
!pip3 install keras
!pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
```

A system performance test on the virtual machine prior to conducting the experiments produces a usage of general RAM free at 12.9 GB, processor size of 142.7 MB, and GPU RAM free at 11,441 MB. 

<a name="experiments"/>
## Experiments
My experiments using ImageAI 2.0.2 with ImageNet, MNIST, and CUB-2011 datasets demonstrate significant viability of the library. Analysis of videos from the VIMS lab in University of Delaware and images from Kaggle using the ImageAI Python Library.

<a name="imageai"/>
### imageai.ipynb
A series of Image Detection tests including the following: 

1. A basic implementation of the FirstDetection.py file as found in the tutorial from Medium, [Object Detection with 10 lines of code](https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606). 

2. A custom dataset implementation which included [1,000 cat and dog images](https://www.kaggle.com/dhainjeamita/dogs-and-cats-image-classification) from Kaggle. 

<a name="imageaimodel"/>
### imageaimodel.ipynb
A series of Custom Image Prediction tests including the following: 

1. Training a data augmented subset of the custom dataset from imageai.ipynb #2 with 45 degree rotated images as found in the tutorial from Medium, [Train Image Recognition AI with 5 lines of code](https://towardsdatascience.com/train-image-recognition-ai-with-5-lines-of-code-8ed0bdd8d9ba). 

2. A comparison of the existing ResNet model vs. my trained model which includes both single image prediction and multiple image prediction implementations.

<a name="imageaivideo"/>
### imageaivideo.ipynb
A series of Image Detection Testing including the following:  

1. A basic implementation of the FirstVideoDetection.py file as found in the tutorial from Medium, [Detecting objects in videos and camera feeds using Keras, OpenCV, and ImageAI](https://heartbeat.fritz.ai/detecting-objects-in-videos-and-camera-feeds-using-keras-opencv-and-imageai-c869fe1ebcdb).

2. Custom video analysis using data supplied by the Video/Image Modeling and Synthesis Lab (VIMS) at the University of Delaware.

<a name="neuralnetwork"/>
### neuralnetwork.ipynb
A series of Neural Network examples including the following:

1. A simple neural network implementation used to create a model based on the MNIST dataset to test and train. Accuracy is relatively stable at 0.9764, 97.64%.

2. A Convolutional Neural Network implementation given the same MNIST dataset for comparison of improved accuracy. Accuracy is relatively stable at 0.9912, 99.64%.

3. A Convolutional Neural Network implementation given the CIFAR10 dataset to test and train. Accuracy dropped in comparison to the previous implementation which is 0.8146, 81.46%.

4. Data Augmentation example using a Convolutional Neural Network the the MNIST dataset.

<a name="documentation"/>
## Documentation
The Official GitHub Repository of ImageAI is located [here](https://github.com/OlafenwaMoses/ImageAI). Read the documentation of ImageAI found [here](https://imageai.readthedocs.io/en/latest/index.html).
