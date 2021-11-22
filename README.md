# Anatomy Aware Chest X-ray findings classification network
Radiologists usually observe anatomical regions of chest X-ray images as well as the overall image before making a decision. However, most existing deep learning models only look at the entire X-ray image for classification, failing to utilize important anatomical information. In this paper, we propose a novel multi-label chest X-ray classification model that accurately classifies the image finding and also localizes the findings to their correct anatomical regions. Specifically, our model consists of two modules, the detection module and the anatomical dependency module. The latter utilizes graph convolutional networks, which enable our model to learn not only the label dependency but also the relationship between the anatomical regions in the chest X-ray. We further utilize a method to efficiently create an adjacency matrix for the anatomical regions using the correlation of the label across the different regions. Detailed experiments and analysis of our results show the effectiveness of our method when compared to the current state-of-the-art multi-label chest X-ray image classification methods while also providing accurate location information.

![alt text](https://github.com/Nkechinyere-Agu/AnaXNet/blob/master/imgs/network.jpg?raw=true)


### Dependencies
* python 3.5+
* pytorch 1.0+
* torchvision
* numpy
* pandas
* sklearn
* matplotlib
* tensorboardX
* tqdm

### Dataset
Requires MIMIC-CXR Dataset and Chest Imagenome dataset which can both be found on https://physionet.org

to generate the format for object detection from the chest imagenome scene graphs run:
```
python ./data/coco_format.py
```

To train Detectron 2 Faster R-CNN on the coco format dataset, run:
```
python ./detection/objectDetection.py
```

To extract the Anatomical region features from the Faster R-CNN model run:
```
python ./data/featureExtraction.py
```

### Usage
To train a model using default batch size, learning:
```
python train.py  
```

### Results

