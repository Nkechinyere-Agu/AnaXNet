import pandas as pd
import numpy as np
import shutil
import json
from PIL import Image
import statistics
import matplotlib.pyplot as plt
from collections import Counter

def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

def word2vec():
    filepath = './embedding/glove.6B.300d.txt'
    file_name = './VG/word2vec.csv'
    diseases = ["lung opacity", "pleural effusion", "atelectasis", "enlarged cardiac silhouette", "pulmonary edema", 
            "pneumothorax", "consolidation" , "heart failure", "pneumonia"]

    gloveModel = loadGloveModel(filepath)

    word2vec = []
    for disease in diseases:
        words = disease.split()
        vectorList = []
        for word in words:
            vector = gloveModel[word]
            vectorList.append(vector)
        disease_vector = np.average(vectorList, axis=0)
        word2vec.append(disease_vector)

    a = np.matrix(word2vec)
    df = pd.DataFrame(a)
    df.to_csv(file_name, sep='\t', index=False)

def copy_files():
    rootdir = '../physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
    # rootdir = './VG/data/'
    extensions = [".jpg", ".jpeg", ".png"]
    count = 0
    image_json = {}
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if 'DS_Store' not in file:
                if 'jpg' in file:
                    count += 1
                    print('Processing file {}'.format(str(count)))
                    filename = os.path.join(subdir, file)
                    path = "./VG/data/" + file
                    shutil.copy2(filename, path)


def attributeCounter():
    print("Loading json data ...")
    filename = './VG/xray_coco_test.json'
    f = open(str(filename),) 
    im_data = json.load(f)
    print(len(im_data))
    print("Done loading json data ...")

    diseaselist = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette',
    'pulmonary edema/hazy opacity', 'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']

    pred_list = ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
    "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
    "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
    "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette"]
    
    token_counter = Counter()
    object_counter = Counter()

    for img in im_data:
        for relation in img['annotations']:
            for i, attributes in enumerate(relation['attributes']):
                # print(attributes)
                if int(attributes) == 1:
                    attribute = diseaselist[i]
                    token_counter.update([attribute])
                    # object_counter.update([objects])

    
    print(token_counter)