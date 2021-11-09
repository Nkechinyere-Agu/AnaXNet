import pandas as pd
import numpy as np
import os
import glob
import shutil
import json
# import ujson
from PIL import Image
import random


# read_data()
class IteratorAsList(list):
    def __init__(self, it):
        self.it = it
    def __iter__(self):
        return iter(self.it)
    def __len__(self):
        return 1    

def read_csv(direct, split):
    path = direct + split
    data = pd.read_csv(path)

    return data['dicom_id'].tolist()

def coco_format():
    splits = './splits/'
    # rootdir = './scene_graph/'
    rootdir = '/Users/nneka/Downloads/subset-2/test/'
    image_root = "/home/agun/mimic/dataset/VG/data/"
    
    # diseaselist = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette',
    # 'pulmonary edema/hazy opacity', 'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']
    
    diseaselist = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette',
    'pulmonary edema/hazy opacity', 'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']
    
    pred_list = ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
    "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
    "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
    "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette", "trachea"]
    
    #split into training and validation

    # filelist = [f for f in os.listdir(rootdir) if f.endswith('.json')]
    # random.shuffle(filelist)
    # trainlen = int(0.8 * len(filelist))
    # print("Length of training data is {}".format(str(trainlen)))
    # print("Length of test data is {}".format(str(len(filelist) - trainlen)))
    my_data = {}
    my_data['train'] = read_csv(splits, 'train.csv') #filelist[:trainlen]
    my_data['valid'] = read_csv(splits, 'valid.csv') #filelist[:trainlen]
    my_data["test"] = read_csv(splits, 'test.csv') #filelist[trainlen:]
    # dataset = [test_data, train_data]
    
    # for keys in my_data.keys():
    images = []
    for file in os.listdir(rootdir):
        categories = []
        
        count = 0
        coco_data = {}
        imagesID = []
        labels = []
    # for file in my_data[keys]:
        # for file in files:
        image_json = {}
        annotations = []
        # f1 = open('image_data.json',) 
        # imagedata = json.load(f1)
        if 'DS_Store' not in file:
            # if 'json' in file:
            count += 1
            hasattributes = 0
            # print('{} Set: Processing file {}'.format(str(keys), str(count)))
            myfile = file + '_SceneGraph.json'
            # filename = os.path.join(rootdir, myfile)
            filename = os.path.join(rootdir, file)
            try:
                f = open(str(filename),) 
            except FileNotFoundError:
                print('{} not in directory'.format(myfile))
            else:
                data = json.load(f)
                imageID = data['image_id']

                ids = [obj['object_id'] for obj in data['objects']]
                ignore = 0
                hasdisease = 1
                for attribute in data['attributes']:
                    objectID = attribute['object_id']
                    if (objectID not in ids):
                        print("Faulty JSON")
                        ignore = 0
                if ignore == 0:
                    # row = np.zeros([len(diseaselist)])
                    for objects in data['objects']:
                        hasdisease = 1
                        objectID = attribute['object_id']
                        row = np.zeros([len(diseaselist)])
                        if objects['object_id'].split('_')[1] in pred_list:
                            annotation_json = {}
                            for attribute in data['attributes']:
                                # print("List is not empty")
                                if attribute['object_id'] == objects['object_id']:
                                    # for diseases,labels in zip(attribute['attributes'], attribute['contexts']):
                                    # att_list = attribute['attributes']
                                    # diseases = att_list[len(att_list)-1]
                                    for diseases in attribute['attributes']:
                                        for disease in diseases:                                                
                                            if disease.split('|')[2] in diseaselist:
                                                hasdisease = 1
                                                hasattributes = 1
                                                class_index = diseaselist.index(disease.split('|')[2])
                                                if disease.split('|')[1] == 'yes':
                                                    row[class_index] = int(1)
                                                else:
                                                    row[class_index] = int(0)
                            annotation_json['id'] = objects['object_id']
                            annotation_json['category_id'] = pred_list.index(objects['object_id'].split('_')[1])
                            annotation_json['iscrowd'] = 0
                            annotation_json["bbox_mode"] = 1
                            annotation_json['image_id'] = data['image_id']
                            int_row = row.astype(int)
                            annotation_json['attributes'] = int_row.tolist()
                            # for objects in data['objects']:
                            #     if objectID == objects['object_id']:
                            annotation_json['bbox'] = [objects['original_x1'], objects['original_y1'],
                            objects['original_width'], objects['original_height']]
                            annotations.append(annotation_json)
                            # labels.append(int_row)
                            # imagesID.append(objects['object_id'])
        
                # if hasattributes == 1:
                image_json['image_id'] = data['image_id']
                myfile = str(data['image_id']) + '.jpg'
                image_json['file_name'] = myfile
                # path = "./VG/data/" + myfile
                # im = Image.open(path)
                # image_json['width'], image_json['height'] = im.size
                image_json['annotations'] = annotations
                # imagesID.append(myfile)
                # labels.append(int_row)
                # image_json['width'] = imagedata[imageID][0]
                # image_json['height'] = imagedata[imageID][1]
                images.append(image_json)
    
    for i, objs in enumerate(pred_list):
        categories_json = {}
        categories_json['id'] = i
        categories_json['name'] = objs
        categories.append(categories_json)
    
    coco_data['images'] = images
    # coco_data['annotations'] = annotations
    coco_data['categories'] = categories

    # df = pd.DataFrame(labels, columns = diseaselist)
    # df['image_id'] = imagesID
    save_file_name = "test.json"#'./VG/xray_coco_baseline_vis_{}.csv'.format(str("keys"))
    with open(save_file_name, 'w', encoding='utf-8') as f:
        json.dump(images, f, ensure_ascii=False, indent=4)

    # df.to_csv("./save_file_name.csv", sep='\t', index=False)
        
        
coco_format()