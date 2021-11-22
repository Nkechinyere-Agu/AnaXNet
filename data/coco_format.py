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

class coco_format():
    def __init__(self) -> None:
        
        self.organs = ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
            "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
            "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
            "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette", "trachea"]
        
        self.diseaselist = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette',
        'pulmonary edema/hazy opacity', 'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']
       
        self.filename = './../data/VQA_RAD.json'
        self.splits = '/home/agun/mimic/dataset/splits/'
        self.rootdir = '/home/agun/mimic/dataset/VG/scene_graph/'
        self.imageroot = '/home/agun/mimic/dataset/VG/data/'
        self.reportroot = '/home/agun/mimic/dataset/VG/MIMIC-CXR-Reports/'
        self.path = '/home/agun/mimic/dataset/VG/ViTdata'
        self.outputdir = "/home/agun/mimic/dataset/VG/"
        
        print('Disease length is {}'.format(str(len(self.diseaselist))))

        if not os.path.exists(self.path):
            # Create a new directory because it does not exist 
            os.makedirs(self.path)
        
        #obtain list of train, valid, test files
        self.my_data = {}
        self.my_data['train'] = read_csv(self.splits, 'train.csv') #filelist[:trainlen]
        self.my_data['valid'] = read_csv(self.splits, 'valid.csv') #filelist[:trainlen]
        self.my_data["test"] = read_csv(self.splits, 'test.csv') #filelist[trainlen:]

    def generate_data(self):  
        for keys in self.my_data.keys():
            images = []
            categories = []
            count = 0
            coco_data = {}
            imagesID = []
            labels = []
            for file in self.my_data[keys]:
                image_json = {}
                annotations = []
                if 'DS_Store' not in file:
                    count += 1
                    hasattributes = 0
                    print('{} Set: Processing file {}'.format(str(keys), str(count)))
                    myfile = file + '_SceneGraph.json'
                    filename = os.path.join(self.rootdir, myfile)
                    # filename = os.path.join(rootdir, file)
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
                        for objects in data['objects']:
                            hasdisease = 1
                            # objectID = attribute['object_id']
                            row = np.zeros([len(self.diseaselist)])
                            if objects['object_id'].split('_')[1] in self.organs:
                                annotation_json = {}
                                for attribute in data['attributes']:
                                    if attribute['object_id'] == objects['object_id']:
                                        for diseases in attribute['attributes']:
                                            for disease in diseases:                                                
                                                if disease.split('|')[2] in self.diseaselist:
                                                    hasdisease = 1
                                                    hasattributes = 1
                                                    class_index = self.diseaselist.index(disease.split('|')[2])
                                                    if disease.split('|')[1] == 'yes':
                                                        row[class_index] = int(1)
                                                    else:
                                                        row[class_index] = int(0)
                                    annotation_json['id'] = objects['object_id']
                                    annotation_json['category_id'] = self.organs.index(objects['object_id'].split('_')[1])
                                    annotation_json['iscrowd'] = 0
                                    annotation_json["bbox_mode"] = 1
                                    annotation_json['image_id'] = data['image_id']
                                    int_row = row.astype(int)
                                    annotation_json['attributes'] = int_row.tolist()
                                    annotation_json['bbox'] = [objects['original_x1'], objects['original_y1'],
                                    objects['original_width'], objects['original_height']]
                                    annotations.append(annotation_json)
                
                        image_json['image_id'] = data['image_id']
                        myfile = str(data['image_id']) + '.jpg'
                        image_json['file_name'] = myfile
                        # path = os.path.join(self.imageroot, myfile) #"./VG/data/" + myfile
                        # im = Image.open(path)
                        # image_json['width'], image_json['height'] = im.size
                        image_json['annotations'] = annotations
                        # # imagesID.append(myfile)
                        images.append(image_json)
        
            save_file_name = "xray_coco_{}.json".format(str(keys))
            filename = os.path.join(self.outputdir, save_file_name)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(images, f, ensure_ascii=False, indent=4)            
        
x_ray_coco = coco_format()
x_ray_coco.generate_data()