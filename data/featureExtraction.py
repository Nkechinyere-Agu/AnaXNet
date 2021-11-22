from torch import nn

import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler


# import some common libraries
import numpy as np
import cv2
import random
import torch, torchvision
import json
import io
import pandas as pd
import PIL.Image
import os

class FeatureExtraction():
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
        self.path = '/home/agun/mimic/dataset/VG/FeatureData/'
        self.outputdir = "/home/agun/mimic/dataset/VG/"
        

    def setup(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 

        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.organs) # No. of classes = [HINDI, ENGLISH, OTHER]
        #Use the final weights generated after successful training for inference  
        path = '/home/agun/mimic/detection/output/'
        cfg.MODEL.WEIGHTS = os.path.join(path, "model_final.pth")

        # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
        #Pass the validation dataset
        cfg.DATASETS.TEST = ("mimic_cxr_test", )
        return cfg

    def get_board_dicts(self,imgdir):
        json_file = imgdir #Fetch the json file
        with open(json_file) as f:
            dataset_dicts = json.load(f)
            # print(len(dataset_dicts))
        for i in dataset_dicts:
            filename = i["file_name"] 
            i["file_name"] = self.imageroot + filename 
            for j in i["annotations"]:
                j["bbox_mode"] = BoxMode.XYWH_ABS #Setting the required Box Mode
                j["category_id"] = int(j["category_id"])
        return dataset_dicts

    def load_dict(self,imgdir):
        json_file = imgdir #Fetch the json file
        with open(json_file) as f:
            dataset_dicts = json.load(f)
            # print(len(dataset_dicts))
        data_dict = {}
        for images in dataset_dicts:
            image_id = images["image_id"]
            image_dict = {}
            for annotations in images["annotations"]:
                category_id = annotations["category_id"]
                attributes = annotations["attributes"]
                image_dict[category_id] = attributes
            data_dict[image_id] = image_dict
        
        return data_dict

    def extractFeatures(self,raw_image, predictor):
        image = raw_image
        outputs = predictor(image)
        boxes = outputs["instances"].pred_boxes #.numpy()
        given_boxes=boxes 
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        with torch.no_grad():
            # Preprocessing
            images = predictor.model.preprocess_image(inputs)  # don't forget to preprocess
            # Run Backbone Res1-Res4
            features = predictor.model.backbone(images.tensor)  # set of cnn features
            
            # Run RoI head for each proposal (RoI Pooling + Res5)
            boxes = given_boxes.clone()
            proposal_boxes = [boxes]
            features = [features[f] for f in predictor.model.roi_heads.in_features]

            #get proposed boxes + rois + features + predictions
            proposal_rois = predictor.model.roi_heads.box_pooler(features, proposal_boxes)
            box_features = predictor.model.roi_heads.box_head(proposal_rois)
            # predictions = predictor.model.roi_heads.box_predictor(box_features)#found here: https://detectron2.readthedocs.io/_modules/detectron2/modeling/roi_heads/roi_heads.html

            #['bbox', 'num_boxes', 'objects', 'image_width', 'image_height', 'cls_prob', 'image_id', 'features']
            result = {
                'bbox': given_boxes.tensor.to('cpu').numpy(),
                'num_boxes' : outputs["instances"].pred_classes.to('cpu').numpy().shape[0],
                'objects' : outputs["instances"].pred_classes.to('cpu').numpy(),
                #'image_height': img.image_sizes[0][0],
                #'image_width': img.image_sizes[0][1],
                'cls_prob': np.asarray(outputs["instances"].scores.to('cpu')), #needs to turn into vector!!!!!!!!!!
                'features': box_features.to('cpu').detach().numpy()
            }
            
            # return pred_instances, pred_inds, box_features
            return result


    def main(self,predictor):
        for batch in ["train", "valid", "test"]:
            save_file_name = "/home/agun/mimic/dataset/VG/xray_graph_{}.json".format(batch)
            save_file_name2 = "/home/agun/mimic/dataset/VG/xray_image_{}.json".format(batch)
            train_dict = FeatureExtraction().load_dict("/home/agun/mimic/dataset/VG/xray_coco_{}.json".format(batch))
            dataset_dicts = FeatureExtraction().get_board_dicts("/home/agun/mimic/dataset/VG/xray_coco_{}.json".format(batch))
            board_metadata = MetadataCatalog.get("mimic_cxr_{}".format(batch))
            cat = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            # print(board_metadata)

            #iterate through the dataset to extract the features
            data = []
            count = 0
            image_data = []
            img_data = {}
            # print(len(dataset_dicts))
            # for d in random.sample(dataset_dicts, 1):
            for d in dataset_dicts:
                count += 1
                print('{} Set: Processing file {} of {}'.format(str(batch), str(count), str(len(train_dict)) ))
                image = cv2.imread(d["file_name"])
                results = FeatureExtraction().extractFeatures(image, predictor)
                
                # print(results)
                # if results['num_boxes'] == 18:
                bboxes = results['bbox']
                objects = results['objects']
                cls_prob = results['cls_prob']
                features = results['features']
                image_id = d["image_id"]
                annotations = train_dict[image_id]
                diff = list(set(cat).difference(objects))
                
                df = pd.DataFrame(
                    {
                        'obj': objects,
                        'features': features.tolist()
                    }
                )
                
                if diff:
                    feat = np.zeros((len(diff), 1024))
                    df2 = pd.DataFrame(
                        {
                            'obj': diff,
                            'features': feat.tolist()
                        }
                    )
                    frames = [df, df2]
                    df = pd.concat(frames)
                df = df.sort_values('obj')
                df = df.drop_duplicates(subset='obj', keep="first")
                
                try:
                    assert len(df) == 18
                except:
                    print(df)
                
                image_data.append(image_id)
                single_data = {}
                single_data['image_id'] = image_id
                
                objs,target, boxes = [],[],[]
                
                 
                for bbox, obj, probs, feature in zip(bboxes, objects, cls_prob, features):
                    
                    try:
                        things = annotations[obj]
                    except:
                        print("Target for {} not in File {}".format(str(self.organs[obj]), image_id))
                        annotation = np.zeros([len(self.diseaselist)])
                        annotation = annotation.astype(int)
                        target.append(annotation.tolist())
                    else:
                        annotation = annotations[obj]
                        objs.append(self.organs[obj])
                        target.append(annotation)
                        boxes.append(bbox.tolist())
                
                single_data['bbox'] = boxes
                single_data['objects'] = objs
                single_data['target'] = target
                single_data['features'] = df['features'].tolist()
                # print(single_data)
                save_file_name = "{}.json".format(str(image_id))
                filename = os.path.join(self.path, save_file_name)        
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(single_data, f, ensure_ascii=False, indent=4)
            
            coco_data = pd.DataFrame(
                    {'image_id': image_data
                    })
            coco_data.to_csv(os.path.join(self.outputdir, "new_{}.csv".format(str(batch))), sep='\t', index=False)
            print(coco_data.head(5))

if __name__ == '__main__':
    features = FeatureExtraction()
    cfg = features.setup()
    predictor = DefaultPredictor(cfg)
    features.main(predictor)