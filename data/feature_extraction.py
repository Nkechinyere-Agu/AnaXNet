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
from extraction import extract_feat


# image_root = "../dataset/VG/data/"
image_root = "/home/agun/mimic/dataset/VG/data/"
# json_file = "../dataset/VG/xray_coco_train.json"
thing_classes = ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
    "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
    "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
    "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette", "trachea"]

def setup():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 

    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes) # No. of classes = [HINDI, ENGLISH, OTHER]
    #Use the final weights generated after successful training for inference  
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    #Pass the validation dataset
    cfg.DATASETS.TEST = ("mimic_cxr_test", )
    return cfg

def get_board_dicts(imgdir):
    json_file = imgdir #Fetch the json file
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        filename = i["file_name"] 
        i["file_name"] = image_root + filename 
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYWH_ABS #Setting the required Box Mode
            j["category_id"] = int(j["category_id"])
    return dataset_dicts

def load_dict(imgdir):
    json_file = imgdir #Fetch the json file
    with open(json_file) as f:
        dataset_dicts = json.load(f)
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

def extract_features(raw_image, predictor):
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


def main(predictor):
    for batch in ["train", "valid", "test"]:
    # for batch in ["train"]:
        save_file_name = "../dataset/VG/xray_graph_{}.json".format(batch)
        save_file_name2 = "../dataset/VG/xray_image_{}.json".format(batch)
        train_dict = load_dict("../dataset/VG/xray_coco_{}.json".format(batch))
        dataset_dicts = get_board_dicts("../dataset/VG/xray_coco_{}.json".format(batch))
        board_metadata = MetadataCatalog.get("mimic_cxr_{}".format(batch))
        cat = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        # print(board_metadata)

        #iterate through the dataset to extract the features
        data = []
        count = 0
        image_data = []
        img_data = {}
        # for d in random.sample(dataset_dicts, 100):
        for d in dataset_dicts:
            
            count += 1
            print('{} Set: Processing file {} of {}'.format(str(batch), str(count), str(len(train_dict)) ))
            image = cv2.imread(d["file_name"])
            results = extract_features(image, predictor)
            
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
            # print(df)
            img_data[image_id] = df['features'].tolist()
            # image_data.append(img_data)
            
            for bbox, obj, probs, feature in zip(bboxes, objects, cls_prob, features):
                single_data = {}
                try:
                    things = annotations[obj]
                except:
                    print("Target for {} not in File {}".format(str(thing_classes[obj]), image_id))
                else:
                    annotation = annotations[obj]
                    single_data['image_id'] = image_id
                    # single_data['bbox'] = bbox.tolist()
                    single_data['objects'] = thing_classes[obj]
                    # single_data['cls_prob'] = str(probs)
                    single_data['features'] = feature.tolist()
                    # single_data['features_list'] = df['features'].tolist()
                    single_data['target'] = annotation

                    data.append(single_data)
                    # print(json.dumps(single_data))
                    # bbox_list.append(bbox.tolist())
                    # category.append(thing_classes[obj])
                    # cls_prob_list.append(probs)
                    # imageID.append(image_id)
                    # features_list.append(feature.tolist())
                    # target_list.append(annotation)
            # else:
            #     print('Too few boxes in {}'.format(image_id))
        with open(save_file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        with open(save_file_name2, 'w', encoding='utf-8') as f:
            json.dump(img_data, f, ensure_ascii=False, indent=4)
        # coco_data = pd.DataFrame(
        #         {'image_id': imageID,
        #         'bbox': bbox_list,
        #         'objects': category,
        #         'cls_prob': cls_prob_list,
        #         'features': features_list,
        #         'target': target_list
        #         })
        # coco_data.to_csv(save_file_name, sep='\t', index=False)
        # print(coco_data)

if __name__ == '__main__':
    cfg = setup()
    predictor = DefaultPredictor(cfg)
    main(predictor)