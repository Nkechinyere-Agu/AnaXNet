from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.structures import BoxMode

# import some common libraries
import numpy as np
import pandas as pd
import cv2
import random
import torch, torchvision
import json
import os

class visualizeDetection():
    def __init__(self) -> None:
        self.filename = '/home/agun/mimic/dataset/VG/data/32a8f331-711282df-420eca1a-f5e8531e-02bc5db2.jpg'
        self.outputdir = "/home/agun/mimic/dataset/VG/"
        self.image_root = "/home/agun/mimic/dataset/VG/data/"
        self.diseaselist = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette',
        'pulmonary edema/hazy opacity', 'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']

        self.organs = ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
        "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
        "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
        "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette", "trachea"]

    def get_board_dicts(self, imgdir):
        json_file = imgdir #Fetch the json file
        with open(json_file) as f:
            dataset_dicts = json.load(f)
        for i in dataset_dicts:
            filename = i["file_name"] 
            i["file_name"] = self.image_root + filename 
            for j in i["annotations"]:
                j["bbox_mode"] = BoxMode.XYWH_ABS #Setting the required Box Mode
                j["category_id"] = int(j["category_id"])
        return dataset_dicts

    def registerDataset():
        #Registering the Dataset
        for d in ["train", "test", "valid"]:
            filename = os.path.join(self.outputdir, "xray_coco_{}.json".format(d))
            DatasetCatalog.register("mimic_cxr_{}".format(d), lambda d=d: get_board_dicts(filename))
            MetadataCatalog.get("mimic_cxr_{}".format(d)).set(thing_classes=self.organs)
        board_metadata = MetadataCatalog.get("mimic_cxr_train")
        print(board_metadata)

    def setup(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
        #Passing the Train and Validation sets
        cfg.DATASETS.TRAIN = ("mimic_cxr_train",)
        cfg.DATASETS.TEST = ("mimic_cxr_test",)
        # Number of data loading threads
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        # Number of images per batch across all machines.
        cfg.SOLVER.IMS_PER_BATCH = 16
        cfg.SOLVER.BASE_LR = 1e-4  # pick a good LearningRate
        cfg.SOLVER.MAX_ITER = 50000  #No. of iterations   
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes) # No. of classes = [HINDI, ENGLISH, OTHER]
        cfg.TEST.EVAL_PERIOD = 10000 # No. of iterations after which the Validation Set is evaluated. 
        #Use the final weights generated after successful training for inference  
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
        #Pass the validation dataset
        cfg.DATASETS.TEST = ("mimic_cxr_test", )
        return cfg

    def visualizeGT(self):
        visualizer = Visualizer(im[:, :, ::-1], metadata=board_metadata)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite("train_{}.png".format(str(count)), vis.get_image()[:, :, ::-1])
        print(outputs["instances"])
        cv2.imwrite("train_{}.png".format(str(count)), vis.get_image()[:, :, ::-1])
        cv2_imshow(v.get_image()[:, :, ::-1])

    def visualizeDetection(self):
        count = 0 
        im = cv2.imread(self.filename)
        outputs = predictor(im)
        print(outputs)
        #float matrix of Nx4. Each row is (x1, y1, x2, y2).
        boxes = outputs["instances"].pred_boxes.tensor.to('cpu').numpy()
        objects = outputs["instances"].pred_classes.to('cpu').numpy()
        print(boxes)
        lst = ["right costophrenic angle", "cardiac silhouette"]
                #red, green, blue,
        clrs = [[255,0,0],[0,255,0], [0,0,255], [238,130,238], [255,165,0]]
        for box, obj in zip(boxes,objects):
                anatomy = thing_classes[obj]
                if anatomy in lst:
                    image = cv2.imread(self.filename)
                    start_point = (int(box[0]), int(box[1]))
                    end_point = (int(box[2]), int(box[3]))
                    # color = list(np.random.random(size=3) * 256)
                    color = clrs[lst.index(anatomy)]
                    image = cv2.rectangle(image, start_point, end_point, color, 10)
                    im = cv2.rectangle(im, start_point, end_point, color, 10)
                    crop = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    cv2.imwrite("./imgs/{}.png".format(str(anatomy)), crop)  
        cv2.imwrite("./imgs/viz2.png", im) 

if __name__ == '__main__':
    viz = visualizeDetection()
    cfg = viz.setup()
    predictor = DefaultPredictor(cfg)
    viz.registerDataset()
    dataset_dicts = viz.get_board_dicts("../dataset/VG/xray_coco_test.json")
    viz.visualizeDetection()


 


