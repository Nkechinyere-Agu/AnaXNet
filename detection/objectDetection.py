# You may need to restart your runtime prior to this, to let your installation take effect
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import torch, torchvision
import json
import os

# import some common detectron2 utilities
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

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

class detectronDetection():
    def __init__(self) -> None:
        self.filename = '/home/agun/mimic/dataset/VG/xray_coco_train.json'
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
        cfg.DATALOADER.NUM_WORKERS = 16
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        # Number of images per batch across all machines.
        cfg.SOLVER.IMS_PER_BATCH = 16
        cfg.SOLVER.BASE_LR = 1e-4  # pick a good LearningRate
        cfg.SOLVER.MAX_ITER = 50000  #No. of iterations   
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.organs) # No. of classes = [HINDI, ENGLISH, OTHER]
        cfg.TEST.EVAL_PERIOD = 25000 # No. of iterations after which the Validation Set is evaluated. 
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        return cfg

if __name__ == '__main__':
    detectron = detectronDetection()
    cfg = detectron.setup()
    detectron.registerDataset()
    trainer = CocoTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()