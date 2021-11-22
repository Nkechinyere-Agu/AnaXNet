import os
import pandas as pd
import json

rootdir = '/home/agun/mimic/dataset/VG/FeatureData/'
outputdir = "/home/agun/mimic/dataset/VG/"
files = os.listdir(rootdir)

coco_data = pd.DataFrame(
                    {'image_id': files
                    })
coco_data.to_csv(os.path.join(outputdir, "new_train.csv"), sep='\t', index=False)
print(coco_data.head(5))