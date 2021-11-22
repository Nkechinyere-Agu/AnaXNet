import pandas as pd
import numpy as np
import os
import glob
import shutil
import json
import statistics
from PIL import Image
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import jaccard_score

class AdjacencyMatrices():
    def __init__(self) -> None:
        self.filename = '/home/agun/mimic/dataset/VG/xray_coco_test.json'
        self.outputdir = "/home/agun/mimic/dataset/VG/"
        self.diseaselist = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette',
        'pulmonary edema/hazy opacity', 'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']

        self.organs = ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
        "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
        "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
        "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette", "trachea"]

        print("Loading json data ...")
        f = open(str(self.filename),) 
        self.data = json.load(f)
        self.data_size = len(self.data)
        print(self.data_size)
        print("Done loading json data ...")
        

    '''
    The Similarity measure between each pair of anatomy objects A and B
    Jaccard similarity measure is used to measure the similarity between 
    each object, by measuring the average similarity over every disease class
    '''
    def anatomy(self):
        error = 1e-9
        anatomy_len = len(self.organs)
        row = self.organs
        column = self.organs
        adj_matrix = []

        for ind, B in enumerate(row):
            print("Processing {} from row {}".format(B, str(ind)))
            rows = np.zeros([len(self.organs)]) 
            for inde, A in enumerate(column):
                # print("Processing {} from column {}".format(A, str(inde)))
                AnB_count = 0
                B_count = 0
                row_counter = Counter()
                column_counter = Counter()
                a_val = []
                b_val = []
                p_anb = 0
                
                for img in self.data:                
                    ids = [self.organs[int(obj['category_id'])] for obj in img['annotations']]
                    aa = []
                    bb = []
                    if set(ids) == set(self.organs):
                        for relation in img['annotations']:
                            if int(relation['category_id']) == ind:
                                bb = relation['attributes']
                        for relations in img['annotations']:
                            if int(relations['category_id']) == inde:
                                aa = relations['attributes']
                        if np.count_nonzero(np.array(aa)) > 0 or np.count_nonzero(np.array(bb)) > 0:
                            b_val.append(bb)
                            a_val.append(aa)
                    else:
                        continue

                df_A = pd.DataFrame(a_val, columns=self.diseaselist)
                df_B = pd.DataFrame(b_val, columns=self.diseaselist)
                
                assert len(b_val) == len(a_val)
                
                if not df_A.empty:
                    jaccard_list = []
                    for disease in self.diseaselist:
                        jaccard = jaccard_score(df_B[disease], df_A[disease], average='macro')
                        jaccard_list.append(jaccard)
                    p_anb = statistics.mean(jaccard_list)

                if ind == inde:
                    p_anb = 1
                if p_anb > 0.5:
                    p_anb = 1
                else:
                    p_anb = 0
                
                rows[inde] = p_anb
            adj_matrix.append(rows.tolist())

        
        df = pd.DataFrame(adj_matrix, columns=self.organs)
        # print(df)
        filename = os.path.join(self.outputdir, 'anatomy_matrix.csv')
        df.to_csv(filename, sep='\t', index=False)
        return df

    '''
    The Conditional Probability of A (disease row) given B (disease Column)
    P(A|B) = P(AnB)/P(B)
    '''
    def findings(self):
        filename = os.path.join(self.outputdir, 'findings_matrix.csv')
        error = 1e-9
        row = self.diseaselist
        column = self.diseaselist
        adj_matrix = []

        for ind, B in enumerate(row):
            print("Processing {} from row {}".format(B, str(ind)))
            rows = np.zeros([len(self.diseaselist)]) 
            for inde, A in enumerate(column):
                # print("Processing {} from column {}".format(A, str(inde)))
                AnB_count = 0
                B_count = 0
                for img in self.data:
                    for relation in img['annotations']:
                        if relation['attributes'][ind] == 1:
                            B_count += 1 
                        if (relation['attributes'][inde] == 1) and (relation['attributes'][ind] == 1):
                            AnB_count += 1 
                
                p_anb = AnB_count/self.data_size
                p_b = B_count/self.data_size
                a_given_b = p_anb / (p_b + error)
                if a_given_b > 0.4:
                    a_given_b = 1
                else:
                    a_given_b = 0
                rows[inde] = a_given_b
            adj_matrix.append(rows.tolist())

        print(adj_matrix)
        df = pd.DataFrame(adj_matrix, columns=self.diseaselist)
        df.to_csv(filename, sep='\t', index=False)
        return df

if __name__ == '__main__':
    matrix = AdjacencyMatrices()
    anatomy = matrix.anatomy()
    findings = matrix.findings()