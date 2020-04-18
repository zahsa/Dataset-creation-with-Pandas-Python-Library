import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
csvpath = 'mid_crops_emotions_v2.csv'
csvpath_out1 = homepath + 'face_emotions_prune1+.csv'
csvpath_out2 = homepath + 'face_emotions_prune2+.csv'
csvpath_out3 = homepath + 'face_emotions_prune3.csv'


imname = []; emo_cat = []; emo_arous = []
df = pd.read_csv(csvpath, index_col = 0)
df_org = df.copy()


df_processed = pd.DataFrame({'image_name' : [] , 'Angry' : [], 'Disgust' : [],
                             'Fear': [] , 'Happy':[], 'Sad':[], 'Surprise':[],
                             'Neutral': []})

import ipdb;ipdb.set_trace()
cntr = 0
nval = 0
emos_keys=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

for col in df.columns:
    if "happiness" in col:
       df.drop(columns=col,inplace=True)
       print('column removed')


import ipdb;ipdb.set_trace()
print('prune1')
df.to_csv(csvpath_out1 , index = False, header=True)

for imidx, imattr in df.iterrows():
    cntr+=1
    impath = imattr[0]
    imgname = impath.split('/')[-1]
    imgemo_cats = [imattr[1],imattr[2],imattr[3],imattr[4],imattr[5],imattr[6],imattr[7],imattr[8]]

    if "valid" in imgemo_cats:
        # print(raw_attr)
        nval+=1
        print(nval)
        df.drop(index=imidx, inplace=True)
    else:
        imgemo_votes = {}
        for emo in emos_keys:
            imgemo_votes[emo] = imgemo_cats.count(emo)
        df_processed = df_processed.append({'image_name': imgname, "Angry" : int(imgemo_votes["Angry"]) ,
                        "Disgust" : int(imgemo_votes["Disgust"]),"Fear" : int(imgemo_votes["Fear"]),
                        "Happy" : int(imgemo_votes["Happy"]),"Sad" :int(imgemo_votes["Sad"]) ,
                        "Surprise": int(imgemo_votes["Surprise"]),"Neutral":int(imgemo_votes["Neutral"])}, ignore_index=True)
        emo_cat.append(imgemo_cats)
        imname.append(imgname)
import ipdb;ipdb.set_trace()
print('prune 2')
import ipdb;ipdb.set_trace()
df_processed.to_csv(csvpath_out2 , index = False, header=True)
print(cntr)

