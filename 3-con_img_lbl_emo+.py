# from numpy import array, moveaxis, indices, dstack
from PIL import Image
import numpy as np
import sys
import os
import csv
import pandas as pd

csvpath_in = 'face_emotion_img_lbl+.csv'
csvpath_out = 'face_emotion_img_lbl_emo+.csv'
import ipdb;ipdb.set_trace()

df = pd.read_csv(csvpath_in)
df2 = df.copy()
df_cols = df.head(0)

emos = df[df.columns[2:9]]

import ipdb; ipdb.set_trace()
    # majority of voting
emoArr = np.asarray(emos)
mxvote = np.max(emoArr, axis=1)
mxvotecat = np.argmax(emoArr,axis=1)
df2['emotion'] = mxvotecat
import ipdb;ipdb.set_trace()
df2.to_csv(csvpath_out , index=False, header=True)
import ipdb;ipdb.set_trace()

