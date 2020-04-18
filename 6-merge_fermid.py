import pandas as pd

csvpath_midas = 'face_emotion_img_lbl_emo+.csv'
csvpath_fer = './data/fer2013.csv'


df_midas = pd.read_csv(csvpath_midas)
df_fer = pd.read_csv(csvpath_fer)
df_fer_mid = df_fer.copy()
import ipdb;ipdb.set_trace()

emo_fer = df_fer['emotion']#.tolist()
emo_mid = df_midas['emotion']#.tolist()
use_fer = df_fer['Usage']
use_mid = ['Training']*len(emo_mid)
#emo_fer_mid = emo_fer + emo_mid
#emo_fer_mid = emo_fer_mid.tolist()

pxl_fer = df_fer['pixels']
pxl_mid = df_midas['pixels']

data_fer = {'emotion': emo_fer,'pixels':pxl_fer, 'Usage': use_fer }
df_new = pd.DataFrame(data = data_fer)
data_mid = {'emotion': emo_mid,'pixels':pxl_mid, 'Usage': use_mid }
df_new = df_new.append(pd.DataFrame(data_mid))

import ipdb;ipdb.set_trace()
df_new.to_csv('fermid+.csv', index=False, header=True)

