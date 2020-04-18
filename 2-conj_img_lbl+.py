# from numpy import array, moveaxis, indices, dstack
from PIL import Image
import numpy as np
import sys
import os
import csv
import pandas as pd

def createFileList(myDir, format):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

imgList_out = './emo_data_processed/'
csvpath_out = 'face_emotion_img_lbl+.csv'

imgList_in = createFileList('/data/zdata/emo_data_raw/','.jpg')


import ipdb;ipdb.set_trace()


csvpath_in = './face_emotions_prune2.csv'
df = pd.read_csv(csvpath_in)
df_cols = df.head(0)
pixels = pd.Series([])
imnames = df['image_name']
imgnms = imnames.tolist()
# emotions = df[['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']]
emos = df[df.columns[1:8]]
df_img_lbl = pd.DataFrame({'image_name': [], 'pixels': pd.DataFrame(), 'Angry': [],
                               'Disgust': [], 'Fear': [], 'Happy': [], 'Sad': [], 'Surprise': [], 'Neutral': []})
for file in imgList_in:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    img_name = file.split('/')[-1]
    if img_name in imgnms : 
	    imidx = imgnms.index(img_name)
	    
	 
	    img_file_res = img_file.resize((48, 48))
	    img_grey_res = img_file_res.convert('L')
	    img_grey_res.save(imgList_out + img_name )
	    # img_grey_res.show()
	    value_asl_res = np.asarray(img_grey_res.getdata(), dtype=np.int)

	    print(value_asl_res)
	    pixel_vals = value_asl_res.flatten()
	    pvs = ' '.join(map(str, pixel_vals))
	    # print(pixels)
	    df_img_lbl = df_img_lbl.append({'image_name': img_name, 'pixels': pvs , 'Angry': emos['Angry'][imidx],
	                          'Disgust':emos['Disgust'][imidx], 'Fear':emos['Fear'][imidx],
	                           'Happy':emos['Happy'][imidx], 'Sad':emos['Sad'][imidx],
	                           'Surprise':emos['Surprise'][imidx], 'Neutral':emos['Neutral'][imidx]}, ignore_index=True)

df_img_lbl.to_csv( csvpath_out, index=False, header=True)



