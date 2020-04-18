import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

csvpath_fold = './emotion_folds_mid+/'
csvpath_in = 'face_emotion_img_lbl_emo+.csv'
csvpath_out = 'face_emotion_img_lbl_emo_use_fold+'
df = pd.read_csv(csvpath_in)
df_cols = df.head(0)
usage = pd.Series([])
imnames = df['image_name']
emocatnums = df["emotion"]
kf = KFold(n_splits=10, random_state=0)
import ipdb;ipdb.set_trace()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=1)

for train_index, test_index in skf.split(imnames, emocatnums):
    print("TRAIN:", train_index, "TEST:", test_index)
    emocatnums = emocatnums#.tolist()  
    traincats = [emocatnums[i] for i in train_index]  
    [traincats.count(i)  for i in range(0,7)]
    [print('\n num of cases from traincat : ', str(i), 'is: ' , str(traincats.count(i))) for i in range(0,7)] 
    testcats = [emocatnums[i] for i in test_index]  
    [testcats.count(i)  for i in range(0,7)]
    [print('\n num of cases from testcat : ', str(i), 'is: ' , str(testcats.count(i))) for i in range(0,7)]

cntr = 0
for train_index, test_index in kf.split(imnames, emocatnums):
    df_fold = df.copy()
    cntr+=1
    for idx in train_index:
        usage[idx] = "Training"
    for idx in test_index:
        usage[idx] = "Test"
    df_fold['Usage'] = usage

    df_fold.to_csv(csvpath_fold + csvpath_out + str(cntr) + '.csv', index=False, header=True)
