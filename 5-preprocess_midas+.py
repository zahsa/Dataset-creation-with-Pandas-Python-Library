# create data and label for FER2013
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
import h5py

homepath = './emotion_folds_mid+/'
for idx in range(1,11):

    file = homepath + 'face_emotion_img_lbl_emo_use_fold+' + str(idx) + '.csv'
    # Creat the list to store the data and label information
    Training_x = []
    Training_y = []
    Test_x = []
    Test_y = []

    foldname = 'midasdata_fold+'+ str(idx) + '.h5'
    datapath = os.path.join('middata',foldname)
    if not os.path.exists(os.path.dirname(datapath)):
        os.makedirs(os.path.dirname(datapath))

    # import ipdb;ipdb.set_trace()
    with open(file,'r') as csvin:
        data=csv.reader(csvin)
        rowtrn = 0;rowtst = 0
        for row in data:
            if row[-1] == 'Training':
                rowtrn +=1
                print('Train: ',rowtrn)
                temp_list = []
                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)
                Training_y.append(int(row[9]))
                Training_x.append(I.tolist())
                print(np.shape(Training_x))
            # import ipdb;ipdb.set_trace()
            if row[-1] == "Test" :
                rowtst +=1
                print('Test: ', rowtst)
                temp_list = []
                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)
                Test_y.append(int(row[9]))
                Test_x.append(I.tolist())
                print(np.shape(Test_x))
                # import ipdb;ipdb.set_trace()
            # import ipdb;ipdb.set_trace()

    print(np.shape(Training_x))
    print(np.shape(Test_x))
    print('before writing')
    import ipdb;ipdb.set_trace()
    datafile = h5py.File(datapath, 'w')
    datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
    datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
    datafile.create_dataset("Test_pixel", dtype = 'uint8', data=Test_x)
    datafile.create_dataset("Test_label", dtype = 'int64', data=Test_y)


    datafile.close()

    print("Save data finish!!!")
