import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import cv2
import numpy as np
import glob
from os import listdir
from numpy import asarray
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization, concatenate, Subtract
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Add, Activation, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU device 3 (indexing starts from 0)

#### load all images in a directory into memory.
from PIL import Image
import re

resolutions = [(224, 224), (128, 128), (256, 256), (64, 64)]
dilation_rates = [(1, 1), (2, 2), (4, 4), (8, 8)]
# resolutions = [(224, 224), (128, 128)]
# dilation_rates = [(2, 2), (1, 1)]

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s   )]
     
    

def load_images(path, size = (224,224)):  
    data_list = list()# enumerate filenames in directory, assume all are images 
    for filename in sorted(os.listdir(path),key=natural_sort_key):     
      pixels = load_img(path + filename, target_size = size)# Convert to numpy array.             
      pixels = img_to_array(pixels).astype('float32')
      pixels = cv2.resize(pixels,(224,224))# Need to resize images first, otherwise RAM will run out of space. 
      pixels = pixels/255 
      #pixels = cv2.threshold(pixels, 128, 128, cv2.THRESH_BINARY)      
      data_list.append(pixels)
    return asarray(data_list)
        


##################################################  #

path_building = '/home/mferdaus/AIDER/collapsed_building/'
path_fire = '/home/mferdaus/AIDER/fire/'
path_flood = '/home/mferdaus/AIDER/flooded_areas/'
path_traffic = '/home/mferdaus/AIDER/traffic_incident/'
path_normal = '/home/mferdaus/AIDER/normal/'


for di_rate in dilation_rates:
    print(f"Using dilation rate: {di_rate}")
    for resolution in resolutions:
        print(f"Loading images at resolution: {resolution}")
        data_train_building = load_images(path_building, size=resolution)
        data_train_fire = load_images(path_fire, size=resolution)
        data_train_flood = load_images(path_flood, size=resolution)
        data_train_traffic = load_images(path_traffic, size=resolution)
        data_train_normal= load_images(path_normal, size=resolution)




        print("Building:", data_train_building.shape)  
        print("Fire:", data_train_fire.shape) 
        print("Flood:", data_train_flood.shape) 
        print("Traffic:", data_train_traffic.shape) 
        print("Normal:", data_train_normal.shape)

        # Prepare array for storing images and labels.

        imgarraybuilding =  []
        labelarraybuilding = []

        imgarrayfire =  []    
        labelarrayfire = []

        imgarrayflood =  []
        labelarrayflood = []

        imgarraytraffic =  []
        labelarraytraffic = []

        imgarraynormal =  []    
        labelarraynormal = []

        for a in data_train_building:
            imgarraybuilding.append(a)
            labelarraybuilding.append(0)

        for  b  in  data_train_fire : 
            imgarrayfire.append ( b ) 
            labelarrayfire.append ( 1 ) 

        for c in data_train_flood:    
            imgarrayflood.append(c)   
            labelarrayflood.append(2)

        for d in data_train_traffic:    
            imgarraytraffic.append(d)   
            labelarraytraffic.append(3)

        for e in data_train_normal:
            imgarraynormal.append ( e ) 
            labelarraynormal.append ( 4 )

        print('Total array shape building:',np.shape(imgarraybuilding))
        print('Total label array shape building:',np.shape(labelarraybuilding))
        print()
        print('Total array shape fire:',np.shape(imgarrayfire))
        print('Total label array shape fire:',np.shape(labelarrayfire))
        print()
        print('Total array shape flood:',np.shape(imgarrayflood))
        print('Total label array shape flood:',np.shape(labelarrayflood))
        print()
        print('Total array shape traffic:',np.shape(imgarraytraffic))
        print('Total label array shape traffic:',np.shape(labelarraytraffic))
        print()
        print('Total array shape normal:',np.shape(imgarraynormal))
        print('Total label array shape normal:',np.shape(labelarraynormal))

        from sklearn.model_selection import train_test_split

        ###################### Building #########################

        (X_trainb, X_testb, y_trainb, y_testb) = train_test_split(imgarraybuilding,labelarraybuilding,test_size=0.2,random_state=0)
        print(np.shape(X_trainb), np.shape(X_testb))
        (X_trainb, X_valb, y_trainb, y_valb) = train_test_split(X_trainb, y_trainb,test_size=0.1, random_state=0)
        print(np.shape(X_trainb), np.shape(X_valb))

        ###################### Fire #########################

        print()
        (X_trainf, X_testf, y_trainf, y_testf) = train_test_split(imgarrayfire,labelarrayfire,test_size=0.4,random_state=0)
        print(np.shape(X_trainf), np.shape(X_testf))
        (X_trainf, X_valf, y_trainf, y_valf) = train_test_split(X_trainf, y_trainf,test_size=0.2, random_state=0)
        print(np.shape(X_trainf), np.shape(X_valf))


        ###################### Flood #########################

        print()
        (X_trainF, X_testF, y_trainF, y_testF) = train_test_split(imgarrayflood,labelarrayflood,test_size=0.4,random_state=0)
        print(np.shape(X_trainF), np.shape(X_testF))
        (X_trainF, X_valF, y_trainF, y_valF) = train_test_split(X_trainF, y_trainF,test_size=0.2, random_state=0)
        print(np.shape(X_trainF), np.shape(X_valF))

        ###################### Traffic #########################

        print()
        (X_traint, X_testt, y_traint, y_testt) = train_test_split(imgarraytraffic,labelarraytraffic,test_size=0.4,random_state=0)
        print(np.shape(X_traint), np.shape(X_testt))
        (X_traint, X_valt, y_traint, y_valt) = train_test_split(X_traint, y_traint,test_size=0.2, random_state=0)
        print(np.shape(X_traint), np.shape(X_valt))

        ###################### Normal #########################

        print()
        (X_trainn, X_testn, y_trainn, y_testn) = train_test_split(imgarraynormal,labelarraynormal,test_size=0.4,random_state=0)
        print(np.shape(X_trainn), np.shape(X_testn))
        (X_trainn, X_valn, y_trainn, y_valn) = train_test_split(X_trainn, y_trainn,test_size=0.2, random_state=0)
        print(np.shape(X_trainn), np.shape(X_valn))

        # Prepare array for storing images and labels.(Training)

        imgarraybuilding_train =  []
        labelarraybuilding_train = []

        imgarrayfire_train =  []  
        labelarrayfire_train = []

        imgarrayflood_train =  []
        labelarrayflood_train = []

        imgarraytraffic_train =  []
        labelarraytraffic_train = []

        imgarraynormal_train =  []  
        labelarraynormal_train = []

        for a in X_trainb:
            imgarraybuilding_train.append(a)
            labelarraybuilding_train.append(0)

        for b in X_trainf:
            imgarrayfire_train.append(b)
            labelarrayfire_train.append(1)

        for c in X_trainF:    
            imgarrayflood_train.append(c)    
            labelarrayflood_train.append(2)

        for d in X_traint:   
            imgarraytraffic_train.append(d)    
            labelarraytraffic_train.append(3)

        for e in X_trainn:
            imgarraynormal_train.append(e)
            labelarraynormal_train.append(4)

        print('Total array shape building:',np.shape(imgarraybuilding_train))
        print('Total label array shape building:',np.shape(labelarraybuilding_train))
        print()
        print('Total array shape fire:',np.shape(imgarrayfire_train))
        print('Total label array shape fire:',np.shape(labelarrayfire_train))
        print()
        print('Total array shape flood:',np.shape(imgarrayflood_train))
        print('Total label array shape flood:',np.shape(labelarrayflood_train))
        print()
        print('Total array shape traffic:',np.shape(imgarraytraffic_train))
        print('Total label array shape traffic:',np.shape(labelarraytraffic_train))
        print()
        print('Total array shape normal:',np.shape(imgarraynormal_train))
        print('Total label array shape normal:',np.shape(labelarraynormal_train))

        # Prepare array for storing images and labels.(Validation)

        imgarraybuilding_valid = []
        labelarraybuilding_valid = []

        imgarrayfire_valid =   []  
        labelarrayfire_valid =   []

        imgarrayflood_valid = []
        labelarrayflood_valid = []

        imgarraytraffic_valid = []
        labelarraytraffic_valid = []

        imgarraynormal_valid =   []  
        labelarraynormal_valid =   []

        for a in X_valb:
            imgarraybuilding_valid.append(a)
            labelarraybuilding_valid.append(0)

        for b in X_valf:
            imgarrayfire_valid.append(b)
            labelarrayfire_valid.append(1)

        for c in X_valF:    
            imgarrayflood_valid.append(c)    
            labelarrayflood_valid.append(2)

        for d in X_valt:   
            imgarraytraffic_valid.append(d)    
            labelarraytraffic_valid.append(3)

        for e in X_valn:
            imgarraynormal_valid.append(e)
            labelarraynormal_valid.append(4)

        print('Total array shape building:',np.shape(imgarraybuilding_valid))
        print('Total label array shape building:',np.shape(labelarraybuilding_valid))
        print()
        print('Total array shape fire:',np.shape(imgarrayfire_valid))
        print('Total label array shape fire:',np.shape(labelarrayfire_valid))
        print()
        print('Total array shape flood:',np.shape(imgarrayflood_valid))
        print('Total label array shape flood:',np.shape(labelarrayflood_valid))
        print()
        print('Total array shape traffic:',np.shape(imgarraytraffic_valid))
        print('Total label array shape traffic:',np.shape(labelarraytraffic_valid))
        print()
        print('Total array shape normal:',np.shape(imgarraynormal_valid))
        print('Total label array shape normal:',np.shape(labelarraynormal_valid))

        # Prepare array for storing images and labels.(test)

        imgarraybuilding_test = []
        labelarraybuilding_test = []

        imgarrayfire_test =   []  
        labelarrayfire_test =   []

        imgarrayflood_test = []
        labelarrayflood_test= []

        imgarraytraffic_test = []
        labelarraytraffic_test = []

        imgarraynormal_test =   []  
        labelarraynormal_test=   []

        for a in X_testb:
            imgarraybuilding_test.append(a)
            labelarraybuilding_test.append(0)

        for b in X_testf:
            imgarrayfire_test.append(b)
            labelarrayfire_test.append(1)

        for c in X_testF:    
            imgarrayflood_test.append(c)    
            labelarrayflood_test.append(2)

        for d in X_testt:   
            imgarraytraffic_test.append(d)    
            labelarraytraffic_test.append(3)

        for e in X_testn:
            imgarraynormal_test.append(e)
            labelarraynormal_test.append(4)

        print('Total array shape building:',np.shape(imgarraybuilding_test))
        print('Total label array shape building:',np.shape(labelarraybuilding_test))
        print()
        print('Total array shape fire:',np.shape(imgarrayfire_test))
        print('Total label array shape fire:',np.shape(labelarrayfire_test))
        print()
        print('Total array shape flood:',np.shape(imgarrayflood_test))
        print('Total label array shape flood:',np.shape(labelarrayflood_test))
        print()
        print('Total array shape traffic:',np.shape(imgarraytraffic_test))
        print('Total label array shape traffic:',np.shape(labelarraytraffic_test))
        print()
        print('Total array shape normal:',np.shape(imgarraynormal_test))
        print('Total label array shape normal:',np.shape(labelarraynormal_test))

        trainarray = []
        trainlabel = []

        validarray = []
        validlabel = []

        testarray =  [] 
        testlabel =  []

        for  A1  in  imgarraybuilding_train : 
            trainarray.append ( A1 )

        for A2 in imgarraybuilding_valid:
            validarray.append(A2)

        for A3 in imgarraybuilding_test:
            testarray.append(A3)
            
        for A4 in labelarraybuilding_train:
            trainlabel.append(A4)

        for A5 in labelarraybuilding_valid:
            validlabel.append(A5)

        for A6 in labelarraybuilding_test:
            testlabel.append(A6)

        ###############################################
        
        for B1 in imgarrayfire_train:
            trainarray.append(B1)

        for B2 in imgarrayfire_valid:
            validarray.append(B2)

        for B3 in imgarrayfire_test:
            testarray.append(B3)
            
        for B4 in labelarrayfire_train:
            trainlabel.append(B4)

        for B5 in labelarrayfire_valid:
            validlabel.append(B5)

        for B6 in labelarrayfire_test:
            testlabel.append(B6)
        
        ###############################################

        for  C1  in  imgarrayflood_train : 
            trainarray.append ( C1 )

        for C2 in imgarrayflood_valid:
            validarray.append(C2)

        for C3 in imgarrayflood_test:
            testarray.append(C3)

        for C4 in labelarrayflood_train:
            trainlabel.append(C4)

        for C5 in labelarrayflood_valid:
            validlabel.append(C5)

        for C6 in labelarrayflood_test:
            testlabel.append(C6)

        ###############################################

        for  D1  in  imgarraytraffic_train : 
            trainarray.append ( D1 )

        for D2 in imgarraytraffic_valid:
            validarray.append(D2)

        for D3 in imgarraytraffic_test:
            testarray.append(D3)

        for D4 in labelarraytraffic_train:
            trainlabel.append(D4)

        for D5 in labelarraytraffic_valid:
            validlabel.append(D5)

        for D6 in labelarraytraffic_test:
            testlabel.append(D6)

        ###############################################

        for  E1  in  imgarraynormal_train : 
            trainarray.append ( E1 )

        for E2 in imgarraynormal_valid:
            validarray.append(E2)

        for E3 in imgarraynormal_test:
            testarray.append(E3)
            
        for E4 in labelarraynormal_train:
            trainlabel.append(E4)
            
        for E5 in labelarraynormal_valid:
            validlabel.append(E5)  

        for E6 in labelarraynormal_test:
            testlabel.append(E6) 

        ###############################################

        print(np.shape(trainarray))
        print(np.shape(trainlabel))
        print(np.shape(validarray))
        print(np.shape(validlabel))
        print(np.shape(testarray))
        print(np.shape(testlabel))

        training_AIDER =  [] 
        valid_AIDER =  [] 
        test_AIDER = []

        for a, b in zip(trainarray,trainlabel):
            training_AIDER.append([a,b])

        for c, d in zip(validarray,validlabel):
            valid_AIDER.append([c,d])

        for e, f in zip(testarray,testlabel):
            test_AIDER.append([e,f])
            
            
        print(len(training_AIDER))
        print(len(valid_AIDER))
        print(len(test_AIDER))

        print(np.shape(training_AIDER))
        print(np.shape(valid_AIDER))
        print(np.shape(test_AIDER))

        # Shuffle img array and label (Important to randomize order)

        from sklearn.utils import shuffle

        finaltraining = shuffle (training_AIDER) 
        finalval = shuffle (valid_AIDER) 
        finaltest = shuffle(test_AIDER)

        new_X = [x[0] for x in finaltraining]
        new_y = [x[1] for x in finaltraining]

        new_X_valid = [w[0] for w in finalval]
        new_y_valid  = [w[1] for w in finalval]

        new_X_test = [v[0] for v in finaltest]
        new_y_test  = [v[1] for v in finaltest]

        plt.figure()
        plt.imshow(new_X[65])
        print(new_y[65])

        print()

        plt.figure()
        plt.imshow(new_X_valid[65])
        print(new_y_valid[65])

        print()

        plt.figure()
        plt.imshow(new_X_test[65])
        print(new_y_test[65])

        print(np.shape(new_X))
        print(np.shape(new_X_valid))
        print(np.shape(new_X_test))

        # On train datasets.

        from imblearn.over_sampling import RandomOverSampler  # Important library
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import SMOTE
        from collections import Counter  # Important library

        #ros = RandomOverSampler()
        #ros = SMOTE()
        rus = RandomUnderSampler(random_state=42)
        # Resampling training datasets.
        trainarray = np.reshape(new_X,(len(new_X), 224*224*3))
        # Use fit_resample.
        trainarray_rus,trainlabel_rus = rus.fit_resample(trainarray,new_y)
        print(Counter(trainlabel_rus))

        # reshaping X back to the first dims
        new_X = trainarray_rus.reshape(-1,224,224,3)
        print(len(new_X))
        print(np.shape(new_X_test))
        new_y  = trainlabel_rus
        print(len(new_y))

        # On valid datasets.

        from imblearn.over_sampling import RandomOverSampler  # Important library
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from collections import Counter  # Important library

        #ros = RandomOverSampler()
        #ros = SMOTE()
        rus = RandomUnderSampler(random_state=42)
        # Resampling training datasets.

        validarray = np.reshape(new_X_valid,(len(new_X_valid), 224*224*3))
        # Use fit_resample.
        validarray_rus,validlabel_rus = rus.fit_resample(validarray,new_y_valid)
        print(Counter(validlabel_rus))

        # reshaping X back to the first dims
        new_X_valid = validarray_rus.reshape(-1,224,224,3)
        print(len(new_X_valid))
        print(np.shape(new_X_valid))
        new_y_valid  = validlabel_rus
        print(len(new_y_valid))

        # On test datasets.

        from imblearn.over_sampling import RandomOverSampler  # Important library
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from collections import Counter  # Important library

        #ros = RandomOverSampler()
        #ros = SMOTE()
        rus = RandomUnderSampler(random_state=42)
        # Resampling training datasets.

        testarray = np.reshape(new_X_test,(len(new_X_test), 224*224*3))
        # Use fit_resample.
        testarray_rus,testlabel_rus = rus.fit_resample(testarray,new_y_test)
        print(Counter(testlabel_rus))

        # reshaping X back to the first dims
        new_X_test = testarray_rus.reshape(-1,224,224,3)
        print(len(new_X_test))
        print(np.shape(new_X_test))
        new_y_test  = testlabel_rus
        print(len(new_y_test))

        # One hot encode all label training array.

        from numpy import array
        from numpy import argmax
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(new_y)
        #print(integer_encoded)

        print()

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        print(np.shape(onehot_encoded))
        print(len(onehot_encoded))

        # One hot encode all label valid array.

        from numpy import array
        from numpy import argmax
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(new_y_valid)
        #print(integer_encoded)

        print()

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encodedvalid = onehot_encoder.fit_transform(integer_encoded)
        print(np.shape(onehot_encodedvalid))
        print(len(onehot_encodedvalid))

        # One hot encode all label test array.

        from numpy import array
        from numpy import argmax
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(new_y_test)
        #print(integer_encoded)

        print()

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encodedtest = onehot_encoder.fit_transform(integer_encoded)
        print(np.shape(onehot_encodedtest))
        print(len(onehot_encodedtest))

        # Attention Mechanism

        # from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
        # from keras import backend as K
        # from keras.activations import sigmoid

        # tf.config.run_functions_eagerly(True)  # IMPT!!

        # def attach_attention_module(net, attention_module):
        #   if attention_module == 'se_block': # SE_block
        #     net = se_block(net)
        #   elif attention_module == 'cbam_block': # CBAM_block
        #     net = cbam_block(net)
        #   else:
        #     raise Exception("'{}' is not supported attention module!".format(attention_module))

        #   return net

        # def se_block(input_feature, ratio=8):
        #     """Contains the implementation of Squeeze-and-Excitation(SE) block.
        #     As described in https://arxiv.org/abs/1709.01507.
        #     """
            
        #     channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        #     channel = input_feature.shape[channel_axis]

        #     se_feature = GlobalAveragePooling2D()(input_feature)
        #     se_feature = Reshape((1, 1, channel))(se_feature)
        #     assert se_feature.shape[1:] == (1,1,channel)
        #     se_feature = Dense(channel // ratio,
        #                        activation='relu',
        #                        kernel_initializer='he_normal',
        #                        use_bias=True,
        #                        bias_initializer='zeros')(se_feature)
        #     assert se_feature.shape[1:] == (1,1,channel//ratio)
        #     se_feature = Dense(channel,
        #                        activation='sigmoid',
        #                        kernel_initializer='he_normal',
        #                        use_bias=True,
        #                        bias_initializer='zeros')(se_feature)
        #     assert se_feature.shape[1:] == (1,1,channel)
        #     if K.image_data_format() == 'channels_first':
        #         se_feature = Permute((3, 1, 2))(se_feature)

        #     se_feature = multiply([input_feature, se_feature])
        #     return se_feature

        # def cbam_block(cbam_feature, ratio=8):
        #     """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        #     As described in https://arxiv.org/abs/1807.06521.
        #     """
            
        #     cbam_feature = channel_attention(cbam_feature, ratio)
        #     cbam_feature = spatial_attention(cbam_feature)
        #     return cbam_feature

        # def channel_attention(input_feature, ratio=8):
            
        #     channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        #     channel = input_feature.shape[channel_axis]
            
        #     shared_layer_one = Dense(channel//ratio,
        #                              activation='relu',
        #                              kernel_initializer='he_normal',
        #                              use_bias=True,
        #                              bias_initializer='zeros')
        #     shared_layer_two = Dense(channel,
        #                              kernel_initializer='he_normal',
        #                              use_bias=True,
        #                              bias_initializer='zeros')
            
        #     avg_pool = GlobalAveragePooling2D()(input_feature)    
        #     avg_pool = Reshape((1,1,channel))(avg_pool)
        #     assert avg_pool.shape[1:] == (1,1,channel)
        #     avg_pool = shared_layer_one(avg_pool)
        #     assert avg_pool.shape[1:] == (1,1,channel//ratio)
        #     avg_pool = shared_layer_two(avg_pool)
        #     assert avg_pool.shape[1:] == (1,1,channel)
            
        #     max_pool = GlobalMaxPooling2D()(input_feature)
        #     max_pool = Reshape((1,1,channel))(max_pool)
        #     assert max_pool.shape[1:] == (1,1,channel)
        #     max_pool = shared_layer_one(max_pool)
        #     assert max_pool.shape[1:] == (1,1,channel//ratio)
        #     max_pool = shared_layer_two(max_pool)
        #     assert max_pool.shape[1:] == (1,1,channel)
            
        #     cbam_feature = Add()([avg_pool,max_pool])
        #     cbam_feature = Activation('sigmoid')(cbam_feature)
            
        #     if K.image_data_format() == "channels_first":
        #         cbam_feature = Permute((3, 1, 2))(cbam_feature)
            
        #     return multiply([input_feature, cbam_feature])

        # def spatial_attention(input_feature):
        #     kernel_size = 7
            
        #     if K.image_data_format() == "channels_first":
        #         channel = input_feature.shape[1]
        #         cbam_feature = Permute((2,3,1))(input_feature)
        #     else:
        #         channel = input_feature.shape[-1]
        #         cbam_feature = input_feature
            
        #     avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
        #     assert avg_pool.shape[-1] == 1
        #     max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
        #     assert max_pool.shape[-1] == 1
        #     concat = Concatenate(axis=3)([avg_pool, max_pool])
        #     assert concat.shape[-1] == 2
        #     cbam_feature = Conv2D(filters = 1,
        #                     kernel_size=kernel_size,
        #                     strides=1,
        #                     padding='same',
        #                     activation='sigmoid',
        #                     kernel_initializer='he_normal',
        #                     use_bias=False)(concat) 
        #     assert cbam_feature.shape[-1] == 1
            
        #     if K.image_data_format() == "channels_first":
        #         cbam_feature = Permute((3, 1, 2))(cbam_feature)
                
        #     return multiply([input_feature, cbam_feature])

        ### Watt32-- Model

        """START

        Watt32
        """

        import tensorflow as tf
        import math
        from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization, concatenate, Subtract
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Add, Activation, Conv2DTranspose,GlobalAveragePooling2D,DepthwiseConv2D

        from tensorflow.keras import regularizers

        NUM_CLASSES = 5


        def swish(x):
            return x * tf.nn.sigmoid(x)


        def round_filters(filters, multiplier):
            depth_divisor = 8
            min_depth = None
            min_depth = min_depth or depth_divisor
            filters = filters * multiplier
            new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
            if new_filters < 0.9 * filters:
                new_filters += depth_divisor
            return int(new_filters)


        def round_repeats(repeats, multiplier):
            if not multiplier:
                return repeats
            return int(math.ceil(multiplier * repeats))


        class SEBlock(Layer):
            def __init__(self, input_channels, ratio=0.25):
                super(SEBlock, self).__init__()
                self.num_reduced_filters = max(1, int(input_channels * ratio))
                self.pool = GlobalAveragePooling2D()
                self.reduce_conv = Conv2D(filters=self.num_reduced_filters,
                                                        kernel_size=(1, 1),
                                                        strides=1, kernel_regularizer = regularizers.L2(1e-6), padding="same")
                
                self.expand_conv = Conv2D(filters=input_channels,
                                                        kernel_size=(1, 1),
                                                        strides=1, kernel_regularizer = regularizers.L2(1e-6),                                          
                                                        padding="same")

            def call(self, inputs, **kwargs):
                branch = self.pool(inputs)
                branch = tf.expand_dims(input=branch, axis=1)
                branch = tf.expand_dims(input=branch, axis=1)
                branch = self.reduce_conv(branch)
                branch = swish(branch)
                branch = self.expand_conv(branch)
                branch = tf.nn.sigmoid(branch)
                output = inputs * branch
                return output
            
            def from_config(cls, config):
                return cls(**config)


        class MBConv(Layer):
            def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate, dilation_rate=(2, 2)):
                super(MBConv, self).__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.stride = stride if dilation_rate == (1, 1) else (1, 1)
                self.drop_connect_rate = drop_connect_rate
                self.dilation_rate = dilation_rate
                self.conv1 = Conv2D(filters=in_channels * expansion_factor,         # WIDTH ????
                                                    kernel_size=(1, 1),
                                                    strides=1, kernel_regularizer = regularizers.L2(1e-6),
                                                    padding="same")
                self.bn1 = BatchNormalization()
                self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                            strides=self.stride,
                                                            padding="same",
                                                            dilation_rate=self.dilation_rate)
                self.bn2 = tf.keras.layers.BatchNormalization()   
                self.dwconv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                            strides=self.stride,
                                                            padding="same",
                                                            dilation_rate=self.dilation_rate)                          
                self.bn22 = BatchNormalization()
                self.se = SEBlock(input_channels=in_channels * expansion_factor)       # WIDTH ????
                self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                                    kernel_size=(1, 1),
                                                    strides=1, kernel_regularizer = regularizers.L2(1e-6),
                                                    padding="same")
                self.bn3 = tf.keras.layers.BatchNormalization()
                self.dropout = Dropout(rate=drop_connect_rate)
                
                

            def call(self, inputs, training=None, **kwargs):
                x = self.conv1(inputs)
                x = self.bn1(x, training=training)
                x = swish(x)
                x = self.dwconv(x)
                x = self.bn2(x, training=training)
                x = self.dwconv2(x)
                x = self.bn22(x, training=training)

                x = self.se(x)
                x = swish(x)
                x = self.conv2(x)
                x = self.bn3(x, training=training)
                if self.stride == 1 and self.in_channels == self.out_channels:
                    if self.drop_connect_rate:
                        x = self.dropout(x, training=training)
                    x = Add()([x, inputs])
                return x

            def from_config(cls, config):
                return cls(**config)


        def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
            block = tf.keras.Sequential()
            for i in range(layers):
                if i == 0:
                    block.add(MBConv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    expansion_factor=expansion_factor,
                                    stride=stride,
                                    k=k,
                                    drop_connect_rate=drop_connect_rate))
                else:
                    block.add(MBConv(in_channels=out_channels,
                                    out_channels=out_channels,
                                    expansion_factor=expansion_factor,
                                    stride=1,
                                    k=k,
                                    drop_connect_rate=drop_connect_rate))
            return block


        class EfficientNet(tf.keras.Model):
            def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.5):
                super(EfficientNet, self).__init__()

                self.conv1 = Conv2D(filters=round_filters(32, width_coefficient),   #32
                                                    kernel_size=(3, 3),
                                                    strides=2, kernel_regularizer = regularizers.L2(1e-6),
                                                    padding="same")
                self.bn1 = BatchNormalization()
                self.block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                                out_channels=round_filters(16, width_coefficient),
                                                layers=round_repeats(1, depth_coefficient),  #1
                                                stride=1,  # 1, 1, 3.
                                                expansion_factor=2, k=3, drop_connect_rate=drop_connect_rate)   # CHANGE EXPANSION FACTOR HERE!
                self.block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                                out_channels=round_filters(24, width_coefficient),
                                                layers=round_repeats(2, depth_coefficient),  #2
                                                stride=2,   # 2,6,3
                                                expansion_factor = 12, k=3, drop_connect_rate=drop_connect_rate)   # CHANGE EXPANSION FACTOR HERE!                         
                #self.block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                                #out_channels=round_filters(40, width_coefficient),
                                                #layers=round_repeats(2, depth_coefficient),
                                                #stride=2,   #2,6,5
                                                #expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)  
            
                self.pool = GlobalAveragePooling2D()
                self.dropout = Dropout(rate=dropout_rate)
                self.fc = Dense(units=5,activation=tf.keras.activations.softmax)

            def call(self, inputs, training=None, mask=None):
                x = self.conv1(inputs)
                x = self.bn1(x, training=training)
                x = swish(x)
                x = self.block1(x)
                #x = cbam_block(x,4
                x = self.block2(x)
            #x = cbam_block(x,4)
                #x = self.block3(x)
                x = swish(x)
                x = Add()([x,x])
                x = self.pool(x)
                x = self.dropout(x, training=training)
                x = self.fc(x)

                return x

            def from_config(cls, config):
                return cls(**config)

        def get_efficient_net(width_coefficient, depth_coefficient, resolution, dropout_rate):
            net = EfficientNet(width_coefficient=width_coefficient,
                            depth_coefficient=depth_coefficient,
                            dropout_rate=dropout_rate)
            net.build(input_shape=(None, resolution, resolution, 3))
            net.call(Input(shape=(resolution, resolution, 3)))

            return net

        def efficient_net_b0():
            return get_efficient_net(1.0, 1.0, 224, 0.2)          # DONT MODIFY EXCEPT LAST VALUE.


        watteffnet32nat =  get_efficient_net(1.0, 1.0, 224, 0.1)    # DONT MODIFY  EXCEPT LAST VALUE.
        #model = Model(Input(224,224,3), efficient_net_b0)
        watteffnet32nat .summary()

        # def net_flops(model, table=False):
        #     if (table == True):
        #         print('%25s | %16s | %16s | %16s | %16s | %6s | %6s' % (
        #             'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
        #         print('-' * 170)

        #     t_flops = 0
        #     t_macc = 0

        #     for l in model.layers:

        #         o_shape, i_shape, strides, ks, filters = ['', '', ''], ['', '', ''], [1, 1], [0, 0], [0, 0]
        #         flops = 0
        #         macc = 0
        #         name = l.name

        #         factor = 1000000

        #         if ('InputLayer' in str(l)):
        #             i_shape = l.input.get_shape()[1:4].as_list()
        #             o_shape = i_shape

        #         if ('Reshape' in str(l)):
        #             i_shape = l.input.get_shape()[1:4].as_list()
        #             o_shape = l.output.get_shape()[1:4].as_list()

        #         if ('Add' in str(l) or 'Maximum' in str(l) or 'Concatenate' in str(l)):
        #             i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
        #             o_shape = l.output.get_shape()[1:4].as_list()
        #             flops = (len(l.input) - 1) * i_shape[0] * i_shape[1] * i_shape[2]

        #         if ('Average' in str(l) and 'pool' not in str(l)):
        #             i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
        #             o_shape = l.output.get_shape()[1:4].as_list()
        #             flops = len(l.input) * i_shape[0] * i_shape[1] * i_shape[2]

        #         if ('BatchNormalization' in str(l)):
        #             i_shape = l.input.get_shape()[1:4].as_list()
        #             o_shape = l.output.get_shape()[1:4].as_list()

        #             bflops = 1
        #             for i in range(len(i_shape)):
        #                 bflops *= i_shape[i]
        #             flops /= factor

        #         if ('Activation' in str(l) or 'activation' in str(l)):
        #             i_shape = l.input.get_shape()[1:4].as_list()
        #             o_shape = l.output.get_shape()[1:4].as_list()
        #             bflops = 1
        #             for i in range(len(i_shape)):
        #                 bflops *= i_shape[i]
        #             flops /= factor

        #         if ('pool' in str(l) and ('Global' not in str(l))):
        #             i_shape = l.input.get_shape()[1:4].as_list()
        #             strides = l.strides
        #             ks = l.pool_size
        #             flops = ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]) * (ks[0] * ks[1] * i_shape[2]))

        #         if ('Flatten' in str(l)):
        #             i_shape = l.input.shape[1:4].as_list()
        #             flops = 1
        #             out_vec = 1
        #             for i in range(len(i_shape)):
        #                 flops *= i_shape[i]
        #                 out_vec *= i_shape[i]
        #             o_shape = flops
        #             flops = 0

        #         if ('Dense' in str(l)):
        #             print(l.input)
        #             i_shape = l.input.shape[1:4].as_list()[0]
        #             if (i_shape == None):
        #                 i_shape = out_vec

        #             o_shape = l.output.shape[1:4].as_list()
        #             flops = 2 * (o_shape[0] * i_shape)
        #             macc = flops / 2

        #         if ('Padding' in str(l)):
        #             flops = 0

        #         if (('Global' in str(l))):
        #             i_shape = l.input.get_shape()[1:4].as_list()
        #             flops = ((i_shape[0]) * (i_shape[1]) * (i_shape[2]))
        #             o_shape = [l.output.get_shape()[1:4].as_list(), 1, 1]
        #             out_vec = o_shape

        #         if ('Conv2D ' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l)):
        #             strides = l.strides
        #             ks = l.kernel_size
        #             filters = l.filters
        #             i_shape = l.input.get_shape()[1:4].as_list()
        #             o_shape = l.output.get_shape()[1:4].as_list()

        #             if (filters == None):
        #                 filters = i_shape[2]

        #             flops = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
        #                     (i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
        #             macc = flops / 2

        #         if ('Conv2D ' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l)):
        #             strides = l.strides
        #             ks = l.kernel_size
        #             filters = l.filters
        #             i_shape = l.input.get_shape()[1:4].as_list()
        #             o_shape = l.output.get_shape()[1:4].as_list()

        #             if (filters == None):
        #                 filters = i_shape[2]

        #             flops = 2 * (
        #                     (ks[0] * ks[1] * i_shape[2]) * ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]))) 
        #             macc = flops / 2

        #         t_macc += macc

        #         t_flops += flops

        #         if (table == True):
        #             print('%25s | %16s | %16s | %16s | %16s | %6s | %5.4f' % (
        #                 name, str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))
        #     t_flops = t_flops / factor

        #     print('\nTotal FLOPS (x 10^6): %10.8f\n' % (t_flops))
        #     print('\nTotal MACCs: %10.8f\n' % (t_macc))

        #     return

        # net_flops(watteffnet32nat,table=True)

        ### Train and test Watt 32
        import time
        import csv
        import json

        watteffnet32nat.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
        )

        from keras import backend as K

        def recall_m(y_true, y_pred):   # 'Recall
            true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):   # Precision
            true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):    # F1 score
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))


        num_runs = 5
        recall_scores = []
        precision_scores = []
        f1_scores = []
        elapsed_times = []

        for i in range(num_runs):
            start_time = time.time()
            
            history = watteffnet32nat.fit(
                np.array(new_X),
                np.array(new_y),
                batch_size=10,
                epochs=300,
                validation_data=(np.array(new_X_valid), np.array(new_y_valid))
            )
            predicted_label = watteffnet32nat.predict(np.asarray(new_X_test))
            
            recall = recall_m(onehot_encodedtest.astype('float32'), predicted_label.astype('float32'))
            precision = precision_m(onehot_encodedtest.astype('float32'), predicted_label.astype('float32'))
            f1 = f1_m(onehot_encodedtest.astype('float32'), predicted_label.astype('float32'))
            
            recall_scores.append(recall)
            precision_scores.append(precision)
            f1_scores.append(f1)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            
            print("Iteration", i+1)
            print("Recall Scores:", recall_scores)
            print("Precision Scores:", precision_scores)
            print("F1 Scores:", f1_scores)
            print("Time Taken:", elapsed_time, "seconds")
            print()

        # Calculate the average and standard deviation
        recall_avg = np.mean(recall_scores)
        precision_avg = np.mean(precision_scores)
        f1_avg = np.mean(f1_scores)

        recall_std = np.std(recall_scores)
        precision_std = np.std(precision_scores)
        f1_std = np.std(f1_scores)
        elapsed_avg = np.mean(elapsed_times)
        elapsed_std = np.std(elapsed_times)

        print("Recall - Average: {:.4f}, Standard Deviation: {:.4f}".format(recall_avg, recall_std))
        print("Precision - Average: {:.4f}, Standard Deviation: {:.4f}".format(precision_avg, precision_std))
        print("F1 score - Average: {:.4f}, Standard Deviation: {:.4f}".format(f1_avg, f1_std))
        print("Elapsed Time - Average: {:.4f} seconds, Standard Deviation: {:.4f} seconds".format(elapsed_avg, elapsed_std))
        
        # Convert these to strings that can be included in file names
        dilation_rate_str = f"{di_rate[0]}x{di_rate[1]}"
        resolution_str = f"{resolution[0]}x{resolution[1]}"

        # Define directory and file names with dilation rate and resolution
        results_dir = f"results_cikm23_{resolution_str}_{dilation_rate_str}"
        csv_file_name = f"results_watteffnet32nat_{resolution_str}_{dilation_rate_str}.csv"
        txt_file_name = f"watteffnet32nat_summary_{resolution_str}_{dilation_rate_str}.txt"
        json_file_name = f"watteffnet32nat_weights_{resolution_str}_{dilation_rate_str}.json"

        # Create directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)


       # Save results to CSV file
        csv_path = os.path.join(results_dir, csv_file_name)
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Recall Scores', 'Precision Scores', 'F1 Scores', 'Time Taken'])
            for i in range(num_runs):
                writer.writerow([i+1, recall_scores[i], precision_scores[i], f1_scores[i], elapsed_times[i]])
            writer.writerow([])
            writer.writerow(['Mean', recall_avg, precision_avg, f1_avg, elapsed_avg])
            writer.writerow(['Standard Deviation', recall_std, precision_std, f1_std, elapsed_std])

        # Save model summary to a text file
        txt_path = os.path.join(results_dir, txt_file_name)
        with open(txt_path, 'w') as file:
            watteffnet32nat.summary(print_fn=lambda x: file.write(x + '\n'))

        # Save model weights to JSON file
        watteffnet32nat_weights = watteffnet32nat.get_weights()
        watteffnet32nat_weights_json = [weights.tolist() for weights in watteffnet32nat_weights]
        json_path = os.path.join(results_dir, json_file_name)
        with open(json_path, 'w') as file:
            json.dump(watteffnet32nat_weights_json, file)