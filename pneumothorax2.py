#matplotlib inline
import matplotlib.pyplot as plt
from keras import backend as K
import os
from tqdm import tqdm
import cv2
import pandas as pd
from skimage import transform
import numpy as np
from keras.utils import multi_gpu_model	
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Input, MaxPooling2D,Conv2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import multi_gpu_model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import concatenate,add

#setting the size of image used for training
img_width, img_height =  299,299
#using greyscale image
channels = 1

#path where train data is present
train_dir = "../../input/split_chest_xray/train/"
#paths where test data is present
test_dir = "../../input/split_chest_xray/test/"
#no of classes the data is trained on
classes = 3

#setting a random seed to ensure that model gives same results consistently
np.random.seed(10)

def get_data(folder):
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0#class label when normal data is fed into the system
            elif folderName in ['PNEUMOTHORAX']:
                label = 1#class label for pnemothorax
            else: 
                label = 2 
            if(label < 2):
                for image_filename in tqdm(os.listdir(folder + folderName)):
                    img_file = cv2.imread(folder + folderName + '/' + image_filename)
                    if img_file is not None:
                        img_file = transform.resize(img_file, (img_width, img_height, channels))
                        #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                        img_arr = np.asarray(img_file)
                        X.append(img_arr)
                        y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

X_train, y_train = get_data(train_dir)
X_test, y_test= get_data(test_dir)
print(X_train.shape)
 
#random sampling data 
import random
class_count=[]
for i in range(classes):
        class_count.append(np.count_nonzero(y_train=classes))

sampleSize = min(class_count)
print(onesCount, zerosCount)
indices = np.where(y_train ==0)[0]
if(len(indices) > 0):
	indices = np.random.choice(indices,sampleSize,replace = True )
	indices = indices.flatten()
	X_train_0 = X_train[indices]
	y_train_0 = np.full(sampleSize,0)

indices = np.where(y_train ==1)[0]
if(len(indices) > 0):
	indices = np.random.choice(indices,sampleSize,replace = True )
	indices = indices.flatten()
	X_train_1 = X_train[indices]
	y_train_1 = np.full(sampleSize,1)

X_train =np.concatenate((X_train_0,X_train_1), axis = 0)
y_train =np.concatenate((y_train_0,y_train_1), axis = 0)
print(X_train.shape)
print(y_train.size)

nb_train_samples = X_train.shape[0] 
nb_validation_samples = X_test.shape[0]
batch_size = 32
epochs = 150

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = classes)
y_testHot = to_categorical(y_test, num_classes = classes)


X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
#calculating base line accuracy in the model
print("Train Baseline", (np.sum(y_train)*100.0)/y_train.size)
print("Test Baseline", (np.sum(y_test)*100.0)/y_test.size)


print(type(X_train),X_train.shape)
print(type(X_test),X_test.shape)
print(type(X_val),X_val.shape)
print(type(y_train),len(y_train))
print(type(y_test),len(y_test))
print(type(y_val),len(y_val))


X_train = np.divide(X_train,np.array(255.0))
X_test = np.divide(X_test,np.array(255.0))
X_val = np.divide(X_val,np.array(255.0))


inputs = Input(shape = (img_width, img_height, channels))
x1 = Conv2D(16,(9,9),name = "1conv2d_1",activation = "relu") (inputs)
x1 = MaxPooling2D(2,name = "1max1") (x1)
#x = LeakyReLU(0.1)(x)
x1 = Conv2D(64,(7,7),name = "1conv2d_2",activation = "relu")(x1)
x1 = MaxPooling2D(2,name = "1max2") (x1)
x1 = Conv2D(32,(5,5),name = "1conv2d_3",activation = "relu")(x1)
x1 = MaxPooling2D(2,name = "1max3") (x1)
x1 = Conv2D(16,(3,3),name = "1conv2d_4",activation = "relu")(x1)
x1 = MaxPooling2D(2,name = "1max4") (x1)
x1 = Conv2D(8,(1,1),name = "1conv2d_5",activation = "relu")(x1)
x1 = MaxPooling2D(2,name = "1max5") (x1)
x1 = Conv2D(4,(1,1),name = "1conv2d_6",activation = "relu")(x1)
x1 = MaxPooling2D(2,name = "1max6") (x1)
x1 = Conv2D(2,(2,2),name = "1conv2d_7",activation = "relu")(x1)
x1 = LeakyReLU(0.1)(x1)
x1 = MaxPooling2D(2,name = "1max7") (x1)
x1 = Conv2D(4,(1,1),name = "1conv2d_8",activation = "relu")(x1)
x1 = Flatten(name = "1flatten_1")(x1)
x1 = Dropout(0.2,name = "1dropout_1")(x1)
x1 = Dense(30,activation = "relu",name = "1dense_1")(x1)
x1 = Dropout(0.3,name = "1dropout_2")(x1)


x2 = Conv2D(8,(9,9),name = "conv2d_1",activation = "relu") (inputs)
x2 = MaxPooling2D(2,name = "max1") (x2)
x2 = Conv2D(16,(7,7),name = "conv2d_2",activation = "relu")(x2)
x2 = MaxPooling2D(2,name = "max2") (x2)
x2 = Conv2D(32,(5,5),name = "conv2d_3",activation = "relu")(x2)
x2 = MaxPooling2D(2,name = "max3") (x2)
x2 = Conv2D(64,(3,3),name = "conv2d_4",activation = "relu")(x2)
x2 = MaxPooling2D(2,name = "max4") (x2)
x2 = Conv2D(32,(1,1),name = "conv2d_5",activation = "relu")(x2)
x2 = MaxPooling2D(2,name = "max5") (x2)
x2 = Conv2D(16,(1,1),name = "conv2d_6",activation = "relu")(x2)
x2 = MaxPooling2D(2,name = "max6") (x2)
x2 = Conv2D(8,(2,2),name = "conv2d_7",activation = "relu")(x2)
x2 = LeakyReLU(0.1)(x2)
x2 = MaxPooling2D(2,name = "max7") (x2)
x2 = Conv2D(4,(1,1),name = "conv2d_8",activation = "relu")(x2)
x2 = Flatten(name = "flatten_1")(x2)
x2 = Dropout(0.2,name = "dropout_1")(x2)
x2 = Dense(30,activation = "relu",name = "dense_1")(x2)
x2 = Dropout(0.3,name = "dropout_2")(x2)

x = add([x1,x2],name = "add_1")
x = Dropout(0.3,name = "dropout_3")(x)
pred = Dense(classes,activation = "sigmoid",name = "dense_200") (x)


model_final = Model(input = inputs,output = pred)

np.random.seed(10)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = "RMSprop", metrics=["accuracy"])


print(model_final.summary())
# checkpoint
filepath="pneumothorax2-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

model_info = model_final.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1,callbacks = [checkpoint])#,early])
# Final evaluation of the model
scores = model_final.evaluate(X_test, y_test, verbose=1)
print("Error: %.2f%%" % ((1-scores[1])*100))
print("Score: %.2f%%" % ((scores[1])*100))
#save the model
model_final.save('savedModel_fused16_18_full.h5')

Y_pred = model_final.predict(X_test)

#evaluation
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, np.round(Y_pred)))

Y_train_pred = model_final.predict(X_train)
print(confusion_matrix(y_train, np.round(Y_train_pred)))



