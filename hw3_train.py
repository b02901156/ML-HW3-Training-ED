# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, Input
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
#from keras.models import Sequential,Model 
#from keras.layers.core import Dense, Dropout, Activation
#from keras.optimizers import SGD, Adam
import numpy as np
import csv
import sys

def load_data(filename):
    with open(filename,'r') as train_csv:
        train = csv.reader(train_csv)
        train_x = []
        train_y = []
        next(train)      #first line is wrong format
        for row in train:
            train_y.append(row[0])
            t = [float(i)/255 for i in row[1].split()]
            train_x.append(t)
    
    return (train_x, train_y)


def split_valid_set(train_x, train_y, valid_per):        
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    #split for validation
    train_num = train_y.shape[0]
    random_num = np.arange(train_num)
    np.random.shuffle(random_num)
    
    validation_perc = valid_per                             # percentage of validation_set
    ind_TS = random_num[0:-int(train_num*validation_perc)]  #train_set index
    ind_VS = random_num[-int(train_num*validation_perc):]   #validation_set index
    x_train_set = np.array(train_x[ind_TS])
    x_validation_set = np.array(train_x[ind_VS])
    y_train_set = np.array(train_y[ind_TS])
    y_validation_set = np.array(train_y[ind_VS])
    
    #convert label to one-hot encoding
    y_train_set = np_utils.to_categorical(y_train_set, num_classes=7)
    y_validation_set = np_utils.to_categorical(y_validation_set, num_classes=7)

    return (x_train_set, y_train_set, x_validation_set, y_validation_set)    

def build_model():
    model = Sequential()
    
    model.add(Conv2D(filters = 32,kernel_size = (3,3),padding = 'same',input_shape=(48,48,1), activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 32,kernel_size = (3,3),padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters =64,kernel_size = (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters =64,kernel_size = (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128,kernel_size =(3,3),padding = 'same',activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128,kernel_size =(3,3),padding = 'same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
        
    model.add(Flatten())
    
    model.add(Dense(units=512,activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=7,activation='softmax'))
    model.summary()
    
    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    # opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    return model
    
def Start_training(x_train, x_validation, y_train, y_validation, model, batch_size, epoch):
      
    x_train = x_train.reshape(x_train.shape[0],48,48,1)
    x_validation = x_validation.reshape(x_validation.shape[0],48,48,1)
      
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,   # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            zoom_range = 0.05,
            vertical_flip=False)  # randomly flip images
    
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    
    batch_size =32
    epochs_1 = 200
    # Fit the model on the batches generated by datagen.flow().
    mm1 = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs_1,
                        validation_data=(x_validation,y_validation))
    
    score = model.evaluate(x_train,y_train)
    
    print('\nTrain Acc:', score[1])
    
    model.save('hw3_model_1.h5')

#####################
def read_test():
    with open('test.csv', 'r') as csvfile_test:
    	test = csv.reader(csvfile_test)
    	x_test = []
    	next(test)
    	test_index = 0
    	for row in test:
    		t = [float(i)/255 for i in row[1].split()]
    		x_test.append(t)
    		test_index = test_index + 1 
    x_test = np.array(x_test)
    return x_test

def predict(a, name):
    out = []
    for i in range(len(a)):
        max, num = 0, 0
        for j in range(7):
            if a[i][j] > max:
                max, num = a[i][j], j
        out.append(num)
    with open('./'+name, 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        d = 0
        for ele in out:
            writer.writerow([str(d), str(int(ele))])
            d+=1
    
    #model1 = load_model('hw3_model.h5')
    
    

def main():
    filename = 'train.csv'
    train_x, train_y = load_data(filename)
    x_train_set, y_train_set, x_validation_set, y_validation_set = split_valid_set(train_x, train_y, valid_per = 0.1)
    model = build_model()
    Start_training(x_train_set, x_validation_set, y_train_set, y_validation_set, model, batch_size=32, epoch=20)
    x_test = read_test()
    x_test = x_test.reshape(x_test.shape[0],48,48,1)
    model = load_model('hw3_model_1.h5')
    predict(model.predict(x_test),'output.csv')
    
if __name__ == '__main__':
    main()
    
