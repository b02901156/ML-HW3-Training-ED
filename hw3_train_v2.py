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
import argparse

####################################################################################
#-----------------------------------for training-----------------------------------#
####################################################################################

def load_train(train_data_path):
    with open(train_data_path,'r') as train_csv:
        train = csv.reader(train_csv)
        train_x = []
        train_y = []
        next(train)      #from the second line
        for row in train:
            train_y.append(row[0])
            t = [float(i)/255 for i in row[1].split()]
            train_x.append(t)
    
    return (train_x, train_y)


def split_valid_set(train_x, train_y, valid_per=0.1):        
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
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 32,kernel_size = (3,3),padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters =64,kernel_size = (3,3),padding='same',activation='relu'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(filters =64,kernel_size = (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128,kernel_size =(3,3),padding = 'same',activation='relu'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128,kernel_size =(3,3),padding = 'same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
        
    model.add(Flatten())
    
    model.add(Dense(units=512,activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=256,activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=7,activation='softmax'))
    model.summary()
    
    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    # opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    return model
    
def train(train_x, train_y, save_model_path, batch_size=32, epoch=150):
    x_train, y_train, x_validation, y_validation = split_valid_set(train_x, train_y, valid_per=0.1)  
      
    x_train = x_train.reshape(x_train.shape[0],48,48,1)
    x_validation = x_validation.reshape(x_validation.shape[0],48,48,1)
      
    datagen = ImageDataGenerator(
            featurewise_center=False,               # set input mean to 0 over the dataset
            samplewise_center=False,                # set each sample mean to 0
            featurewise_std_normalization=False,    # divide inputs by std of the dataset
            samplewise_std_normalization=False,     # divide each input by its std
            zca_whitening=False,                    # apply ZCA whitening
            rotation_range=10,                      # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,                   # randomly flip images
            zoom_range = 0.05,
            vertical_flip=False)                    # randomly flip images
    
    datagen.fit(x_train)
    
    # construct model
    model = build_model()
    
    # Fit the model on the batches generated by datagen.flow().
    train_history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                                    steps_per_epoch=len(x_train) // batch_size,
                                    epochs=epoch,
                                    verbose=1,
                                    validation_data=(x_validation,y_validation))
    
    score = model.evaluate(x_train,y_train)
    
    print('\nTrain Acc:', score[1])
    
    model.save(save_model_path)

####################################################################################
#-----------------------------------for testing------------------------------------#
####################################################################################
    
def load_test(test_data_path):
    with open(test_data_path, 'r') as test_csv:
    	test = csv.reader(test_csv)
    	x_test = []
    	next(test)               #from the second line
    	for row in test:
    		t = [float(i)/255 for i in row[1].split()]
    		x_test.append(t)
    x_test = np.array(x_test)
    x_test = x_test.reshape(len(x_test),48,48,1)
    
    return x_test

def get_result(predict_prob, outputs_dir):
    result = []
    for i in range(len(predict_prob)):
        result.append(np.argmax(predict_prob[i]))
        
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        index = 0
        for element in result:
            writer.writerow([str(index), str(int(element))])
            index += 1
    

def main(args):
    train_x, train_y = load_train(args.train_data_path)
    
    if args.train:
        train(train_x, train_y, args.save_model_path, batch_size=32, epoch=200)
        
    elif args.test:
        x_test = load_test(args.test_data_path)
        model = load_model(args.save_model_path)
        prediction_prob = model.predict(x_test)
        get_result(prediction_prob, args.outputs_dir)
        
    else:
        print("Error: Argument --train or --test not found")
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Sentiment Classification")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=True,
                        dest='train', help='Input --train to Train')
    group.add_argument('--test', action='store_true',default=False,
                        dest='test', help='Input --infer to Infer')
    parser.add_argument('--train_data_path', type=str,
                        default='data/train.csv', dest='train_data_path',
                        help='training data path')
    parser.add_argument('--test_data_path', type=str,
                        default='Data/test.csv', dest='test_data_path',
                        help='Path to testing data')
    parser.add_argument('--save_model_path', type=str,
                        default='model/hw3_model_test.h5', dest='save_model_path',
                        help='saved model path')
    parser.add_argument('--results_dir', type=str,
                        default='outputs/output_test.csv', dest='outputs_dir',
                        help='prediction path')
    args = parser.parse_args()
    main(args)

