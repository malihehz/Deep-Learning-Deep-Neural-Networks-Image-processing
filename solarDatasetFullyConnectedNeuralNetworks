rom sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import time
from keras.layers.normalization import BatchNormalization
from keras import regularizers

def load_solar_particle(skip=10, standardize = True):
    print('\nSolar particle dataset')
    x_train = np.load('x_ray_data_train.npy')[::skip]
    y_train = np.load('x_ray_target_train.npy')[::skip]
    x_test = np.load('x_ray_data_test.npy')[::skip]
    y_test = np.load('x_ray_target_test.npy')[::skip]
    if standardize:
        s = np.std(x_train,axis=0) 
        m = np.mean(x_train,axis=0) 
        x_train = x_train - m
        x_test = x_test - m
        x_train = x_train/s
        x_test = x_test/s
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":  
    #print("\n",'l111111111111111 results:',"\n")
    #m=[0.99,0.5,0.0]
    #drop = [0.2,0.5]
    #legOne = [0.1,0.01,0.001,0.0001]
    #legTwo = [0.1,0.01,0.001,0.0001]
    #legTwo = [0.00001,0.000001,0.0000001]
    #for i in legTwo:
    x_train, y_train, x_test, y_test = load_solar_particle(skip=1,standardize = False)
            
    batch_size = 64#128 
    epochs = 10#20
        
    model = Sequential()                                                                    
    model.add(Dense(28, activation='relu', input_shape=(x_train.shape[1],)
                    ,kernel_regularizer = regularizers.l1(0.000001)
                    ,kernel_initializer='he_normal'))
    #model.add(Dropout(i))
    model.add(BatchNormalization(momentum=0.5))  
    model.add(Dense(28, activation='relu'
                    ,kernel_regularizer = regularizers.l1(0.000001)
                    ,kernel_initializer='he_normal'))
    
    #model.add(Dropout(i))
    model.add(BatchNormalization(momentum=0.5))
        
    model.add(Dense(1, activation='linear'))################diff
        
        
    model.summary()
        
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.001),#Adam(lr=0.01),####diff
                  metrics=['mse'])
        
    start = time.time()
    elapsed_time = time.time()-start
        
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
            
    elapsed_time = time.time()-start
            
    score = model.evaluate(x_test, y_test, verbose=0)
    
    #print("\n",i)
    print('{0:.6f} '.format(elapsed_time)) 
    print('{0:.4f} '.format(score[0]))#Test loss:
    print('{0:.4f} '.format(score[1]))#Test mean_squared_error:
