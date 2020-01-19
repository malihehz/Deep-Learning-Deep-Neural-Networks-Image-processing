from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD,Adam
import numpy as np
import time
from keras import regularizers
from keras.layers.normalization import BatchNormalization

def load_gamma_ray(standardize = True):
    x = []
    y = []
    infile = open("magic04.txt","r")
    for line in infile:
        y.append(int(line[-2:-1] =='g'))
        x.append(np.fromstring(line[:-2], dtype=float,sep=','))
    infile.close()
    x = np.array(x).astype(np.float32)
    y = np.array(y) 
    ind = np.random.permutation(len(y))
    split_ind = int(len(y)*0.8)
    x_train= x[ind[:split_ind]]
    x_test = x[ind[split_ind:]]
    y_train = y[ind[:split_ind]]
    y_test = y[ind[split_ind:]]
    if standardize:
        s = np.std(x_train,axis=0) 
        m = np.mean(x_train,axis=0) 
        x_train = x_train - m
        x_test = x_test - m
        x_train = x_train/s
        x_test = x_test/s
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":  
    
    x_train, y_train, x_test, y_test = load_gamma_ray()

    batch_size = 128
    epochs = 20

    model = Sequential()

    model.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
    #model.add(Dropout(0.5))
    #model.add(BatchNormalization())
    
    model.add(Dense(10, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(BatchNormalization())
    
    model.add(Dense(1, activation='sigmoid'))
 

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=0.01),
                  metrics=['accuracy'])
    
    start = time.time()
    elapsed_time = time.time()-start
    
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

    elapsed_time = time.time()-start
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('{0:.6f} '.format(elapsed_time)) 
    print('{0:.4f} '.format(score[0])) #Test loss:
    print('{0:.4f} '.format(score[1])) #Test accuracy:
