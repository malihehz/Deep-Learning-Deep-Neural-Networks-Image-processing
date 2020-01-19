import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import RMSprop
import numpy as np
import time
from keras import regularizers
from keras.layers.normalization import BatchNormalization

#print('L222222222222222222222222222222222222222222222222222')
momentm=[0.99,0.5,0.0]
dec = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
betOne = [0.0,0.5,0.9,0.99,0.999]
LOne = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
#for i in LOne:
batch_size = 128
epochs = 10
    
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = np.max(y_train)+1
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
    
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
    
model = Sequential()
model.add(Conv2D(16,(3, 3), input_shape=input_shape
                 ,activation='relu'
                 ,kernel_regularizer = regularizers.l2(0.001)
                 ,kernel_initializer='glorot_normal'))
model.add(Conv2D(16,(3, 3),activation='relu'
                 ,kernel_regularizer = regularizers.l2(0.001)
                 ,kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization(momentum=i))

model.add(Conv2D(32,(3, 3), activation='relu'
                 ,kernel_regularizer = regularizers.l2(0.001)
                 ,kernel_initializer='glorot_normal'))
model.add(Conv2D(32,(3, 3), activation='relu'
                 ,kernel_regularizer = regularizers.l2(0.001)
                 ,kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization(momentum=i))

model.add(Flatten())
model.add(Dense(124, activation='elu'))#relu
model.add(Dense(num_classes, activation='softmax'))

model.summary()
    
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.adam(beta_1=0.0),#RMSprop
              metrics=['accuracy'])
    
start = time.time()
elapsed_time = time.time()-start
        
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
elapsed_time = time.time()-start
        
score = model.evaluate(x_test, y_test, verbose=0)
    
#print("\n",i)
print('{0:.6f} '.format(elapsed_time)) 
print('{0:.4f} '.format(score[0]))#Test loss:
print('{0:.4f} '.format(score[1]))#Test accuracy:
