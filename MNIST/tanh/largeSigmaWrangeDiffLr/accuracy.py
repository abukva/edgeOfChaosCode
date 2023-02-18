import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_accuracy(sigma_w, sigma_b, lr):
    N = 784
    L = 100

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten())
    for i in range(L):
        if i == 0:
            model.add(layers.Dense(N, 
                                   activation='tanh', 
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_w/(28*28))), 
                                   bias_initializer=keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_b))))
        else:
            model.add(layers.Dense(N, 
                                   activation='tanh', 
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_w/N)), 
                                   bias_initializer=keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_b))))

    model.add(layers.Dense(num_classes, 
                           activation='softmax', 
                           kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_w/N)), 
                           bias_initializer=keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_b))))

    batch_size = 64
    epochs = 100

    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.8)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    term = tf.keras.callbacks.TerminateOnNaN()

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[term])

    return np.array(history.history['val_accuracy'])

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

eoc_line = np.load('EOCline100.npz')
sws = eoc_line['arr_0'][0]
bs = eoc_line['arr_0'][1]

num_ws = sws.shape[0]

lrs = np.geomspace(1e-7, 1e-2, 20)

score = np.zeros((2, num_ws, lrs.shape[0], 100))

for i, (x, y) in enumerate(zip(sws, bs)):    
    for j, lr in enumerate(lrs):
        result = get_accuracy(x, y, lr)
        score[0,i,j,:] += result
        score[1,i,j,:] += result**2
        
np.save('accuracySlice.npy', score)
