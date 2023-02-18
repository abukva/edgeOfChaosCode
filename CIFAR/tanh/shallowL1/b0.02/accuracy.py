import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_accuracy(sigma_w, L):
    sigma_b = 0.02
    N = 8

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten())
    for i in range(L):
        if i == 0:
            model.add(layers.Dense(N, 
                                    activation='tanh', 
                                    kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_w/(32*32))), 
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

    optimizer = keras.optimizers.SGD(learning_rate=1e-3, momentum=0.8)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    term = tf.keras.callbacks.TerminateOnNaN()

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[term])

    return np.array(history.history['val_accuracy'])

num_classes = 10
input_shape = (32, 32, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

rgb_weights = [0.2989, 0.5870, 0.1140]

x_train = x_train[:,:,:,0] * rgb_weights[0] + x_train[:,:,:,1] * rgb_weights[1] + x_train[:,:,:,2] * rgb_weights[2]
x_test = x_test[:,:,:,0] * rgb_weights[0] + x_test[:,:,:,1] * rgb_weights[1] + x_test[:,:,:,2] * rgb_weights[2]

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

sws = np.linspace(0.1, 100, 100)

score = np.zeros((2, 100, 100))

num_rep = 50

for _ in range(num_rep):
    for i, x in enumerate(sws):
        result = get_accuracy(x, 1)
        score[0, i, :] += result
        score[1, i, :] += result**2

np.save('accuracySlice.npy', score / num_rep)
