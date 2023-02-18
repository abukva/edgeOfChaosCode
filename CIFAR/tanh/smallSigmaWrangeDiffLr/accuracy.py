import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class BatchAccuracy(tf.keras.callbacks.ModelCheckpoint):
    batch_accuracy = []
    def __init__(self, save_freq):
        self.model_name = "a"
        self.save_freq = save_freq
        super().__init__(self.model_name, save_freq=self.save_freq)
    
    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self.batch_accuracy.append(logs.get('accuracy'))
            
    def on_epoch_end(self, epoch, logs=None):
        self.batch_accuracy.append(logs.get('accuracy'))

def get_accuracy(sigma_w, sigma_b, L, lr):
    N = 1024

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
    epochs = 30

    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.8)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    term = tf.keras.callbacks.TerminateOnNaN()
    batch_acc = BatchAccuracy(100)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[term, batch_acc])

    return np.array(batch_acc.batch_accuracy)

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

num_ws = 20
sws = np.linspace(2.2, 6, num_ws)
bs = 9.51e-3 * (sws - 1)**2 + 0.130 * (sws - 1)**3 - 4.45e-2 * (sws - 1)**4 + 1.04e-2 * (sws - 1)**5 - 1.6e-3 * (sws - 1)**6 + 1.53e-4 * (sws - 1)**7 - 8.13e-6 * (sws - 1)**8 + 1.85e-7 * (sws - 1)**9

score = []
score2 = []

for lr in np.linspace(1e-4, 1e-3, 10):
    for i, (x, y) in enumerate(zip(sws, bs)):    
            result = get_accuracy(x, y, 100, lr)
            score += [result]
            score2 += [result**2]
        
np.savez('accuracySlice', score, score2)
