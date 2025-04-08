# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflowjs as tfjs

#Descargar el set de datos de perro y gato
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True,with_info=True)

import matplotlib.pyplot as plt
import cv2

TAMANO_IMG=100

plt.figure(figsize=(20,20))

for i, (imagen, etiqueta) in enumerate(datos['train'].take(25)):
  imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
  imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


datos_entrenamiento = []

for i, (imagen, etiqueta) in enumerate(datos['train']):
  imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
  imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
  imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG,1)
  datos_entrenamiento.append([imagen, etiqueta])

datos_entrenamiento[0]

len(datos_entrenamiento)

X = []
y=  []

for imagen, etiqueta in datos_entrenamiento:
  X.append(imagen)
  y.append(etiqueta)

import numpy as np

X = np.array(X).astype(float) / 255

y = np.array(y)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    def build_model(hp):
        model = Sequential([
            Input(shape=(100, 100, 1)),
            # Primer bloque convolucional
            Conv2D(filters=hp.Int("conv_1_filters", min_value=32, max_value=64, step=16),
                   kernel_size=(3, 3),
                   activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            # Segundo bloque convolucional
            Conv2D(filters=hp.Int("conv_2_filters", min_value=64, max_value=128, step=32),
                   kernel_size=(3, 3),
                   activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            # Tercer bloque convolucional
            Conv2D(filters=hp.Int("conv_3_filters", min_value=128, max_value=256, step=64),
                   kernel_size=(3, 3),
                   activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            # Capa de Dropout con hiperparámetro ajustable
            Dropout(hp.Float("dropout_rate", min_value=0.3, max_value=0.7, step=0.1, default=0.5)),
            Flatten(),
            # Capa densa ajustable
            Dense(hp.Int("dense_units", min_value=250, max_value=500, step=50, default=250),
                  activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Configuración del optimizador y tasa de aprendizaje
        lr = hp.Choice('lr', values=[1e-1, 1e-2, 1e-3, 1e-4])
        optimizer_choice = hp.Choice("optimizer", values=["SGD", "Adam", "Adagrad"])
        optimizers_dict = {
            "Adam":    tf.keras.optimizers.Adam(learning_rate=lr),
            "SGD":     tf.keras.optimizers.SGD(learning_rate=lr),
            "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=lr)
        }

        model.compile(optimizer=optimizers_dict[optimizer_choice],
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_accuracy", "max"),
    executions_per_trial=1,
    max_epochs=10,
    factor=3,
    directory='salida',
    project_name='intro_to_HP',
    overwrite=True,
)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(X)

X_entrenamiento = X[:19700]
X_validacion = X[19700:]
y_entrenamiento = y[:19700]
y_validacion = y[19700:]

data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)

steps_per_epoch = X_entrenamiento.shape[0] // 32

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

hist = tuner.search(
    data_gen_entrenamiento,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=(X_validacion, y_validacion),
    callbacks=callbacks
)

best_hps = tuner.get_best_hyperparameters()[0]

mi_mejor_modelo = tuner.hypermodel.build(best_hps)
mi_mejor_modelo.summary()

def plot_hist(hist):
    history = hist.history
    plt.plot(history["accuracy"], label="Entrenamiento")
    plt.plot(history["val_accuracy"], label="Validación")
    plt.title("Precisión del modelo (Accuracy)")
    plt.ylabel("Precisión")
    plt.xlabel("Época")
    plt.ylim((0, 1.1))
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("accuracy.png")
    plt.close()
    plt.plot(history["loss"], 'r', label="Entrenamiento")
    plt.plot(history["val_loss"], 'b', label="Validación")
    plt.title("Pérdida del modelo (Loss)")
    plt.ylabel("Pérdida")
    plt.xlabel("Época")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig("loss.png")
    plt.close()


historial = mi_mejor_modelo.fit(
    data_gen_entrenamiento,
    epochs=100, batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
    validation_steps=int(np.ceil(len(X_validacion) / float(32)))
)

plot_hist(historial)

mi_mejor_modelo.save('perros-gatos-cnn-ad.h5')

tfjs.converters.save_keras_model(mi_mejor_modelo, "modelo_tfjs")
