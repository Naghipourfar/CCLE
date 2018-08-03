import keras
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model

"""
    Created by Mohsen Naghipourfar on 8/1/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def create_model(n_features, layers):
    input_layer = Input(shape=(n_features,))
    dense = Dense(layers[0], activation='relu', name="dense_0")(input_layer)
    for i, layer in enumerate(layers[1:]):
        dense = Dense(layer, activation='relu', name="dense_{0}".format(i))(dense)
    model = Model(inputs=input_layer, outputs=dense)
    model.compile(optimizer=keras.optimizers.adam, loss=keras.losses.mse)
    return model


def load_data(data_path="../Data/"):
    data = pd.read_csv(data_path)

    return None, None


def main():
    x_data, y_data = load_data()


if __name__ == '__main__':
    main()
