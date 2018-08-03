import keras
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import LabelEncoder

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


def load_data(data_path="../Data/Drugs_data/drug_response.csv"):
    data = pd.read_csv(data_path, index_col='Unnamed: 0')
    data.drop(labels=['Target',
                      'Primary Cell Line Name',
                      'Compound',
                      'Doses (uM)',
                      'Activity Data (median)',
                      'Activity SD',
                      'Num Data',
                      'EC50 (uM)',
                      'IC50 (uM)',
                      'Amax'], axis=1, inplace=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(data['FitType'])
    label_encoder = label_encoder.transform(data['FitType'])
    data['FitType'] = pd.DataFrame(keras.utils.to_categorical(label_encoder))
    y_data = data['ActArea']
    x_data = data.drop(['ActArea'], axis=1)
    return x_data, y_data


def main():
    data_directory = '../Data/Drugs_data/'
    compound = '17-AAG.csv'
    x_data, y_data = load_data(data_path=data_directory + compound)


if __name__ == '__main__':
    main()
