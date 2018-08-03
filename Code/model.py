import os

import keras
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

"""
    Created by Mohsen Naghipourfar on 8/1/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def create_model(n_features, layers, n_outputs, optimizer=None):
    input_layer = Input(shape=(n_features,))
    dense = Dense(layers[0], activation='relu', name="dense_0")(input_layer)
    for i, layer in enumerate(layers[1:]):
        dense = Dense(layer, activation='relu', name="dense_{0}".format(i + 1))(dense)
    dense = Dense(n_outputs, activation='sigmoid', name="output")(dense)
    model = Model(inputs=input_layer, outputs=dense)
    if optimizer is None:
        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=optimizer, loss=["mae"], metrics=["mse", "mape"])
    return model


def load_data(data_path="../Data/Drugs_data/drug_response.csv"):
    data = pd.read_csv(data_path, index_col='Unnamed: 0')
    data.drop(labels=['Target',
                      'FitType',
                      'Primary Cell Line Name',
                      'Compound',
                      'Doses (uM)',
                      'Activity Data (median)',
                      'Activity SD',
                      'Num Data',
                      'EC50 (uM)',
                      'IC50 (uM)',
                      'Amax'], axis=1, inplace=True)

    # label_encoder = LabelEncoder()
    # label_encoder.fit(data['FitType'])
    # label_encoder = label_encoder.transform(data['FitType'])
    # data['FitType'] = pd.DataFrame(keras.utils.to_categorical(label_encoder))
    y_data = data['ActArea']
    x_data = data.drop(['ActArea'], axis=1)
    return x_data, y_data


def normalize_data(x_data, y_data):
    y_data = pd.DataFrame(np.reshape(y_data.as_matrix(), (-1, 1)))
    x_data = pd.DataFrame(normalize(x_data.as_matrix(), axis=0, norm='max'))
    y_data = pd.DataFrame(normalize(y_data.as_matrix(), axis=0, norm='max'))
    return x_data, y_data


def feature_selection(x_data, y_data, k=500):
    mi = mutual_info_regression(x_data, y_data)


# def main():
#     data_directory = '../Data/Drugs_data/'
#     compounds = os.listdir(data_directory)
#     optimizers = [
#         # keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True),
#         # keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True),
#         # keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True),
#         # keras.optimizers.Adagrad(lr=0.01, decay=1e-6)
#         # keras.optimizers.Adadelta(lr=1.0, rho=0.95, decay=1e-6),
#         # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, decay=1e-6),
#         keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
#     ]
#     print("All Compounds:")
#     print(compounds)
#     for compound in compounds:
#         print("*" * 50)
#         print(compound)
#         print("Loading Data...")
#         x_data, y_data = load_data(data_path=data_directory + compound)
#         print("Data has been Loaded!")
#         x_data, y_data = normalize_data(x_data, y_data)
#         print("Data has been normalized!")
#         x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
#         print("x_train shape\t:\t" + str(x_train.shape))
#         print("y_train shape\t:\t" + str(y_train.shape))
#         print("x_test shape\t:\t" + str(x_test.shape))
#         print("y_test shape\t:\t" + str(y_test.shape))
#         for optimizer in optimizers:
#             model = create_model(x_train.shape[1], [1024, 64, 16], 1, optimizer)
#             logger_path = '../Results/'
#             if isinstance(optimizer, keras.optimizers.SGD):
#                 session = keras.backend.get_session()
#                 lr = session.run(optimizer.lr)
#                 momentum = session.run(optimizer.momentum)
#                 decay = session.run(optimizer.decay)
#                 logger_path += '%s_SGD_lr_%1.4f_momentum_%1.4f_decay_%1.6f.log' % (
#                     compound.split('.')[0], lr, momentum, decay)
#             else:
#                 logger_path += "%s_NAdam.log" % compound.split(".")[0]
#             csv_logger = CSVLogger(logger_path)
#             model.summary()
#             model.fit(x=x_train,
#                       y=y_train,
#                       batch_size=32,
#                       epochs=100,
#                       validation_data=(x_test, y_test),
#                       verbose=2,
#                       shuffle=True,
#                       callbacks=[csv_logger])
#         break


def regressor_with_k_best_features(k=50):
    data_directory = '../Data/Drugs_data/'
    compounds = os.listdir(data_directory)
    feature_names = list(pd.read_csv("../Data/BestFeatures.csv", header=None).loc[0, :])
    for compound in compounds:
        print("Loading Data...")
        x_data, y_data = load_data(data_path=data_directory + compound)
        print("Data has been Loaded!")
        x_data = x_data[feature_names]
        x_data, y_data = normalize_data(x_data, y_data)
        print("Data has been normalized!")
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
        print("x_train shape\t:\t" + str(x_train.shape))
        print("y_train shape\t:\t" + str(y_train.shape))
        print("x_test shape\t:\t" + str(x_test.shape))
        print("y_test shape\t:\t" + str(y_test.shape))


if __name__ == '__main__':
    regressor_with_k_best_features()
