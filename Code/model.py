import os

import keras
import matplotlib.pyplot as plt
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


def create_regressor(n_features, layers, n_outputs, optimizer=None):
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


def create_classifier(n_features, layers, n_outputs):
    input_layer = Input(shape=(n_features,))
    dense = Dense(layers[0], activation='relu', name="dense_0")(input_layer)
    for i, layer in enumerate(layers[1:]):
        dense = Dense(layer, activation='relu', name="dense_{0}".format(i + 1))(dense)
    dense = Dense(n_outputs, activation='sigmoid', name="output")(dense)
    model = Model(inputs=input_layer, outputs=dense)
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_data(data_path="../Data/Drugs_data/drug_response.csv"):
    if data_path.endswith("_classif.csv"):
        data = pd.read_csv(data_path, index_col='Unnamed: 0.1')
    else:
        data = pd.read_csv(data_path, index_col='Unnamed: 0')
    data.drop(labels=['Target',
                      'FitType',
                      'Primary Cell Line Name',
                      'Compound',
                      'Doses (uM)',
                      'Activity Data (median)',
                      'Activity SD',
                      'Num Data',
                      'Amax',
                      ], axis=1, inplace=True)

    if data.columns.__contains__("EC50 (uM)"):
        data.drop(labels=['EC50 (uM)',
                          'IC50 (uM)'], axis=1, inplace=True)

    # label_encoder = LabelEncoder()
    # label_encoder.fit(data['FitType'])
    # label_encoder = label_encoder.transform(data['FitType'])
    # data['FitType'] = pd.DataFrame(keras.utils.to_categorical(label_encoder))
    if data_path.endswith("_classif.csv"):
        y_data = data['class']
        x_data = data.drop(['ActArea', 'class'], axis=1)
    else:
        y_data = data['ActArea']
        x_data = data.drop(['ActArea'], axis=1)
    return x_data, y_data


def normalize_data(x_data, y_data=None):
    x_data = pd.DataFrame(normalize(x_data.as_matrix(), axis=0, norm='max'))
    if y_data:
        y_data = pd.DataFrame(np.reshape(y_data.as_matrix(), (-1, 1)))
        y_data = pd.DataFrame(normalize(y_data.as_matrix(), axis=0, norm='max'))
        return x_data, y_data
    return x_data


def feature_selection(x_data, y_data, k=500):
    mi = mutual_info_regression(x_data, y_data)


def main():
    data_directory = '../Data/Drugs_data/'
    compounds = os.listdir(data_directory)
    optimizers = [
        # keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True),
        # keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True),
        # keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True),
        # keras.optimizers.Adagrad(lr=0.01, decay=1e-6)
        # keras.optimizers.Adadelta(lr=1.0, rho=0.95, decay=1e-6),
        # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, decay=1e-6),
        keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
    ]
    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        print("*" * 50)
        print(compound)
        print("Loading Data...")
        x_data, y_data = load_data(data_path=data_directory + compound)
        print("Data has been Loaded!")
        x_data, y_data = normalize_data(x_data, y_data)
        print("Data has been normalized!")
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
        print("x_train shape\t:\t" + str(x_train.shape))
        print("y_train shape\t:\t" + str(y_train.shape))
        print("x_test shape\t:\t" + str(x_test.shape))
        print("y_test shape\t:\t" + str(y_test.shape))
        for optimizer in optimizers:
            model = create_regressor(x_train.shape[1], [1024, 64, 16], 1, optimizer)
            logger_path = '../Results/'
            if isinstance(optimizer, keras.optimizers.SGD):
                session = keras.backend.get_session()
                lr = session.run(optimizer.lr)
                momentum = session.run(optimizer.momentum)
                decay = session.run(optimizer.decay)
                logger_path += '%s_SGD_lr_%1.4f_momentum_%1.4f_decay_%1.6f.log' % (
                    compound.split('.')[0], lr, momentum, decay)
            else:
                logger_path += "%s_NAdam.log" % compound.split(".")[0]
            csv_logger = CSVLogger(logger_path)
            model.summary()
            model.fit(x=x_train,
                      y=y_train,
                      batch_size=32,
                      epochs=100,
                      validation_data=(x_test, y_test),
                      verbose=2,
                      shuffle=True,
                      callbacks=[csv_logger])
        break


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

        for k in [50, 40, 30, 20, 10, 5, 4, 3, 2, 1]:
            model = create_regressor(x_train.shape[1], [32, 16, 4], 1)
            dir_name = "../Results/Drugs/%s/%dFeaturesSelection" % (compound.split(".")[0], k)
            os.makedirs(dir_name)
            csv_logger = CSVLogger(dir_name + '/best_%s_%d.log' % (compound.split(".")[0], k))
            model.fit(x=x_train,
                      y=y_train,
                      batch_size=64,
                      epochs=250,
                      validation_data=(x_test, y_test),
                      verbose=2,
                      shuffle=True,
                      callbacks=[csv_logger])
            import csv
            with open("../Results/Drugs/%s/%s.csv" % (compound.split(".")[0], compound.split(".")[0]), 'a') as file:
                writer = csv.writer(file)
                loss = model.evaluate(x_test.as_matrix(), y_test.as_matrix(), verbose=0)
                loss.insert(0, k)
                writer.writerow(loss)
        df = pd.read_csv("../Results/Drugs/%s/%s.csv" % (compound.split(".")[0], compound.split(".")[0]), header=None)
        plt.figure()
        plt.plot(df[0], df[1], "-o")
        plt.xlabel("# of Features")
        plt.ylabel("Mean Absolute Error")
        plt.title(compound.split(".")[0])
        plt.savefig("../Results/Drugs/%s/%s.png" % (compound.split(".")[0], compound.split(".")[0]))


def classifier():
    data_directory = '../Data/Drugs_data/'
    compounds = os.listdir(data_directory)
    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        if compound.endswith("_classif.csv"):
            print("*" * 50)
            print(compound)
            print("Loading Data...")
            x_data, y_data = load_data(data_path=data_directory + compound)
            print("Data has been Loaded!")
            x_data = normalize_data(x_data)
            # label_encoder = LabelEncoder()
            # label_encoder.fit(y_data)
            # y_data = label_encoder.transform(y_data)
            # y_data = keras.utils.to_categorical(y_data.as_matrix(), 1)
            print("Data has been normalized!")
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
            print("x_train shape\t:\t" + str(x_train.shape))
            print("y_train shape\t:\t" + str(y_train.shape))
            print("x_test shape\t:\t" + str(x_test.shape))
            print("y_test shape\t:\t" + str(y_test.shape))

            model = create_classifier(x_data.shape[1], [1024, 128, 16], 1)
            model.summary()
            csv_logger = CSVLogger("../Results/Classifier/%s.log" % compound.split(".")[0])
            model.fit(x=x_train,
                      y=y_train,
                      batch_size=64,
                      epochs=100,
                      validation_data=(x_test, y_test),
                      verbose=2,
                      shuffle=True,
                      callbacks=[csv_logger])


if __name__ == '__main__':
    classifier()
