import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder

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
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)

    for i, layer in enumerate(layers[1:]):
        dense = Dense(layer, activation='relu', name="dense_{0}".format(i + 1))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
    dense = Dense(n_outputs, activation='sigmoid', name="output")(dense)
    model = Model(inputs=input_layer, outputs=dense)
    if optimizer is None:
        # optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
        optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=["mse"], metrics=["mae"])
    return model


def create_classifier(n_features, layers, n_outputs):
    input_layer = Input(shape=(n_features,))
    dense = Dense(layers[0], activation='relu', name="dense_0")(input_layer)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    for i, layer in enumerate(layers[1:]):
        dense = Dense(layer, activation='relu', name="dense_{0}".format(i + 1))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
    optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, schedule_decay=0.008)
    if n_outputs > 1:
        dense = Dense(n_outputs, activation='softmax', name="output")(dense)
        loss = keras.losses.categorical_crossentropy
    else:
        dense = Dense(n_outputs, activation='sigmoid', name="output")(dense)
        loss = keras.losses.binary_crossentropy
    model = Model(inputs=input_layer, outputs=dense)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def load_data(data_path="../Data/CCLE/drug_response.csv", feature_selection=False):
    data = pd.read_csv(data_path, index_col="Cell Line")
    if data_path.__contains__("Regression"):
        y_data = data['IC50 (uM)']
        x_data = data.drop(['IC50 (uM)'], axis=1)
    else:
        y_data = data['class']
        x_data = data.drop(['class'], axis=1)
        label_encoder = LabelEncoder()
        y_data = label_encoder.fit_transform(y_data)
        y_data = np.reshape(y_data, (-1, 1))
        y_data = keras.utils.to_categorical(y_data, 2)
    if feature_selection:
        feature_names = list(pd.read_csv("../Data/BestFeatures.csv", header=None).loc[0, :])
        x_data = data[feature_names]
    return x_data, y_data


def normalize_data(x_data, y_data=None):
    x_data = pd.DataFrame(normalize(x_data.as_matrix(), axis=0, norm='max'))
    if y_data is not None:
        y_data = pd.DataFrame(np.reshape(y_data.as_matrix(), (-1, 1)))
        y_data = pd.DataFrame(normalize(y_data.as_matrix(), axis=0, norm='max'))
        return x_data, y_data
    return x_data


def feature_selection(x_data, y_data, k=500):
    mi = mutual_info_regression(x_data, y_data)


def regressor():
    data_directory = '../Data/CCLE/Regression/'
    compounds = os.listdir(data_directory)

    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        print("*" * 50)
        print(compound)
        print("Loading Data...")
        x_data, y_data = load_data(data_path=data_directory + compound, feature_selection=True)
        print("Data has been Loaded!")
        x_data, y_data = normalize_data(x_data, y_data)
        print("Data has been normalized!")
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, shuffle=True)
        print("x_train shape\t:\t" + str(x_train.shape))
        print("y_train shape\t:\t" + str(y_train.shape))
        print("x_test shape\t:\t" + str(x_test.shape))
        print("y_test shape\t:\t" + str(y_test.shape))
        # for optimizer in optimizers:
        model = create_regressor(x_train.shape[1], [1024, 256, 64, 4], 1, None)
        logger_path = '../Results/Regression/' + compound.split(".")[0] + ".log"
        csv_logger = CSVLogger(logger_path)
        model.summary()
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=32,
                  epochs=150,
                  validation_data=(x_test, y_test),
                  verbose=2,
                  shuffle=True,
                  callbacks=[csv_logger])


def regressor_with_different_optimizers():
    data_path = "../Data/CCLE/Regression/ZD-6474_preprocessed.csv"
    optimizers = [
        keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True),
        keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True),
        keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True),
        keras.optimizers.Adagrad(lr=0.01, decay=1e-6),
        keras.optimizers.Adadelta(lr=1.0, rho=0.95, decay=1e-6),
        keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, decay=1e-6),
        keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
    ]
    print("Loading Data...")
    x_data, y_data = load_data(data_path, feature_selection=True)
    print("Data has been Loaded.")
    print("Normalizing Data...")
    x_data, y_data = normalize_data(x_data, y_data)
    print("Data has been normalized.")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
    print("x_train shape\t:\t" + str(x_train.shape))
    print("y_train shape\t:\t" + str(y_train.shape))
    print("x_test shape\t:\t" + str(x_test.shape))
    print("y_test shape\t:\t" + str(y_test.shape))

    n_features = x_train.shape[1]
    layers = [1024, 256, 64, 8]
    n_outputs = 1

    for idx, optimizer in enumerate(optimizers):
        model = create_regressor(n_features, layers, n_outputs, optimizer)
        logger_path = "../Results/Optimizers/"
        optimizer_name = str(optimizer.__class__).split(".")[-1].split("\'")[0] + "_"
        optimizer_name += '_'.join(
            ["%s_%.4f" % (key, value) for (key, value) in optimizer.get_config().items()])
        optimizer_name += '.log'
        csv_logger = CSVLogger(logger_path + optimizer_name)
        model.summary()
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=32,
                  epochs=250,
                  validation_data=(x_test, y_test),
                  verbose=2,
                  shuffle=True,
                  callbacks=[csv_logger])


def regressor_with_k_best_features(k=50):
    data_directory = '../Data/CCLE/'
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
    data_directory = '../Data/CCLE/Classification/'
    compounds = os.listdir(data_directory)
    print("All Compounds:")
    print(compounds)
    for compound in compounds:
        if compound.endswith(".csv"):
            print("*" * 50)
            print(compound)
            print("Loading Data...")
            x_data, y_data = load_data(data_path=data_directory + compound, feature_selection=True)
            print("Data has been Loaded!")
            x_data = normalize_data(x_data)
            print("Data has been normalized!")
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)
            print("x_train shape\t:\t" + str(x_train.shape))
            print("y_train shape\t:\t" + str(y_train.shape))
            print("x_test shape\t:\t" + str(x_test.shape))
            print("y_test shape\t:\t" + str(y_test.shape))

            model = create_classifier(x_data.shape[1], [512, 128, 64, 16, 4], 2)
            model.summary()
            logger_path = "../Results/Classification/%s.log" % compound.split(".")[0]
            csv_logger = CSVLogger(logger_path)
            model.fit(x=x_train,
                      y=y_train,
                      batch_size=64,
                      epochs=160,
                      validation_data=(x_test, y_test),
                      verbose=2,
                      shuffle=True,
                      callbacks=[csv_logger])
            model.save(filepath="../Results/Classification/%s.h5" % compound.split(".")[0])
            result = pd.read_csv(logger_path, delimiter=',')
            plt.figure(figsize=(15, 10))
            plt.plot(result['epoch'], result["val_acc"])
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.xticks([i for i in range(0, 165, 5)])
            plt.yticks(np.arange(0, 1.05, 0.05).tolist())
            plt.title(compound.split(".")[0])
            plt.grid()
            plt.savefig("../Results/Classification/images/%s.png" % compound.split(".")[0])
            plt.close("all")
    print("Finished!")


def plot_results(path="../Results/Classification/"):
    logs = os.listdir(path)
    print(logs)
    for log in logs:
        if os.path.isfile(path + log) and log.endswith(".log"):
            result = pd.read_csv(path + log, delimiter=',')
            plt.figure(figsize=(15, 10))
            plt.plot(result['epoch'], result["val_acc"])
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title(log.split(".")[0])
            plt.savefig("../Results/Classification/images/%s.png" % log.split(".")[0])
            plt.close("all")


def plot_roc_curve(path="../Results/Classification/"):
    models = os.listdir(path)
    models = [models[i] for i in range(len(models)) if models[i].endswith(".h5")]
    for model in models:
        drug_name = model.split(".")[0]
        # print(drug_name + "\t:\t", end="")
        model = keras.models.load_model(path + model)
        x_data, y_data = load_data(data_path='../Data/CCLE/Classification/' + drug_name + ".csv",
                                   feature_selection=True)
        y_pred = model.predict(x_data.as_matrix())

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_data[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    regressor()
