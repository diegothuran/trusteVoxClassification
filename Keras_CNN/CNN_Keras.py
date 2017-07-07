from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU

from keras.callbacks import ModelCheckpoint

from KEF.DataLoaders import ImageLoader_FGChallenge
from KEF.Controllers import ExperimentManager

from keras.layers.normalization import BatchNormalization

import numpy

import os

from keras import backend as K

K.set_image_dim_ordering('th')


def data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    dataDirectory = "/data/kerasFramework/CK/"

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory, "Experimento_Baseline_CohnKanade_Hyperas_64",
                                                            verbose=True)

    preProcessingProperties = [(64, 64)]

    datasetFolderTrain = "/data/datasets/Cohn-Kanade/ImagesCK_lastFrame_Face/train/"
    datasetFolderTest = "/data/datasets/Cohn-Kanade/ImagesCK_lastFrame_Face/test/"

    dataLoader = ImageLoader_FGChallenge.ImageLoader_FGChallenge(experimentManager.logManager, preProcessingProperties)
    #
    dataLoader.loadTrainData(datasetFolderTrain)
    #
    dataLoader.loadTestData(datasetFolderTest)

    dataLoader.dataTrain.dataX, dataLoader.dataTrain.dataY, dataLoader.dataTest.dataX, dataLoader.dataTest.dataY

    x_train = dataLoader.dataTrain.dataX
    y_train = dataLoader.dataTrain.dataY

    x_test = dataLoader.dataTest.dataX
    y_test = dataLoader.dataTest.dataY

    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''

    if os.path.exists("weights._hyperas_best.hdf5"):
        os.remove("weights._hyperas_best.hdf5")

    model = Sequential()
    #
    #    model.add(Convolution2D({{choice([3, 5, 10, 20, 40])}},{{choice([3, 5, 7, 11])}},{{choice([3, 5, 7, 11])}}, activation="relu"))
    #    model.add(Dropout({{uniform(0, 1)}}))
    #    model.add(MaxPooling2D(pool_size=(2,2)))

    # numberOfLayers = {{choice(['one', 'two', 'three', 'four'])}}
    numberOfLayers = {{choice(['one', 'two'])}}
    # If we choose 'four', add an additional fourth layer

    sizeK1 = {{choice([3, 5, 7, 11])}}
    numberOfFilters = {{choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])}}

    model.add(Conv2D(numberOfFilters / 8, (sizeK1, sizeK1), input_shape=(1, 64, 64)))

    activation1 = {{choice(["relu", "leaky"])}}
    if activation1 == "leaky":
        model.add(LeakyReLU(0.2))
    else:
        model.add(Activation("relu"))

    model.add(Dropout({{uniform(0, 1)}}))
    model.add(MaxPooling2D(pool_size=({{choice([2, 4, 6])}}, {{choice([2, 4, 6])}})))

    sizeK2 = {{choice([3, 5, 7, 11])}}
    model.add(Conv2D(numberOfFilters / 4, (sizeK2, sizeK2), activation={{choice(["relu", LeakyReLU(0.2)])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    sizeK3 = {{choice([3, 5, 7, 11])}}
    model.add(Conv2D(numberOfFilters / 2, (sizeK3, sizeK3), activation={{choice(["relu", LeakyReLU(0.2)])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if conditional(numberOfLayers) == 'four':
        sizeK4 = {{choice([3, 5, 7, 11])}}
        model.add(Conv2D(numberOfFilters, (sizeK4, sizeK4), activation={{choice(["relu", LeakyReLU(0.2)])}}))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    #
    #    elif   conditional(numberOfLayers) == 'four':
    #
    #        model.add(Convolution2D({{choice([3, 5, 10, 20, 40])}},{{choice([3, 5, 7, 11])}},{{choice([3, 5, 7, 11])}}, activation="relu", input_shape=(32,32,1)))
    #        model.add(Dropout({{uniform(0, 1)}}))
    #        model.add(MaxPooling2D(pool_size=(2,2)))
    #
    #        model.add(Convolution2D({{choice([3, 5, 10, 20, 40])}},{{choice([3, 5, 7])}},{{choice([3, 5, 7])}}, activation="relu"))
    #        model.add(Dropout({{uniform(0, 1)}}))
    #        model.add(MaxPooling2D(pool_size=(2,2)))
    #
    #        model.add(Convolution2D({{choice([3, 5, 10, 20, 40])}},{{choice([3, 5])}},{{choice([3, 5])}}, activation="relu"))
    #        model.add(Dropout({{uniform(0, 1)}}))
    #        model.add(MaxPooling2D(pool_size=(2,2)))
    #
    #        model.add(Convolution2D({{choice([3, 5, 10, 20, 40])}},{{choice([3, 5])}},{{choice([3, 5])}}, activation="relu"))
    #        model.add(Dropout({{uniform(0, 1)}}))
    #        model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    # model.add(Dense(50, activation="relu"))
    model.add(Dense({{choice([64, 128, 256, 512, 1024])}}, activation={{choice(["relu", "tanh"])}}, name="Dense1"))
    model.add(Dropout({{uniform(0, 1)}}))

    numberOfLayers = {{choice(['one', 'two', 'three', 'four'])}}

    if conditional(numberOfLayers) == 'two' or conditional(numberOfLayers) == 'three' or conditional(
            numberOfLayers) == 'four':
        model.add(Dense({{choice([64, 128, 256, 512, 1024])}}, activation={{choice(["relu", "tanh"])}}, name="Dense2"))

        batchNormalization2 = {{choice(['yes', 'no'])}}

        if conditional(batchNormalization2) == 'yes':
            model.add(BatchNormalization())

        model.add(Dropout({{uniform(0, 1)}}))

    if conditional(numberOfLayers) == 'three' or conditional(numberOfLayers) == 'four':
        model.add(Dense({{choice([64, 128, 256, 512, 1024])}}, activation={{choice(["relu", "tanh"])}}, name="Dense3"))

        batchNormalization3 = {{choice(['yes', 'no'])}}

        if conditional(batchNormalization3) == 'yes':
            model.add(BatchNormalization())

        model.add(Dropout({{uniform(0, 1)}}))

    if conditional(numberOfLayers) == 'four':
        model.add(Dense({{choice([64, 128, 256, 512, 1024])}}, activation={{choice(["relu", "tanh"])}}, name="Dense4"))

        batchNormalization4 = {{choice(['yes', 'no'])}}

        if conditional(batchNormalization4) == 'yes':
            model.add(BatchNormalization())

        model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(7, activation="softmax"))

    optimizer_choice = {{choice(['rmsprop', 'adam', 'sgd'])}}
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer_choice)
    model.summary()

    filepath = "weights._hyperas_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(x_train, y_train,
              batch_size=10,
              epochs=50,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks_list)

    model.load_weights("weights._hyperas_best.hdf5")

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_choice,
                  metrics=['accuracy'])

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    #    metrics = trainingHistory.history.keys()

    #    for i in range(len(metrics)):
    #
    #        if not "val" in metrics[i]:
    #            #print "Models:"+ metrics[i]+" - "+ str(trainingHistory.history[metrics[i]])
    #            #print "Models:val_"+ metrics[i]+" - "+ str(trainingHistory.history["val_"+metrics[i]])
    #            #print "-"
    #
    #            plt.plot(trainingHistory.history[metrics[i]])
    #            plt.plot(trainingHistory.history["val_"+metrics[i]])
    #
    #
    #            plt.title("Model's " + metrics[i])
    #
    #            plt.ylabel(metrics[i])
    #            plt.xlabel('epoch')
    #            plt.legend(['train', 'test'], loc='upper left')
    #            #print "Saving Plot:", self.plotsDirectory+"/"+modelName+metrics[i]+".png"
    #            plt.savefig("/data/FGChallenge/Hyperas/"+str(acc)+".png")
    #            plt.clf()
    #

    model.save("/data/kerasFramework/CK/64_Full/" + str(acc) + ".h5")

    model.summary()
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())

    best_model.save("/data/kerasFramework/CK/64_Full/BestModel.h5")
    best_model.summary()
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print("X_Train:", numpy.array(X_train).shape)
    print(best_model.evaluate(X_test, Y_test))
    print("chosen architecture:", best_run)