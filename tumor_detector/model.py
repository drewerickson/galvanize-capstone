import config
import numpy as np
from dataset import DataSet, DataSet3D
from keras.layers import Input, concatenate, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import confusion_matrix
import time


def convolution_pooling_2d_pass(layer00, filters=64, kernel_size=(3, 3),
                                activation='relu', border_mode='same', pool_size=(2, 2), strides=(2, 2)):
    """
    Keras model pass for 2D convolution / pooling.
    layer00: Input data
    filters: number of neurons
    kernel_size: window of convolution
    activation: string or function for the activation step
    border_mode: how to handle convolution of data at the borders
    pool_size: grouping size for pooling
    strides: distance between pooling groups
    """
    layer01 = Conv2D(filters, kernel_size, border_mode=border_mode)(layer00)
    layer02 = BatchNormalization()(layer01)
    layer03 = Activation(activation)(layer02)
    layer04 = MaxPooling2D(pool_size=pool_size, strides=strides)(layer03)
    return layer04


def convolution_upsample__2d_pass(layer00, filters=64, kernel_size=(3, 3),
                                  activation='relu', border_mode='same', size=(2, 2)):
    """
    Keras model pass for 2D convolution / upscaling.
    layer00: Input data
    filters: number of neurons
    kernel_size: window of convolution
    activation: string or function for the activation step
    border_mode: how to handle convolution of data at the borders
    size: increase factor for upscaling
    """
    layer01 = Conv2D(filters, kernel_size, border_mode=border_mode)(layer00)
    layer02 = BatchNormalization()(layer01)
    layer03 = Activation(activation)(layer02)
    layer04 = UpSampling2D(size=size)(layer03)
    return layer04


def model_2d_unet(channels=4, categories=2, optimizer=Adam()):
    """
    Keras model based on UNet CNN.
    channels: number of input data channels
    categories: number of output data categories
    optimizer: method used for stochastic gradient descent
    """
    layer00 = Input(shape=(240, 240, channels))

    layer01 = convolution_pooling_2d_pass(layer00)
    layer02 = convolution_pooling_2d_pass(layer01)
    layer03 = convolution_pooling_2d_pass(layer02)
    layer04 = convolution_pooling_2d_pass(layer03)

    layer05 = convolution_upsample__2d_pass(layer04)
    concat01 = concatenate([layer03, layer05])
    layer06 = convolution_upsample__2d_pass(concat01)
    concat02 = concatenate([layer02, layer06])
    layer07 = convolution_upsample__2d_pass(concat02)
    concat03 = concatenate([layer01, layer07])
    layer08 = convolution_upsample__2d_pass(concat03)
    concat04 = concatenate([layer00, layer08])

    layer09 = Conv2D(64, (3, 3), border_mode='same')(concat04)
    layer10 = BatchNormalization()(layer09)
    layer11 = Activation('relu')(layer10)
    layer12 = Conv2D(categories, (1, 1), activation='sigmoid', border_mode='same')(layer11)

    model = Model(layer00, layer12)

    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


def model_2d_unet_with_drop(channels=4, categories=2, optimizer=Adam()):
    """
    Keras model based on UNet CNN.  Adds dropout layers after most of the passes.
    channels: number of input data channels
    categories: number of output data categories
    optimizer: method used for stochastic gradient descent
    """
    layer00 = Input(shape=(240, 240, channels))

    layer01 = convolution_pooling_2d_pass(layer00)
    layer02 = Dropout(0.5)(layer01)
    layer03 = convolution_pooling_2d_pass(layer02)
    layer04 = Dropout(0.4)(layer03)
    layer05 = convolution_pooling_2d_pass(layer04)
    layer06 = Dropout(0.3)(layer05)
    layer07 = convolution_pooling_2d_pass(layer06)
    layer08 = Dropout(0.2)(layer07)

    layer09 = convolution_upsample__2d_pass(layer08)
    layer10 = Dropout(0.2)(layer09)
    concat01 = concatenate([layer06, layer10])
    layer11 = convolution_upsample__2d_pass(concat01)
    layer12 = Dropout(0.2)(layer11)
    concat02 = concatenate([layer04, layer12])
    layer13 = convolution_upsample__2d_pass(concat02)
    layer14 = Dropout(0.2)(layer13)
    concat03 = concatenate([layer02, layer14])
    layer15 = convolution_upsample__2d_pass(concat03)
    layer16 = Dropout(0.2)(layer15)
    concat04 = concatenate([layer00, layer16])

    layer17 = Conv2D(64, (3, 3), border_mode='same')(concat04)
    layer18 = BatchNormalization()(layer17)
    layer19 = Activation('relu')(layer18)
    layer20 = Conv2D(categories, (1, 1), activation='sigmoid', border_mode='same')(layer19)

    model = Model(layer00, layer20)

    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


def model_2d_1up(channels=4, categories=2, optimizer=Adam()):
    """
    Keras model with a one-step upscale pass.
    channels: number of input data channels
    categories: number of output data categories
    optimizer: method used for stochastic gradient descent
    """
    layer00 = Input(shape=(240, 240, channels))

    layer01 = convolution_pooling_2d_pass(layer00, kernel_size=(5, 5))
    layer02 = convolution_pooling_2d_pass(layer01, kernel_size=(5, 5))
    layer03 = convolution_pooling_2d_pass(layer02, kernel_size=(5, 5))
    layer04 = convolution_pooling_2d_pass(layer03, kernel_size=(5, 5))

    layer05 = convolution_upsample__2d_pass(layer04, size=(16, 16))
    concat01 = concatenate([layer00, layer05])

    layer09 = Conv2D(64, (3, 3), border_mode='same')(concat01)
    layer10 = BatchNormalization()(layer09)
    layer11 = Activation('relu')(layer10)
    layer12 = Conv2D(categories, (1, 1), activation='sigmoid', border_mode='same')(layer11)

    model = Model(layer00, layer12)

    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


def model_2d_1up_with_drop(channels=4, categories=2, optimizer=Adam()):
    """
    Keras model with a one-step upscale pass.  Adds dropout layers after most of the passes.
    channels: number of input data channels
    categories: number of output data categories
    optimizer: method used for stochastic gradient descent
    """
    layer00 = Input(shape=(240, 240, channels))

    layer01 = convolution_pooling_2d_pass(layer00, kernel_size=(5, 5))
    layer02 = Dropout(0.5)(layer01)
    layer03 = convolution_pooling_2d_pass(layer02, kernel_size=(5, 5))
    layer04 = Dropout(0.5)(layer03)
    layer05 = convolution_pooling_2d_pass(layer04, kernel_size=(5, 5))
    layer06 = Dropout(0.5)(layer05)
    layer07 = convolution_pooling_2d_pass(layer06, kernel_size=(5, 5))
    layer08 = Dropout(0.5)(layer07)

    layer09 = convolution_upsample__2d_pass(layer08, size=(16, 16))
    layer10 = Dropout(0.5)(layer09)
    concat01 = concatenate([layer00, layer10])

    layer11 = Conv2D(64, (3, 3), border_mode='same')(concat01)
    layer12 = BatchNormalization()(layer11)
    layer13 = Activation('relu')(layer12)
    layer14 = Conv2D(categories, (1, 1), activation='sigmoid', border_mode='same')(layer13)

    model = Model(layer00, layer14)

    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


def metrics_table(y_true, y_pred):
    """
    Calculates the confusion matrix and dice coefficient for the given y data.
    Returns a dictionary containing the values of the metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    dice = dice_coeff(y_true, y_pred)
    return {"dice": dice, "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}


def dice_coeff(y_true, y_pred):
    """
    Calculates the Dice coefficient for the given true y and predicted y data.
    """
    return np.sum(y_pred[y_true == 1]) * 2. / (np.sum(y_pred) + np.sum(y_true))


class LossHistory(Callback):
    """
    LossHistory is a Callback object that stores the logs after each epoch of a model training session.
    """
    def __init__(self):
        self.logs = None

    def on_train_begin(self, logs={}):
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)


if __name__ == '__main__':
    """
    Main script for running Keras neural network models.
    
    """
    print("Job Started!")

    model_id = "100-2d_unet-brainN-histN-HQ-000"
    local = False
    brain_cat = False
    multi_cat = True
    num_cats = 3
    hist_equal = False
    build_test = False
    set_optimizer = "adam"
    model_to_test = "2d_unet"  # "2d_unet", "2d_unet_with_drop", "2d_1up", "2d_1up_with_drop"
    epochs = 100
    train = False
    save_predict = False
    is_3d = True

    print("Model ID: ", model_id)
    print("Local Data: ", local)
    print("Brain Mask Included: ", brain_cat)
    print("Multiple Categories: ", multi_cat)
    print("Number of Categories: ", num_cats)
    print("Histogram Equalization Used: ", hist_equal)
    print("Build Train / Test Split: ", build_test)
    print("Optimizer: ", set_optimizer)
    print("Model Name: ", model_to_test)
    print("Epochs: ", epochs)
    print("Model Training: ", train)
    print("Save Prediction Images: ", save_predict)

    if not is_3d:
        print("Initializing DataSet... ", end="", flush=True)
        ds = None
        if local:
            ds = DataSet(local_path=config.local_path)
        else:
            ds = DataSet(bucket_name=config.bucket_name, prefix_folder=config.prefix_folder)
        print("Done.")

        print("Loading Data... ", end="", flush=True)
        ds.load_dataset(config.pids_of_interest, brain_cat=brain_cat, hist_equal=hist_equal, multi_cat=multi_cat)
        print("Done.")

        print("Train / Test Split... ", end="", flush=True)
        if build_test:
            X_train, X_test, y_train, y_test = ds.build_train_test_split()
        else:
            ds.test_patients = config.test_pids
            ds.train_patients = [patient for patient in ds.data.keys() if patient not in config.test_pids]
            X_train, X_test, y_train, y_test = ds.get_train_test_split()
        print("Done.")

        print("Selecting Optimizer... ", end="", flush=True)
        optimizer = None
        if set_optimizer == "sgd":
            optimizer = SGD(lr=0.00001, momentum=0.9, nesterov=True)
        elif set_optimizer == "adam":
            optimizer = Adam(lr=0.001, decay=0.001)
        print("Done.")

        print("Initializing Score Metric... ", end="", flush=True)
        metrics = ['accuracy']  # [f1_score]
        print("Done.")

        print("Train: ", len(X_train))
        print("Test:  ", len(X_test))
        print("Total: ", len(X_train) + len(X_test))
        print("Test Patients: ", ds.test_patients)

        print("Building Checkpoint... ", end="", flush=True)
        checkpoint = ModelCheckpoint(filepath="weights-" + model_id + ".hdf5", verbose=1, save_best_only=True)
        print("Done.")

        print("Building History... ", end="", flush=True)
        loss_history = LossHistory()
        print("Done.")

        print("Initializing Model... ", end="", flush=True)
        model = None
        if model_to_test == "2d_unet":
            model = model_2d_unet(categories=num_cats, optimizer=optimizer)
        elif model_to_test == "2d_unet_with_drop":
            model = model_2d_unet_with_drop(categories=num_cats, optimizer=optimizer)
        elif model_to_test == "2d_1up":
            model = model_2d_1up(categories=num_cats, optimizer=optimizer)
        elif model_to_test == "2d_1up_with_drop":
            model = model_2d_1up_with_drop(categories=num_cats, optimizer=optimizer)
        print("Done.")

        if train:
            print(model.summary())
            print("Training Model... ", end="", flush=True)
            model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[checkpoint, loss_history])
            print("Done.")
        else:
            print("Loading Model... ", end="", flush=True)
            model.load_weights(config.model_folder + "/weights-" + model_id + ".hdf5")

        print("Loss History:")
        print(loss_history.logs)

        start = time.time()

        print("Analysing Test Predictions... ", end="", flush=True)
        ds.predict_metrics(model, metrics_table)
        print("Done.")
        print(ds.prediction_metrics)

        stop = time.time()
        print("Total Time: ", stop-start, " s")

        if save_predict:
            print("Saving Prediction Images... ", end="", flush=True)
            ds.predict(model, model_id)
            print("Done.")

        print("Saving Model... ", end="", flush=True)
        model_json = model.to_json()
        with open("model-" + model_id + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model-" + model_id + ".h5")
        print("Done.")

    else:

        print("Initializing DataSet... ", end="", flush=True)
        ds = None
        if local:
            ds = DataSet3D(local_path=config.local_path, local_out_path=config.local_out_path)
        else:
            ds = DataSet3D(bucket_name=config.bucket_name, prefix_folder=config.prefix_folder, output_folder=config.output_folder)
        print("Done.")

        print("Loading Data... ", end="", flush=True)
#        ds.load_dataset(["Brats17_2013_09_1"])  # "['Brats17_CBICA_AAB_1'])
        ds.load_dataset(config.all_pids)
        print("Done.")

        print("Selecting Optimizer... ", end="", flush=True)
        optimizer = None
        if set_optimizer == "sgd":
            optimizer = SGD(lr=0.00001, momentum=0.9, nesterov=True)
        elif set_optimizer == "adam":
            optimizer = Adam(lr=0.001, decay=0.001)
        print("Done.")

        print("Initializing Model... ", end="", flush=True)
        model = None
        if model_to_test == "2d_unet":
            model = model_2d_unet(categories=num_cats, optimizer=optimizer)
        elif model_to_test == "2d_unet_with_drop":
            model = model_2d_unet_with_drop(categories=num_cats, optimizer=optimizer)
        elif model_to_test == "2d_1up":
            model = model_2d_1up(categories=num_cats, optimizer=optimizer)
        elif model_to_test == "2d_1up_with_drop":
            model = model_2d_1up_with_drop(categories=num_cats, optimizer=optimizer)
        print("Done.")

        print("Loading Model... ", end="", flush=True)
        model.load_weights(config.model_folder + "/weights-" + model_id + ".hdf5")
        print("Done.")

        start = time.time()

#        print("Analysing Test Predictions... ", end="", flush=True)
        ds.predict_3d(model, metrics_table)
#        print("Done.")
#        print(ds.prediction_metrics)

        stop = time.time()
        print("Total Time: ", stop - start, " s")

    print("")
    print("Job Complete.")
