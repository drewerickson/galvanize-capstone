import config
import numpy as np
from dataset import DataSet, DataSet3D
from keras.layers import Input, concatenate, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import time


def precision(y_true, y_pred):
    """
    Copied from earlier version of Keras.
    Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """
    Copied from earlier version of Keras.
    Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """
    Copied from earlier version of Keras.
    Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """
    Copied from earlier version of Keras.
    Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


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


def slice_batch(x, n_gpus, part):
    """
    Code from jonilaserson on Keras issues thread #2436.
    Divide the input batch into [n_gpus] slices, and obtain slice no. [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]


def to_multi_gpu(model, n_gpus=2):
    """
    Code from jonilaserson on Keras issues thread #2436.
    Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor, 
    hence the user sees a model that behaves the same as the original.
    """
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)

    return Model(inputs=[x], outputs=[merged])


def convolve_norm_activate_2d_pass(layer00, filters=64, kernel_size=(3, 3),
                                activation='relu', border_mode='same'):
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
    return layer03


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


def model_2d_unet_full(input_shape=(240, 240, 4), categories=3,
                       pool_size=(2, 2), strides=(2, 2),
                       loss=binary_crossentropy, optimizer=Adam(), metrics=[precision, recall, fmeasure],
                       n_gpus=8):
    """
    Keras model based on UNet CNN.  Full implementation based on 2015 paper.
    Differences:
    BatchNormalization steps have been added.
    UpSampling is used instead of UpConvolution, so the filter count doesn't decrease until the next Convolution step.
    The original input (normed) is also concatenated after upscaling to the original size.

    channels: number of input data channels
    categories: number of output data categories
    optimizer: method used for stochastic gradient descent
    """
    layer00 = Input(shape=input_shape)
    layer01 = BatchNormalization()(layer00)

    layer02 = convolve_norm_activate_2d_pass(layer01, filters=64)
    layer03 = convolve_norm_activate_2d_pass(layer02, filters=64)
    layer04 = MaxPooling2D(pool_size=pool_size, strides=strides)(layer03)

    layer05 = convolve_norm_activate_2d_pass(layer04, filters=128)
    layer06 = convolve_norm_activate_2d_pass(layer05, filters=128)
    layer07 = MaxPooling2D(pool_size=pool_size, strides=strides)(layer06)

    layer08 = convolve_norm_activate_2d_pass(layer07, filters=256)
    layer09 = convolve_norm_activate_2d_pass(layer08, filters=256)
    layer10 = MaxPooling2D(pool_size=pool_size, strides=strides)(layer09)

    layer11 = convolve_norm_activate_2d_pass(layer10, filters=512)
    layer12 = convolve_norm_activate_2d_pass(layer11, filters=512)
    layer13 = MaxPooling2D(pool_size=pool_size, strides=strides)(layer12)

    layer14 = convolve_norm_activate_2d_pass(layer13, filters=1024)
    layer15 = convolve_norm_activate_2d_pass(layer14, filters=1024)
    layer16 = UpSampling2D(size=pool_size)(layer15)
    concat01 = concatenate([layer12, layer16])

    layer17 = convolve_norm_activate_2d_pass(concat01, filters=512)
    layer18 = convolve_norm_activate_2d_pass(layer17, filters=512)
    layer19 = UpSampling2D(size=pool_size)(layer18)
    concat02 = concatenate([layer09, layer19])

    layer20 = convolve_norm_activate_2d_pass(concat02, filters=256)
    layer21 = convolve_norm_activate_2d_pass(layer20, filters=256)
    layer22 = UpSampling2D(size=pool_size)(layer21)
    concat03 = concatenate([layer06, layer22])

    layer23 = convolve_norm_activate_2d_pass(concat03, filters=128)
    layer24 = convolve_norm_activate_2d_pass(layer23, filters=128)
    layer25 = UpSampling2D(size=pool_size)(layer24)
    concat04 = concatenate([layer03, layer25])

    layer26 = convolve_norm_activate_2d_pass(concat04, filters=64)
    layer27 = convolve_norm_activate_2d_pass(layer26, filters=64)
#    concat05 = concatenate([layer01, layer27])

    layer28 = Conv2D(categories, (1, 1), activation='sigmoid', border_mode='same')(layer27)

    model = Model(layer00, layer28)

    model = to_multi_gpu(model, n_gpus=n_gpus)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

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


if __name__ == '__main__':
    """
    Main script for running Keras neural network models.
    
    """
    print("Job Started!")

    model_id = "200-2d_unet_full-N-Y-N-280-001"
    local = False
    brain_cat = False
    multi_cat = True
    num_cats = 3
    hist_equal = False
    build_test = True
    set_optimizer = "adam"
    model_to_test = "2d_unet_full"  # "2d_unet", "2d_unet_with_drop", "2d_1up", "2d_1up_with_drop"
    epochs = 200
    batch_size = 112
    train = True
    run_predict = False
    save_predict = False
    is_3d = False
    n_gpus = 16

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
    print("Batch Size: ", batch_size)
    print("Model Training: ", train)
    print("Run Predictions: ", run_predict)
    print("Save Prediction Images: ", save_predict)
    print("3D Data: ", is_3d)

    if not is_3d:

        print("Initializing DataSet... ", end="", flush=True)
        ds = None
        if local:
            ds = DataSet(local_path=config.local_path)
        else:
            ds = DataSet(bucket_name=config.bucket_name, prefix_folder=config.prefix_folder)
        print("Done.")

        print("Loading Data... ", end="", flush=True)
        ds.load_dataset(config.train_pids_280, brain_cat=brain_cat, hist_equal=hist_equal, multi_cat=multi_cat)
        print("Done.")

        print("Train / Test Split... ", end="", flush=True)
        if build_test:
            X_train, X_test, y_train, y_test = ds.build_train_test_split()
        else:
            ds.test_patients = config.test_pids
            ds.train_patients = [patient for patient in ds.data.keys() if patient not in config.test_pids]
            X_train, X_test, y_train, y_test = ds.get_train_test_split()
        print("Done.")

        print("Train: ", len(X_train))
        print("Test:  ", len(X_test))
        print("Total: ", len(X_train) + len(X_test))
        print("Test Patients: ", ds.test_patients)

        print("Selecting Loss Measurement... ", end="", flush=True)
        loss = binary_crossentropy
        print("Done.")

        print("Selecting Optimizer... ", end="", flush=True)
        optimizer = None
        if set_optimizer == "sgd":
            optimizer = SGD(lr=0.00001, momentum=0.9, nesterov=True)
        elif set_optimizer == "adam":
            optimizer = Adam(lr=0.001, decay=0.001)
        print("Done.")

        print("Initializing Score Metric... ", end="", flush=True)
        metrics = [precision, recall, fmeasure]  # [f1_score]
        print("Done.")

        print("Building Checkpoint... ", end="", flush=True)
        checkpoint = ModelCheckpoint(filepath="weights-" + model_id + ".hdf5", verbose=1, save_best_only=True)
        print("Done.")

        print("Building History... ", end="", flush=True)
        loss_history = LossHistory()
        print("Done.")

        print("Initializing Model... ", end="", flush=True)
        model = None
        if model_to_test == "2d_unet_full":
            model = model_2d_unet_full(categories=num_cats,
                                       loss=loss, optimizer=optimizer, metrics=metrics,
                                       n_gpus=n_gpus)
        elif model_to_test == "2d_unet":
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
            model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=epochs, batch_size=batch_size,
                      callbacks=[checkpoint, loss_history])
            print("Done.")
        else:
            print("Loading Model... ", end="", flush=True)
            model.load_weights(config.model_folder + "/weights-" + model_id + ".hdf5")

        print("Loss History:")
        print(loss_history.logs)

        if run_predict:
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
