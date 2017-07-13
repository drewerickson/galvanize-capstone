import config
from dataset import DataSet
from keras.layers import Dense, Activation, Input, concatenate, BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dropout
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam, SGD


def model_3d(channels=4, categories=2, optimizer=Adam()):

    conv_kernel = (3, 3, 3)
    pool_size = (2, 2, 2)
    pool_strides = (2, 2, 2)

    layer00 = Input(shape=(240, 240, 240, channels))
    layer00_2 = BatchNormalization()(layer00)

    layer01 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(layer00_2)
    layer02 = MaxPooling3D(pool_size=pool_size, strides=pool_strides)(layer01)

    layer03 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(layer02)
    layer04 = MaxPooling3D(pool_size=pool_size, strides=pool_strides)(layer03)

    layer05 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(layer04)
    layer06 = MaxPooling3D(pool_size=pool_size, strides=pool_strides)(layer05)

    layer07 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(layer06)
    layer08 = MaxPooling3D(pool_size=pool_size, strides=pool_strides)(layer07)

    layer09 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(layer08)
    layer10 = UpSampling3D(size=pool_size)(layer09)

    concat01 = concatenate([layer07, layer10])

    layer11 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(concat01)
    layer12 = UpSampling3D(size=pool_size)(layer11)

    concat02 = concatenate([layer05, layer12])

    layer13 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(concat02)
    layer14 = UpSampling3D(size=pool_size)(layer13)

    concat03 = concatenate([layer03, layer14])

    layer15 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(concat03)
    layer16 = UpSampling3D(size=pool_size)(layer15)

    concat04 = concatenate([layer01, layer16])

    layer17 = Conv3D(64, conv_kernel, activation='relu', border_mode='same')(concat04)
    layer18 = Conv3D(categories, (1, 1, 1), activation='sigmoid', border_mode='same')(layer17)

    model = Model(layer00, layer18)

    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


def model_2d(channels=4, categories=2, optimizer=Adam()):

    layer00 = Input(shape=(240, 240, channels))
    layer00_2 = BatchNormalization()(layer00)

    layer01 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer00_2)
    layer02 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer01)

    layer03 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer02)
    layer04 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer03)

    layer05 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer04)
    layer06 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer05)

    layer07 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer06)
    layer08 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer07)

    layer09 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer08)
    layer10 = UpSampling2D(size=(2, 2))(layer09)

    concat01 = concatenate([layer07, layer10])

    layer11 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(concat01)
    layer12 = UpSampling2D(size=(2, 2))(layer11)

    concat02 = concatenate([layer05, layer12])

    layer13 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(concat02)
    layer14 = UpSampling2D(size=(2, 2))(layer13)

    concat03 = concatenate([layer03, layer14])

    layer15 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(concat03)
    layer16 = UpSampling2D(size=(2, 2))(layer15)

    concat04 = concatenate([layer01, layer16])

    layer17 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(concat04)
    layer18 = Conv2D(categories, (1, 1), activation='sigmoid', border_mode='same')(layer17)

    model = Model(layer00, layer18)

    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':

    print("Job Started!")

    model_id = "2000-005"
    local = False
    set_optimizer = "adam"
    model_to_test = "2D"
    num_cats = 4
    train = True

    print("Initializing DataSet... ", end="", flush=True)
    ds = None
    if local:
        ds = DataSet(local_path=config.local_path)
    else:
        ds = DataSet(bucket_name=config.bucket_name, prefix_folder=config.prefix_folder)
    print("Done.")

    print("Selecting Optimizer... ", end="", flush=True)
    optimizer = None
    if set_optimizer == "sgd":
        optimizer = SGD(lr=0.00001, momentum=0.9, nesterov=True)
    elif set_optimizer == "adam":
        optimizer = Adam(lr=0.0001, decay=0.001)
    print("Done.")

    print("Loading Data...", end="", flush=True)
    if model_to_test == "2D":
        ds.load_dataset()
    elif model_to_test == "3D":
        ds.load_dataset()
    print("Done.")

    print("Train: ", len(ds.index_train))
    print("Test:  ", len(ds.index_test))
    print("Total: ", len(ds.X))
    print("Files: ", ds.y_keys)
    print("Test Files: ", [ds.y_keys[i] for i in ds.index_test])

    print("Initializing Model... ", end="", flush=True)
    model = None
    if model_to_test == "2D":
        model = model_2d(categories=num_cats, optimizer=optimizer)
    elif model_to_test == "3D":
        model = model_3d(categories=num_cats, optimizer=optimizer)
    print("Done.")

    if train:
        print(model.summary())
        print("Training Model:")
        model.fit(ds.X_train(), ds.y_train(), epochs=1000, validation_data=(ds.X_test(), ds.y_test()))
        print("Done.")
    else:
        model.load_weights("model-" + model_id + ".h5")

    print("Losses: ", model.losses)
    print("Test: ", ds.index_test)
    score = model.evaluate(ds.X_test(), ds.y_test())
    print("Score: ", score)

    print("Building Prediction... ", end="", flush=True)
    ds.save_y_predict(model, model_id)
    print("Done.")

    print("Saving Model... ", end="", flush=True)
    model_json = model.to_json()
    with open("model-" + model_id + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model-" + model_id + ".h5")
    print("Done.")
    print("")
    print("Job Complete.")

#    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#    classes = model.predict(x_test, batch_size=128)

