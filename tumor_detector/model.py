import config
from dataset import DataSet
from keras.layers import Input, concatenate, BatchNormalization
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


def model_2d_with_drop(channels=4, categories=2, optimizer=Adam()):

    layer00 = Input(shape=(240, 240, channels))
    layer00_2 = BatchNormalization()(layer00)

    layer01 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer00_2)
    layer02 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer01)
    layer03 = Dropout(0.5)(layer02)

    layer04 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer03)
    layer05 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer04)
    layer06 = Dropout(0.4)(layer05)

    layer07 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer06)
    layer08 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer07)
    layer09 = Dropout(0.3)(layer08)

    layer10 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer09)
    layer11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer10)
    layer12 = Dropout(0.2)(layer11)

    layer13 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(layer12)
    layer14 = UpSampling2D(size=(2, 2))(layer13)
    layer15 = Dropout(0.2)(layer14)

    concat01 = concatenate([layer10, layer15])

    layer16 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(concat01)
    layer17 = UpSampling2D(size=(2, 2))(layer16)
    layer18 = Dropout(0.2)(layer17)

    concat02 = concatenate([layer07, layer18])

    layer19 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(concat02)
    layer20 = UpSampling2D(size=(2, 2))(layer19)
    layer21 = Dropout(0.2)(layer20)

    concat03 = concatenate([layer04, layer21])

    layer22 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(concat03)
    layer23 = UpSampling2D(size=(2, 2))(layer22)
    layer24 = Dropout(0.2)(layer23)

    concat04 = concatenate([layer01, layer24])

    layer25 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(concat04)
    layer26 = Conv2D(categories, (1, 1), activation='sigmoid', border_mode='same')(layer25)

    model = Model(layer00, layer26)

    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':

    print("Job Started!")

    model_id = "2000-drop-000"
    local = True
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

    print("Loading Data... ", end="", flush=True)
    ds.load_dataset(config.pids_of_interest)
    print("Done.")

    print("Train / Test Split... ", end="", flush=True)
    X_train, X_test, y_train, y_test = ds.train_test_split()
    print("Done.")

    print("Selecting Optimizer... ", end="", flush=True)
    optimizer = None
    if set_optimizer == "sgd":
        optimizer = SGD(lr=0.00001, momentum=0.9, nesterov=True)
    elif set_optimizer == "adam":
        optimizer = Adam(lr=0.001, decay=0.001)
    print("Done.")

    print("Train: ", len(X_train))
    print("Test:  ", len(X_test))
    print("Total: ", len(X_train) + len(X_test))
    print("Test Patients: ", ds.test_patients)

    print("Initializing Model... ", end="", flush=True)
    model = None
    if model_to_test == "2D":
        model = model_2d(categories=num_cats, optimizer=optimizer)
    elif model_to_test == "2D_drop":
        model = model_2d_with_drop(categories=num_cats, optimizer=optimizer)
    print("Done.")

    if train:
        print(model.summary())
        print("Training Model:")
        model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))
        print("Done.")
    else:
        model.load_weights("model-" + model_id + ".h5")

    print("Building Prediction... ", end="", flush=True)
    ds.predict(model, model_id)
    print("Done.")

    print("Saving Model... ", end="", flush=True)
    model_json = model.to_json()
    with open("model-" + model_id + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model-" + model_id + ".h5")
    print("Done.")
    print("")
    print("Job Complete.")
