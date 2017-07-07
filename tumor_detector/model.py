import config
from dataset import DataSet
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dropout
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD


def model_3D_full(learning_rate=0.01, batch_momentum=0.9):

    model = Sequential()

    # 1st layer group
    model.add(Conv3D(16, (3, 3, 2), name='conv1', activation='relu', border_mode='same', subsample=(1, 1, 1), input_shape=(240, 240, 155, 1)))
    model.add(MaxPooling3D(name='pool1', pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

    # 2nd layer group
    model.add(Conv3D(16, (3, 3, 3), name='conv2', activation='relu', border_mode='same', subsample=(1, 1, 1)))
    model.add(MaxPooling3D(name='pool2', pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

    # 3rd layer group
    model.add(Conv3D(16, (3, 3, 2), name='conv3', activation='relu', border_mode='same', subsample=(1, 1, 1)))
    model.add(MaxPooling3D(name='pool3', pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

    # 4th layer group
    model.add(Conv3D(16, (3, 3, 3), name='conv4', activation='relu', border_mode='same', subsample=(1, 1, 1)))
    model.add(MaxPooling3D(name='pool4', pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

    # 5th layer group
    model.add(Conv3D(16, (3, 3, 3), name='conv5', activation='relu', border_mode='same', subsample=(1, 1, 1)))
    model.add(MaxPooling3D(name='pool5', pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

    # FC layers group
    model.add(Dense(16, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(16, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(1, activation='softmax', name='fc8'))

    model.add(UpSampling3D(size=(30, 30, 31)))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=SGD(lr=learning_rate, momentum=batch_momentum, nesterov=True))
    print(model.summary())

    return model


def model_3D_slim(learning_rate=0.01, batch_momentum=0.9):

    model = Sequential()

    model.add(Conv3D(16, (5, 5, 5), name='conv', activation='relu', border_mode='same', subsample=(1, 1, 1), input_shape=(240, 240, 155, 1)))
    model.add(MaxPooling3D(name='pool', pool_size=(5, 5, 5), strides=(5, 5, 5), padding='same'))
    model.add(Dense(1, activation='softmax', name='fc'))
    model.add(UpSampling3D(size=(5, 5, 5)))
    model.add(Activation('sigmoid'))

    model.compile(loss=binary_crossentropy,
                  optimizer=SGD(lr=learning_rate, momentum=batch_momentum, nesterov=True))
    print(model.summary())

    return model


def model_2D_slim(learning_rate=0.01, batch_momentum=0.9):

    model = Sequential()

    model.add(Conv2D(16, (5, 5), name='conv',
                     activation='relu', border_mode='same', subsample=(1, 1),
                     input_shape=(240, 240, 1)))
    model.add(MaxPooling2D(name='pool',
                           pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Dense(1, name='fc', activation='softmax'))
    model.add(UpSampling2D(size=(5, 5)))
    model.add(Activation('sigmoid'))

    model.compile(loss=binary_crossentropy,
                  optimizer=SGD(lr=learning_rate, momentum=batch_momentum, nesterov=True))
    print(model.summary())

    return model

if __name__ == '__main__':

    model_to_test = "2D_slim"

    ds = DataSet(local_path=config.local_path)

    if model_to_test == "2D_slim":
        ds.load_dataset(local=True, all_dims=False, multi_cat=False)
        model = model_2D_slim()
        model.fit(ds.X, ds.y, epochs=3)
    elif model_to_test == "3D_slim":
        model = model_3D_slim()
        model.fit(ds.X, ds.y, epochs=5, batch_size=32)

    print("Losses: ", model.losses)

#    y_predict = model.predict(ds.X)
#    print(y_predict)

    score = model.evaluate(ds.X, ds.y)
    print("Score: ", score)

#    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#    classes = model.predict(x_test, batch_size=128)

