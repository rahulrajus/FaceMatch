import keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import PIL
from PIL import Image
from keras.layers import Dense, Activation, MaxPool2D, Conv2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential, Model, model_from_json
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from PIL import Image
import tensorflow as tf
import numpy
import face_recognition
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

def load_image(name):
    # image = face_recognition.load_image_file(name)

    # image = resize(image, (96, 96),
    #                    anti_aliasing=True)
    # face_location = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")[0]
    # top, right, bottom, left = face_location
    im = Image.open(name).convert('L')
    im = im.resize((96,96),Image.ANTIALIAS)
    # pix = numpy.array(pic.getdata()).reshape(pic.width, pic.height, 1)
    # pix = pix[top:bottom, left:right]
    # im = Image.fromarray(pix)
#    im = im.crop((left,top,right, bottom))
    # im = im.resize((96,96), Image.ANTIALIAS)

    im = numpy.array(im.getdata()).reshape(96, 96, 1)

    # print(face_location)
    #pix = pix.reshape(96, 96, 1)

    # X_tespix
    return numpy.array([im])

def load(test=False, cols=None):

    fname = "../data/face-keypoints/test.csv" if test else "../data/face-keypoints/training.csv"
    df = pd.read_csv(fname)

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols)+['Image']]

    #print( df.count())
    df = df.dropna()
    columns = df.columns

    X = np.vstack(df['Image'].values)#/255
    X = X.astype(np.float32)

    y = None

    #print(X)
    return X, y, columns

def load2d(test=False, cols=None):

    X, y, columns = load(test, cols)
    X = X.reshape(-1,96,96, 1)
    print(X.shape)

    return X, y, columns
def display(X_test, y_test):
    plt.figure(0, figsize=(12,6))
    for i in range(0, 1):
        plt.subplot(3,4,1)
        plt.imshow(X_test[i, :, :, 0], cmap="gray")
        plt.scatter(y_test[i, range(0, 30, 2)], y_test[i, range(1, 30, 2)], marker='x')

    plt.tight_layout()
    plt.show()
def run_face(model,path):
    X_test = load_image(path)
    y_test = model.predict(X_test)
    print(y_test)
    return y_test
def compare(f1,f2):
    f1_x = f1[0,range(0, 30, 2)]
    f1_y = f1[0,range(1, 30, 2)]
    f2_x = f2[0, range(0, 30, 2)]
    f2_y = f2[0, range(1, 30, 2)]
    min_x = min(f1_x)
    min_y = min(f1_y)
    f1_x -= [min_x]*len(f1_x)
    f1_y -= [min_y]*len(f1_y)

    f1_x = f1_x/max(f1_x)
    f1_y = f1_y/max(f1_y)

    min_x = min(f2_x)
    min_y = min(f2_y)
    f2_x -= [min_x]*len(f2_x)
    f2_y -= [min_y]*len(f2_y)

    f2_x = f2_x/max(f2_x)
    f2_y = f2_y/max(f2_y)


    err_x = 0.0
    err_y = 0.0
    count = 0
    for i in range(0,len(f1_x)):
        if(i != 10):

            err_x += abs(f2_x[i] - f1_x[i])

            err_y += abs(f2_y[i] - f1_y[i])
        count+=1
    err_x = err_x/count
    err_y= err_y/count
    err_x = err_x
    err_y = err_y
    w1 = (err_x+err_y)/2

    goldenRatio = (abs(f1_x[12] -f1_x[11])**2 + abs(f1_y[12] - f1_y[11])**2) / (abs(f1_x[14]-f1_x[13])**2 + abs(f1_y[14] - f1_y[13])**2)

    goldenRatio2 = (abs(f2_x[12] -f2_x[11])**2 + abs(f2_y[12] - f2_y[11])**2) / (abs(f2_x[14]-f2_x[13])**2 + abs(f2_y[14] - f2_y[13])**2)

    magic_error = abs(goldenRatio2-goldenRatio)/goldenRatio
    magic_score = 1-magic_error
    return 1000*(0.3*(1-magic_score) + 0.7*(w1))



model = Sequential()
model.add(BatchNormalization(input_shape=(96, 96, 1)))
model.add(Conv2D(24, 5, data_format="channels_last", kernel_initializer="he_normal",
                 input_shape=(96, 96, 1), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(36, 5))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(48, 5))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(64, 3))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(90))
model.add(Activation("relu"))

model.add(Dense(30))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.summary()

model.load_weights("face_model.h5")
y1 = run_face(model, "rahul2.jpg")
y2 = run_face(model, "vamsi4.jpg")
print(compare(y1,y2))
# X_test, _, __ = load2d(test=True)
# X_test_new = tf.image.rgb_to_grayscale(X_test[0])
# # print(X_test[:1].shape)
# X_test_new = X_test
