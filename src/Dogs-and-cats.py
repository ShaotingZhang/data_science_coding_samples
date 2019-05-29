import os
import cv2          # pip3 install opencv-python
import numpy as np
import tflearn
import matplotlib.pyplot as plt
import tensorflow as tf

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from random import shuffle
from tqdm import tqdm


def label_img(img):
    # change the label into dummy variables
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]

def create_train_data(train_dir, img_size):
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        if not img.endswith('.jpg'):
            continue
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    # read the grey images
        img = cv2.resize(img, (img_size, img_size))     # motify the size of images
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    return training_data

def process_test_data(test_dir, img_size):
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        if not img.endswith('.jpg'):
            continue
        path = os.path.join(test_dir, img)
        imgnum = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        testing_data.append([np.array(img), imgnum])
    shuffle(testing_data)
    return testing_data

def Model_generation(train_dir, img_size, lr):
    train_data = create_train_data(train_dir, img_size)
    tf.reset_default_graph()        # empty the graph
    convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

    # Three CNN layers and two max pooling layers
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    # Two fully connected layer and prediction layer
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

    # split the training data set
    model = tflearn.DNN(convnet, tensorboard_dir='log')
    train_now = train_data[:-500]
    test_now = train_data[-500:]

    train_in = np.array([i[0] for i in train_now], dtype=np.float64).reshape(-1, img_size, img_size, 1)
    train_out = np.array([i[1] for i in train_now], dtype=np.float64)
    test_in = np.array([i[0] for i in test_now], dtype=np.float64).reshape(-1, img_size, img_size, 1)
    test_out = np.array([i[1] for i in test_now], dtype=np.float64)

    # Train data in the training set
    model.fit({'input': train_in}, {'targets': train_out}, n_epoch=3, validation_set=({'input': test_in}, {'targets': test_out}), snapshot_step=500, show_metric=True, run_id='model')
    return model


def main():
    train_dir = '../input/train/'
    test_dir = '../input/test/'
    img_size = 50
    lr = 1e-3
    
    # Generate the model
    model = Model_generation(train_dir, img_size, lr)

    # Read the testing set
    test_data = process_test_data(test_dir, img_size)

    # Predict some example in testing set
    fig = plt.figure()
    for num, data in enumerate(test_data[:16]):
        img_num = data[1]
        img_data = data[0]
        output = fig.add_subplot(4, 4, num+1)
        orig = img_data
        data = img_data.reshape(img_size, img_size, 1)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 1:
            label = 'Dog'
        else:
            label = 'Cat'

        output.imshow(orig, cmap='gray')
        plt.title(label)
        output.axes.get_xaxis().set_visible(False)
        output.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
