import tensorflow_datasets as tfds  # Get the Datasets
import matplotlib.pyplot as plt  # for plotting the images
import numpy as np  # dealing with arrays
import os  # dealing with directories

import tensorflow as tf  # Import Tensorflow
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from tensorflow import keras

#https://keras.io/api/models/
#SAMPLE/HELLOWORLD OF Deep Learning Using Keras 
'''
You can also adjust the verbosity by changing the value of TF_CPP_MIN_LOG_LEVEL:

0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def compile_model(new_model):
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    return new_model


def create_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(300, activation='relu'), #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax') #64,10 CHANGED TO 300, 100, 10
    ])

    return compile_model(new_model)


def wrangle_data(dataset, split):
    wrangled = dataset.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl))

    #https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    wrangled = wrangled.cache()
    if split == 'train':
        wrangled = wrangled.shuffle(60000)  # to ensure we generalize and don't memorize

    return wrangled.batch(64).prefetch(tf.data.AUTOTUNE)  # batch the dataset to make training faster


def TestModel(image):
    #plt.imshow(image, cmap='Greys')
    plt.imshow(image, cmap='Greys')
    plt.show()

    # img = load_img('D:/Kaggle_data/MNIST/Test/img_110.jpg', False, target_size=(img_width, img_height))

    #https://keras.io/api/preprocessing/image/#imgtoarray-function
    x = img_to_array(image)

    #https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    x = np.expand_dims(x, axis=0)


    print("***********", test_model.predict(x))
    predict_x = np.argmax(test_model.predict(x), axis=1)

    print("Predicted Values is ", predict_x)


if __name__ == '__main__':
    # https://www.tensorflow.org/datasets/catalog/mnist
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/load
    mnist_train, info = tfds.load('mnist', split='train', as_supervised=True, with_info=True)
    mnist_test = tfds.load('mnist', split='test', as_supervised=True)


    train_data = wrangle_data(mnist_train, 'train')
    test_data = wrangle_data(mnist_test, 'test')

    model = create_model()
    history = model.fit(train_data, epochs=5)
    model.evaluate(test_data)
    model.save('mnist.h5')

    # https://graphviz.org/download/
    keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

    # TESTING AGAINST THE SAVED MODEL
    test_model = load_model('mnist.h5')
    print("**************MODEL SUMMARY****************")
    print(test_model.summary())
    print("**************MODEL SUMMARY ENDS****************")

    #https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take
    for example in mnist_test.take(5):
        image = example[0]
        # label = example["label"]
        print("********* The image label is ", example[1].numpy())
        TestModel(image)

