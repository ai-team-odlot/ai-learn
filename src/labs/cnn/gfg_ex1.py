import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt_ex1
from itertools import product


def run_ex1():
    # set the param
    plt_ex1.rc('figure', autolayout = True)
    plt_ex1.rc('image', cmap = 'magma')

    # define the kernel
    kernel = tf.constant([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])

    # # load the image
    image = tf.io.read_file('C:/Users/RS7136/workspace/machine_learning_tests/src/pics/Ganesh.jpg')
    image = tf.io.decode_jpeg(image, channels = 1)
    image = tf.image.resize(image, size = [300, 300])

    # plot the image
    img = tf.squeeze(image).numpy()
    plt_ex1.figure(figsize = (5, 5))
    plt_ex1.imshow(img, cmap = 'gray')
    plt_ex1.axis('off')
    plt_ex1.title('Original Gray Scale image')
    plt_ex1.show()

    # Reformat
    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    image = tf.expand_dims(image, axis = 0)
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    kernel = tf.cast(kernel, dtype = tf.float32)

    # convolution layer
    conv_fn = tf.nn.conv2d

    image_filter = conv_fn(input = image,
                           filters = kernel,
                           strides = 1,  # or (1, 1)
                           padding = 'SAME',
                           )

    plt_ex1.figure(figsize = (15, 5))

    # Plot the convolved image
    plt_ex1.subplot(1, 3, 1)
    plt_ex1.imshow(tf.squeeze(image_filter))
    plt_ex1.axis('off')
    plt_ex1.title('Convolution')

    # activation layer
    relu_fn = tf.nn.relu

    # Image detection
    image_detect = relu_fn(image_filter)
    plt_ex1.subplot(1, 3, 2)
    plt_ex1.imshow(tf.squeeze(image_detect))
    plt_ex1.axis('off')
    plt_ex1.title('Activation')

    # Pooling layer
    pool = tf.nn.pool
    image_condense = pool(input = image_detect,
                          window_shape = (2, 2),
                          pooling_type = 'MAX',
                          strides = (2, 2),
                          padding = 'SAME')

    plt_ex1.subplot(1, 3, 3)
    plt_ex1.imshow(tf.squeeze(image_condense))
    plt_ex1.axis('off')
    plt_ex1.title('Pooling')
    plt_ex1.show()
