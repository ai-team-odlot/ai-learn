import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import labs.cnn.gfg_ex1 as gfg_ex1


def cnn_menu():
    print("#######################################################")
    print("# you're in CNN lab")
    print("# what do you want to do?")
    print("#\t- press 0 to download and prepare images")
    print("#\t- press 1 to plot first 25 images")
    print("#\t- press 2 to create new model")
    print("#\t- press 3 to display model summary")
    print("#\t- press 4 to train model")
    print("#\t- press 5 to evaluate model")
    print("#\t- press 6 to run example #1")
    print("#\t- press 7 to run example #2")
    print("#\t- press b to back to main menu")
    print("#######################################################")


history = train_images = train_labels = test_images = test_labels = model = None


def prepare_images():
    global train_images, train_labels, test_images, test_labels

    print("Start downloading...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    print("Images has been downloaded.")

    print("Normalize pixel values to be between 0 and 1")
    train_images, test_images = train_images / 255.0, test_images / 255.0


def plot_first_25_test_images():
    if train_images is None or train_labels is None:
        print("No train images or train labels have been loaded.\n"
              "Please load or download them first and then try to display.")
        return

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Displaying 25 test images.")

    plt.figure(figsize = (10, 10))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])

        # The CIFAR labels happen to be arrays, which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])

    plt.show()

    print("Close 25 test images.")


def create_new_model():
    global model

    print("Creating new model...")

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(10))

    print("Compiling model...")

    model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])

    print("New model is ready to use.")


def display_model_summary():
    global model

    if model is None:
        print("Model has not been created.\n"
              "Please create new model first and then train it.")
        return

    model.summary()


def train_model():
    global train_images, train_labels, test_images, test_labels, history

    if train_images is None or train_labels is None or test_images is None or test_labels is None:
        print("There are no train or test images or labels loaded.\n"
              "Please load or download them first and then try to train model.")
        return

    if model is None:
        print("Model has not been created.\n"
              "Please create new model first and then train it.")
        return

    print("Start training model...")

    history = model.fit(train_images, train_labels, epochs = 30, validation_data = (test_images, test_labels))

    print("Model has been trained.")


def evaluate_model():
    global history

    if history is None:
        print("History has not been created.\n"
              "Please train model first and then evaluate it.")
        return

    print("Evaluating model...")

    plt.plot(history.history['accuracy'], label = 'accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc = 'lower right')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)

    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    plt.show()


def cnn_main():
    while True:
        cnn_menu()
        action = input()
        if action == "0":
            prepare_images()
        elif action == "1":
            plot_first_25_test_images()
        elif action == "2":
            create_new_model()
        elif action == "3":
            display_model_summary()
        elif action == "4":
            train_model()
        elif action == "5":
            evaluate_model()
        elif action == "6":
            gfg_ex1.run_gfg_ex1()
        elif action == "b":
            break
