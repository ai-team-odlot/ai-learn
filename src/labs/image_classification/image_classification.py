import matplotlib.pyplot as plt_ic
import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# use default location if already downloaded
data_dir = pathlib.Path("C:\\Users\\RS7136\\.keras\\datasets\\flower_photos_extracted\\flower_photos").with_suffix('')

img_height = 180
img_width = 180

train_ds = None
val_ds = None
class_names = None
num_classes = None

epochs = 10
model = None
history = None

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape = (img_height,
                                         img_width,
                                         3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)


def ic_menu():
    print("#######################################################")
    print("# you're in image classification lab")
    print("# what do you want to do?")
    print("#\t- press 0 to download photos")
    print("#\t- press 1 to show two roses")
    print("#\t- press 2 to show two tulips")
    print("#\t- press 3 to create new dataset")
    print("#\t- press 4 to show first nine train photos")
    print("#\t- press 5 to retrieve batches of images")
    print("#\t- press 6 to optimize performance")
    print("#\t- press 7 to standardize the data")
    print("#\t- press 8 to create the model")
    print("#\t- press 9 to train the model")
    print("#\t- press 10 to show training results")
    print("#\t- press 11 to show examples of data augmentation")
    print("#\t- press 12 to create augmented and dropouted model")
    print("#\t- press 13 to predict on new data")
    print("#\t- press b to back to main menu")
    print("#######################################################")


def download_photos():
    global data_dir

    print("Downloading flower_photos.tgz archive...")

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin = dataset_url, extract = True)
    data_dir = data_dir + "\\flower_photos"  # need to add for Windows paths
    data_dir = pathlib.Path(data_dir).with_suffix('')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"The archive has been downloaded and {image_count} files has been extracted.")


def show_some_roses():
    global data_dir

    print("Showing two roses")

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0])).show()
    PIL.Image.open(str(roses[1])).show()


def show_some_tulips():
    global data_dir

    print("Showing two tulips")

    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0])).show()
    PIL.Image.open(str(tulips[1])).show()


def create_dataset():
    global train_ds, val_ds, class_names, img_height, img_width

    print("Creating dataset...")

    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset = "validation",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size)

    print("New dataset has been created.")

    class_names = train_ds.class_names
    print(f"Class names: {class_names}")


def show_first_nine_train_photos():
    global class_names

    print("Showing first nine train photos...")

    plt_ic.figure(figsize = (10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            plt_ic.subplot(3, 3, i + 1)
            plt_ic.imshow(images[i].numpy().astype("uint8"))
            plt_ic.title(class_names[labels[i]])
            plt_ic.axis("off")

plt_ic.show()

def retrieve_batches_of_images():
    global train_ds

    print("Batches of images:")
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


def optimize_performance():
    global train_ds, val_ds

    print("Configure the dataset for performance")

    print("Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch.\n"
          "This will ensure the dataset does not become a bottleneck while training your model.\n"
          "If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.")

    print("Dataset.prefetch overlaps data preprocessing and model execution while training.")

    AUTOTUNE = tf.data.AUTOTUNE

    print(f"buffer size: {AUTOTUNE}")

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)


def standardize_the_data():
    global train_ds

    print("Standardize the data")

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    print("Verify first image")
    print(f"Min value: {np.min(first_image)}")
    print(f"Max value: {np.max(first_image)}")


def create_the_model():
    global class_names, num_classes, img_height, img_width, model

    print("Creating the model...")

    num_classes = len(class_names)

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape = (img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(num_classes)
    ])

    print("Compiling the model...")
    model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])

    print("The model has been created and compiled.")

    model.summary()


def train_the_model():
    global model, history, epochs

    print("Training the model...")
    # epochs = input("Please provide number of epochs [default is 10]:")

    if epochs is None:
        epochs = 10

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = epochs
    )


def show_training_results():
    global history, epochs

    print("Showing training results...")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt_ic.figure(figsize = (8, 8))
    plt_ic.subplot(1, 2, 1)
    plt_ic.plot(epochs_range, acc, label = 'Training Accuracy')
    plt_ic.plot(epochs_range, val_acc, label = 'Validation Accuracy')
    plt_ic.legend(loc = 'lower right')
    plt_ic.title('Training and Validation Accuracy')

    plt_ic.subplot(1, 2, 2)
    plt_ic.plot(epochs_range, loss, label = 'Training Loss')
    plt_ic.plot(epochs_range, val_loss, label = 'Validation Loss')
    plt_ic.legend(loc = 'upper right')
    plt_ic.title('Training and Validation Loss')
    plt_ic.show()


def data_augmentation_example():
    global data_augmentation, train_ds

    print("Showing 10 augmented images...")

    plt_ic.figure(figsize = (10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            plt_ic.subplot(3, 3, i + 1)
            plt_ic.imshow(augmented_images[0].numpy().astype("uint8"))
            plt_ic.axis("off")

    plt_ic.show()


def create_augmented_model():
    global model, data_augmentation, num_classes

    print("Creating augmented and dropouted model...")

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(num_classes, name = "outputs")
    ])

    print("Compiling augmented and dropouted model...")

    model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])

    print("The model has been created and compiled.")
    model.summary()


def predict_on_new_data():
    global img_height, img_width, class_names

    print("Predicting using new image...")

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin = sunflower_url)

    img = tf.keras.utils.load_img(sunflower_path, target_size = (img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def ic_main():
    while True:
        ic_menu()
        action = input()
        if action == "0":
            download_photos()
        elif action == "1":
            show_some_roses()
        elif action == "2":
            show_some_tulips()
        elif action == "3":
            create_dataset()
        elif action == "4":
            show_first_nine_train_photos()
        elif action == "5":
            retrieve_batches_of_images()
        elif action == "6":
            optimize_performance()
        elif action == "7":
            standardize_the_data()
        elif action == "8":
            create_the_model()
        elif action == "9":
            train_the_model()
        elif action == "10":
            show_training_results()
        elif action == "11":
            data_augmentation_example()
        elif action == "12":
            create_augmented_model()
        elif action == "13":
            predict_on_new_data()
        elif action == "b":
            break
