import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import shutil

CHUNK_SIZE = 40960

DATA_SOURCE_MAPPING = 'plate_defects;https%3A%2F%2Fdrive.usercontent.google.com/download?id=1nWdFwAtxa1RU5LY7sab9kgfNa8yhNkUU&export=download&authuser=0&confirm=t&uuid=c58868b2-7436-4400-949a-2dca3619d719&at=APZUnTVIJujdOiV8lIzqiDDLAukQ:1717689939778'

MZE_INPUT_PATH='/mze/input'
MZE_WORKING_PATH='/mze/working'
MZE_SYMLINK='mze'

!umount /mze/input/ 2> /dev/null
shutil.rmtree('/mze/input', ignore_errors=True)
os.makedirs(MZE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(MZE_WORKING_PATH, 0o777, exist_ok=True)

try:
    os.symlink(MZE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
    pass
try:
    os.symlink(MZE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
    pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(';')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(MZE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            with ZipFile(tfile.name) as zfile:
                zfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

#three folders are needed in working environment: model_folder,result,processed_data
################################################################################################
import os

# Directory name you want to create
directory_name = "model_folder"
directory_path = "/mze/working/" + directory_name

# Create the directory
os.makedirs(directory_path)
print(f"Directory '{directory_name}' created successfully at {directory_path}")


directory_name = "result"
directory_path = "/mze/working/" + directory_name
os.makedirs(directory_path)
print(f"Directory '{directory_name}' created successfully at {directory_path}")

directory_name = "processed_data"
directory_path = "/mze/working/" + directory_name
os.makedirs(directory_path)
print(f"Directory '{directory_name}' created successfully at {directory_path}")

#This code is for testing the input image for the training model
#It does preproccessing first, then randomly display one sample of every type
#A margin function is added for a bigger bounding box
###############################################################################################
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

# Path to dataset
ANNOTATIONS_DIR = '/mze/input/plate_defects/PCB_DATASET/Annotations'
IMAGES_DIR = '/mze/input/plate_defects/PCB_DATASET/images'

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        defect_type = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append((defect_type, xmin, ymin, xmax, ymax))
    return objects

# Function to find image path
def find_image_path(image_file):
    for root, _, files in os.walk(IMAGES_DIR):
        if image_file in files:
            return os.path.join(root, image_file)
    return None

def show_cropped_image(image, defect_type, coords, margin=10):
    xmin, ymin, xmax, ymax = coords
    height, width = image.shape

    # Apply margin and ensure coordinates are within image boundaries
    xmin = max(0, xmin - margin)
    ymin = max(0, ymin - margin)
    xmax = min(width, xmax + margin)
    ymax = min(height, ymax + margin)

    cropped_img = image[ymin:ymax, xmin:xmax]
    plt.imshow(cropped_img, cmap='gray')
    plt.title(f"Defect type: {defect_type}")
    plt.axis('off')
    plt.show()

# Process each image and annotation
all_data = []
for root, dirs, files in os.walk(ANNOTATIONS_DIR):
    for annotation_file in files:
        annotation_path = os.path.join(root, annotation_file)
        image_file = annotation_file.replace('.xml', '.jpg')
        image_path = find_image_path(image_file)

        if not image_path:
            print(f"Image file does not exist: {image_file}")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        try:
            objects = parse_annotation(annotation_path)
        except ET.ParseError as e:
            print(f"Failed to parse annotation: {annotation_path}, error: {e}")
            continue

        for count, (defect_type, xmin, ymin, xmax, ymax) in enumerate(objects):
            all_data.append((image, defect_type, (xmin, ymin, xmax, ymax), image_file, count))

# Debug output to check the data collection
print(f"Total samples collected: {len(all_data)}")

# Shuffle the data to select random examples
shuffle(all_data)

# Display one example of each defect type
displayed_defects = set()
for image, defect_type, coords, img_name, count in all_data:
    if defect_type not in displayed_defects:
        print(f"Showing example for defect type: {defect_type}")
        show_cropped_image(image, defect_type, coords, margin=10)  # Increase margin as needed
        displayed_defects.add(defect_type)

    if len(displayed_defects) == 6:  # Assuming there are 6 defect types
        break

#Preproccessing the data
#Train, val and test creating in processed_data folder
#change  resized_img = cv2.resize(augmented_img, (96, 96))  if needed, this will be your model input size
################################################################################################
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Path to dataset
ANNOTATIONS_DIR = '/mze/input/plate_defects/PCB_DATASET/Annotations'
IMAGES_DIR = '/mze/input/plate_defects/PCB_DATASET/images'
OUTPUT_DIR = '/mze/working/processed_data'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.15
TEST_RATIO = 0.05

# Create directories for processed data
os.makedirs(OUTPUT_DIR, exist_ok=True)
for split in ['train', 'val', 'test']:
    for defect in ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, defect), exist_ok=True)

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        defect_type = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append((defect_type, xmin, ymin, xmax, ymax))
    return objects

def augment_and_save(image, defect_type, coords, output_dir, img_name, count, margin=10):
    xmin, ymin, xmax, ymax = coords
    height, width = image.shape

    # Apply margin and ensure coordinates are within image boundaries
    xmin = max(0, xmin - margin)
    ymin = max(0, ymin - margin)
    xmax = min(width, xmax + margin)
    ymax = min(height, ymax + margin)

    cropped_img = image[ymin:ymax, xmin:xmax]


    for i in range(5):  # Create 5 augmented images
        offset_x = np.random.randint(-10, 10)
        offset_y = np.random.randint(-10, 10)
        aug_xmin = max(0, xmin + offset_x)
        aug_ymin = max(0, ymin + offset_y)
        aug_xmax = min(image.shape[1], xmax + offset_x)
        aug_ymax = min(image.shape[0], ymax + offset_y)

        augmented_img = image[aug_ymin:aug_ymax, aug_xmin:aug_xmax]

        resized_img = cv2.resize(augmented_img, (96, 96))

        output_path = os.path.join(output_dir, defect_type, f"{img_name}_{count}_{i}.jpg")
        cv2.imwrite(output_path, resized_img)



# Function to find image path
def find_image_path(image_file):
    for root, _, files in os.walk(IMAGES_DIR):
        if image_file in files:
            return os.path.join(root, image_file)
    return None

# Process each image and annotation
all_data = []
for root, dirs, files in os.walk(ANNOTATIONS_DIR):
    for annotation_file in files:
        annotation_path = os.path.join(root, annotation_file)
        image_file = annotation_file.replace('.xml', '.jpg')
        image_path = find_image_path(image_file)

        if not image_path:
            print(f"Image file does not exist: {image_file}")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        try:
            objects = parse_annotation(annotation_path)
        except ET.ParseError as e:
            print(f"Failed to parse annotation: {annotation_path}, error: {e}")
            continue

        for count, (defect_type, xmin, ymin, xmax, ymax) in enumerate(objects):
            all_data.append((image, defect_type, (xmin, ymin, xmax, ymax), image_file, count))

# Debug output to check the data collection
print(f"Total samples collected: {len(all_data)}")

# Split data into train, val, and test
if len(all_data) > 0:
    train_val_data, test_data = train_test_split(all_data, test_size=TEST_RATIO, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=42)

    # Save augmented and resized images
    for data, split in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
        for image, defect_type, coords, img_name, count in data:
            # Remove the original file extension from img_name
            img_name = img_name.split('.')[0]
            augment_and_save(image, defect_type, coords, os.path.join(OUTPUT_DIR, split), img_name, count)
else:
    print("No data to split.")

#Original Densenet network
import tensorflow as tf

# Configure TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


from __future__ import print_function
import keras
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
import math

#Info for the code below
# target_size=(96, 96) and input_shape=(96, 96... are the size from above, below is for easier setting up
size = 96
#############################################################################################


class PyDataset(tf.data.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 7.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate



def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):

    if nb_classes==None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')

    if compression <=0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')

    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1))/dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1))//dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]

    img_input = Input(shape=input_shape)
    nb_channels = growth_rate * 2


    # Initial convolution layer
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(nb_channels, (7,7),strides=2 , use_bias=False, kernel_regularizer=l2(weight_decay))(x) #
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)#
    x = Activation('relu')(x)#
    x = ZeroPadding2D(padding=((1,1), (1, 1)))(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = 2)(x) #

    # Building dense blocks
    for block in range(dense_blocks):

        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck, weight_decay)

        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)
    x = AveragePooling2D(pool_size = 7)(x) #DECIDING LINE
    x = Flatten(data_format = 'channels_last')(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    model_name = None
    if growth_rate >= 36:
        model_name = 'widedense'
    else:
        model_name = 'dense'

    if bottleneck:
        model_name = model_name + 'b'

    if compression < 1.0:
        model_name = model_name + 'c'

    return Model(img_input, x, name=model_name), model_name


def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):

    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels


def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):

    growth_rate = nb_channels/2
    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_channels*compression), (1, 1), padding='same',
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


if __name__ == '__main__':

    # Create the DenseNet model and extract the model object from the returned tuple
    model, model_name = DenseNet(
        input_shape=(size, size, 1),
        dense_blocks=2,
        dense_layers=6,
        growth_rate=32,
        nb_classes=6,
        bottleneck=True,
        depth=27,
        weight_decay=1e-5
    )

    print(model.summary())

    # Compile the model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Create ImageDataGenerator for training data
    train_datagen = ImageDataGenerator(data_format="channels_last")
    train_generator = train_datagen.flow_from_directory(
        '/mze/working/processed_data/train',
        target_size=(size, size),
        color_mode='grayscale',
        batch_size=8
    )

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    # Learning rate scheduler callback
    lrate = LearningRateScheduler(step_decay, verbose=1)
    callbacks_list = [lrate]

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=3,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save the model every 10 epochs
    for epoch in range(1, 3):
        if epoch % 3 == 0:
            model.save(f"/mze/working/model_folder/module_epoch_{epoch}.h5")

model.save(f"/mze/working/model_folder/module_epoch_3.h5")

#Plotting the test accuracy from every epochs
#It is currently plotting every 10 epochs, if you need a more detailed graph you would need to change the variables from above.
#########################################################################################################################################
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configure TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Function to load models and evaluate on test data
def evaluate_models(model_dir, test_generator, steps):
    accuracies = []
    epochs = range(3, 3)

    for epoch in epochs:
        model_path = os.path.join(model_dir, f"module_epoch_{epoch}.h5")
        model = tf.keras.models.load_model(model_path)
        loss, accuracy = model.evaluate(test_generator, steps=steps, verbose=1)
        accuracies.append(accuracy)
        print(f"Epoch {epoch}: Accuracy = {accuracy}")

    return epochs, accuracies

if __name__ == '__main__':
    # Check the contents of the directory
    test_data_dir = '/mze/working/processed_data/test'
    if not os.path.exists(test_data_dir):
        print(f"Directory {test_data_dir} does not exist.")
    else:
        print(f"Directory {test_data_dir} exists. Contents:")
        for root, dirs, files in os.walk(test_data_dir):
            for name in files:
                print(os.path.join(root, name))

    # Create ImageDataGenerator for test data
    test_datagen = ImageDataGenerator(data_format="channels_last")
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(96, 96),
        color_mode='grayscale',
        batch_size=8,
        shuffle=False
    )

    if test_generator.n == 0:
        print("No images found in the test directory.")
    else:
        # Print the shape of a batch to debug
        for data_batch, labels_batch in test_generator:
            print("Data batch shape:", data_batch.shape)
            print("Labels batch shape:", labels_batch.shape)
            break  # Just need one batch to check

        STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

        # Directory where models are saved
        model_dir = "/mze/working/model_folder"

        # Evaluate models and get accuracies
        epochs, accuracies = evaluate_models(model_dir, test_generator, steps=STEP_SIZE_TEST)

        # Plot the accuracy graph
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.xticks(epochs)
        plt.show()

#Installing the imutils library
###########################################
!pip install imutils

#Testing code
#It currently plots every step of the image proccess, note that morphology only displays once even with two kernels
#The defect type is renamed based on your folder sorting from top to bottom
#Morphology has 3 kernel sizes, blurring, morphological opening and closing, adjust these values accordingly
#########################################
import cv2
from PIL import Image
import numpy as np
import imutils
import matplotlib.pyplot as plt
from keras.models import load_model
import os

########################################################################################
image1path = r"/mze/input/plate_defects/PCB_DATASET/PCB_USED/05.JPG"
image2path = r"/mze/input/plate_defects/PCB_DATASET/images/Short/05_short_05.jpg"
size = 96
Epochs = 1
########################################################################################

# Load images
image1 = cv2.imread(image1path)
image2 = cv2.imread(image2path)

# Apply median blur
image1_blurred = cv2.medianBlur(image1, 1)
image2_blurred = cv2.medianBlur(image2, 1)

# Display blurred images
plt.imshow(cv2.cvtColor(image1_blurred, cv2.COLOR_BGR2RGB))
plt.title("Image 1 Blurred")
plt.savefig('/mze/working/image1_blurred.png')
plt.show()

plt.imshow(cv2.cvtColor(image2_blurred, cv2.COLOR_BGR2RGB))
plt.title("Image 2 Blurred")
plt.savefig('/mze/working/image2_blurred.png')
plt.show()

# Compute the bitwise XOR of the images
image_res = cv2.bitwise_xor(image1_blurred, image2_blurred)

# Display XOR result
plt.imshow(cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB))
plt.title("Bitwise XOR Result")
plt.savefig('/mze/working/bitwise_xor_result.png')
plt.show()

# Further processing
# Adjust the kernel size for median blur
median_blur_kernel_size = 13
image_res = cv2.medianBlur(image_res, median_blur_kernel_size)

# Adjust the kernel size for morphological closing
closing_kernel_size = (10, 10)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, closing_kernel_size)
image_res_closed = cv2.morphologyEx(image_res, cv2.MORPH_CLOSE, kernel1)

# Adjust the kernel size for morphological opening
opening_kernel_size = (3, 3)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, opening_kernel_size)
image_res_opened = cv2.morphologyEx(image_res_closed, cv2.MORPH_OPEN, kernel2)


# Display morphology results
plt.imshow(cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB))
plt.title("Morphology Result")
plt.savefig('/mze/working/morphology_result.png')
plt.show()

# Detect edges
edges = cv2.Canny(image_res, 30, 200)

# Display edges
plt.imshow(edges, cmap='gray')
plt.title("Edges Detected")
plt.savefig('/mze/working/edges_detected.png')
plt.show()

# Find contours
cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
img2 = cv2.imread(image2path)

# Initialize lists
CX = []
CY = []
C = []

# Calculate moments and extract coordinates
for c in cnts:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        CX.append(cx)
        CY.append(cy)
        C.append((cx, cy))

print(CX)
print(CY)

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(20, 16))

# Load model
modulename = f"{size}x{size},{Epochs}"
modulepath = r'/mze/working/model_folder/module_epoch_3.h5'
im = Image.open(image2path)
model = load_model(modulepath)
classes = {
    0: "Missing-hole",
    1: "Mousebite",
    2: "Open-circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious-copper"
}

# Initialize prediction lists
pred = []
confidence = []

# Predict defects
for c in C:
    im1 = im.crop((c[0] - size/2, c[1] - size/2, c[0] + size/2, c[1] + size/2))
    im1 = im1.resize((size, size))
    im1 = np.array(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im1 = np.expand_dims(im1, axis=-1)
    im1 = np.expand_dims(im1, axis=0)
    print(im1.shape)
    a = model.predict(im1, verbose=1, batch_size=1)
    pred.append(np.argmax(a))
    confidence.append(a)

# Display images
axs[0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axs[1].scatter(CX, CY, c='r', s=4)
for i, txt in enumerate(pred):
    axs[1].annotate(f"{classes[txt]} ({confidence[i][0][txt]:.2f})", (CX[i], CY[i]), color='r', fontsize=12)

axs[1].axis('off')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)

# Save result
resultpath = r'/mze/working/result'
resultpath = os.path.join(resultpath, modulename)
if not os.path.isdir(resultpath):
    os.makedirs(resultpath)
plt.savefig(os.path.join(resultpath, 'result.png'), dpi=300)
plt.show()
