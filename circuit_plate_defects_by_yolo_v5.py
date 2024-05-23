import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'https%3A%2F%2Fstorage.googleapis.com%archive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-gserviceaccount.com%252F20240523%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240523T194545Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D87e89e2a18a88570119e1526cefb93e85bf3fe57c3cadc2ee7e880741baddd97b412efceb82277290bdb9c702a10cbc0118c81b0f482199a419f99510126a3b94e651702cce7bc152784f5256813cbaebef04dd3f6c641551f59d398a5f7c932ac95e2dbc60af8d092d2d664031096d75b6a3516e3343c049f11c805f841330f923a11bc35ad78ce8f1e6d51f70c73e3bcc2340ff8078e837618568718696aea1c1344ded5319ad8e6eae471441417b91bf01519608d25e4294dfa5b26362bf4eca2af30af8c44060459e9f96d77034505fa7514210266a676fdce0acb3dfdeb2891cc9a187c0424d150c3ea3a55701c3d06a763bc0798eba4d7f9a179dfcd29'

INPUT_PATH='/input'
WORKING_PATH='/working'
SYMLINK=''

!umount /input/ 2> /dev/null
shutil.rmtree('/input', ignore_errors=True)
os.makedirs(INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(INPUT_PATH, directory)
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
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

"""# Download Dataset"""

from google.colab import drive
import zipfile
import os
drive.mount('/content/drive')

zip_file_path = '/content/drive/My Drive/Colab Notebooks/archive.zip'
extracted_folder_path = '/content/data'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

print(f'ZIP The file has been successfully decompressed to "{extracted_folder_path}" The file has been successfully decompressed to the directory.')

"""Data/DATASET"""

import os
directory_path = '/content/data/DATASET'
file_list = os.listdir(directory_path)

print("Files in directory:")
for file_name in file_list:
    print(file_name)

"""# Data Preprocessing

Check JPG quantities
"""

import os

base_path = '/content/Data/DATASET/images'
folders = ['Spurious_copper', 'Mouse_bite', 'Open_circuit', 'Missing_hole', 'Spur', 'Short']

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    jpg_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    print(f"{folder}: {jpg_count} JPG files")

"""Check XML quantities"""

import os
xml_folder_path = "/content/data/DATASET/Annotations/"

xml_files = []
for folder in ['Spurious_copper', 'Mouse_bite', 'Open_circuit', 'Missing_hole', 'Spur', 'Short']:
    folder_path = os.path.join(xml_folder_path, folder)
    if os.path.isdir(folder_path):
        xml_files.extend([os.path.join(folder, f) for f in os.listdir(folder_path) if f.endswith('.xml')])

num_files = len(xml_files)
print(f"The file has been successfully decompressed to the directory. There are a total of. {num_files} There are a total of XML files")

for folder in ['Spurious_copper', 'Mouse_bite', 'Open_circuit', 'Missing_hole', 'Spur', 'Short']:
    folder_path = os.path.join(xml_folder_path, folder)
    num_files_in_folder = len([f for f in os.listdir(folder_path) if f.endswith('.xml')])
    print(f"{folder} The file has been successfully decompressed to the directory. There are a total of  {num_files_in_folder} XML files in the folder.")

!pip install Pillow

from PIL import Image
import os

original_folder = '/content/data/DATASET/images'
resized_folder = '/content/resized'
os.makedirs(resized_folder, exist_ok=True)
folders = ['Spurious_copper', 'Mouse_bite', 'Open_circuit', 'Missing_hole', 'Spur', 'Short']

for folder in folders:
    folder_path = os.path.join(original_folder, folder)
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    os.makedirs(resized_folder, exist_ok=True)
    for image in images:
        image_path = os.path.join(folder_path, image)
        img = Image.open(image_path)
        resized_img = img.resize((640, 640))
        output_path = os.path.join(resized_folder, image)
        resized_img.save(output_path)
        print(f"Processed: {image}, new file saved in: {directory} {output_path}")
print("Processed: {image}, resized and saved to the new folder.")

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def resize_xml(xml_path, output_path, target_size):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for size in root.iter('size'):
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        size.find('width').text = str(target_size)
        size.find('height').text = str(target_size)

    for obj in root.iter('object'):
        for box in obj.iter('bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)

            xmin = int(xmin * target_size / width)
            ymin = int(ymin * target_size / height)
            xmax = int(xmax * target_size / width)
            ymax = int(ymax * target_size / height)

            box.find('xmin').text = str(xmin)
            box.find('ymin').text = str(ymin)
            box.find('xmax').text = str(xmax)
            box.find('ymax').text = str(ymax)

    tree.write(output_path)

original_annotations_folder = '/content/data/DATASET/Annotations'
resized_annotations_folder = '/content/resized'
os.makedirs(resized_annotations_folder, exist_ok=True)

target_size = 640
for folder in folders:
    folder_path = os.path.join(original_annotations_folder, folder)

    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    for xml_file in xml_files:
        xml_path = os.path.join(folder_path, xml_file)

        base_filename = os.path.splitext(xml_file)[0]
        output_xml_path = os.path.join(resized_annotations_folder, f"{base_filename}.xml")
        resize_xml(xml_path, output_xml_path, target_size)

        print(f"Processed: {xml_file}, new file saved in: {output_xml_path}")

print("Completed resizing XML files and saved to a new folder.")

"""# Create train, val datasets and labels"""

import os
import random
from shutil import copyfile

source_folder = '/content/resized'

output_folder = '/content/split'
os.makedirs(output_folder, exist_ok=True)

train_ratio = 0.8
val_ratio = 0.2

for subset in ['train', 'val']:
    os.makedirs(os.path.join(output_folder, subset), exist_ok=True)

for xml_file in os.listdir(source_folder):
    if xml_file.endswith('.xml'):
        base_filename = os.path.splitext(xml_file)[0]

        rand_num = random.random()
        if rand_num < train_ratio:
            subset_folder = 'train'
        else:
            subset_folder = 'val'

        src_xml = os.path.join(source_folder, xml_file)
        dest_xml = os.path.join(output_folder, subset_folder, f'{base_filename}.xml')
        copyfile(src_xml, dest_xml)

        jpg_file = f'{base_filename}.jpg'
        src_jpg = os.path.join(source_folder, jpg_file)
        dest_jpg = os.path.join(output_folder, subset_folder, jpg_file)
        copyfile(src_jpg, dest_jpg)

import os
import xml.etree.ElementTree as ET
from PIL import Image

def convert_xml_to_yolo(xml_path, image_width, image_height, class_mapping):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            continue

        class_id = class_mapping[class_name]
        bbox = obj.find('bndbox')

        x_center = (float(bbox.find('xmin').text) + float(bbox.find('xmax').text)) / 2.0 / image_width
        y_center = (float(bbox.find('ymin').text) + float(bbox.find('ymax').text)) / 2.0 / image_height
        width = (float(bbox.find('xmax').text) - float(bbox.find('xmin').text)) / image_width
        height = (float(bbox.find('ymax').text) - float(bbox.find('ymin').text)) / image_height

        labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return labels

def create_yolo_labels(source_folder, output_folder, class_mapping):
    for xml_file in os.listdir(source_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(source_folder, xml_file)

            image_file = os.path.splitext(xml_file)[0] + '.jpg'
            image_path = os.path.join(source_folder.replace('Annotations', 'JPEGImages'), image_file)
            img = Image.open(image_path)
            image_width, image_height = img.size

            labels = convert_xml_to_yolo(xml_path, image_width, image_height, class_mapping)

            output_path = os.path.join(output_folder, os.path.splitext(xml_file)[0] + '.txt')
            with open(output_path, 'w') as f:
                f.write('\n'.join(labels))

class_mapping = {'spurious_copper': 0, 'mouse_bite': 1, 'open_circuit': 2, 'missing_hole': 3, 'spur': 4, 'short': 5}

create_yolo_labels('/content/split/train', '/content/split/train', class_mapping)
create_yolo_labels('/content/split/val', '/content/split/val', class_mapping)

"""Visual inspection"""

import os
import cv2
import matplotlib.pyplot as plt
import random

def visualize_random_image_with_labels(images_folder, labels_folder):
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    random_image_file = random.choice(image_files)
    print("Randomly selected image:", random_image_file)

    image_path = os.path.join(images_folder, random_image_file)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    label_file = os.path.splitext(random_image_file)[0] + '.txt'
    label_path = os.path.join(labels_folder, label_file)

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            img_height, img_width, _ = image.shape
            x, y, w, h = map(int, [x_center * img_width, y_center * img_height, width * img_width, height * img_height])
            x1, y1, x2, y2 = x - w//2, y - h//2, x + w//2, y + h//2
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

train_folder = '/content/split/train'
labels_folder = '/content/split/train'

visualize_random_image_with_labels(train_folder, labels_folder)

"""# YOLOv5s model"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  # Clone the YOLOv5 repository
# %cd yolov5
!pip install -U -r requirements.txt  # Install dependencies

data_yaml_content = """

train: /content/split/train
val: /content/split/val
nc: 6
names: ['spurious_copper', 'mouse_bite', 'open_circuit', 'missing_hole', 'spur', 'short']
"""

with open('/content/yolov5/data/data.yaml', 'w') as f:
    f.write(data_yaml_content)

!python train.py --img-size 640 --batch-size 16 --epochs 100 --data /content/yolov5/data/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name my_experiment --save-period 1 --project /content/yolov5/runs/

!pip install matplotlib

import pandas as pd
import matplotlib.pyplot as plt

results_path = '/content/yolov5/runs/my_experiment/results.csv'
df = pd.read_csv(results_path)

df.columns = df.columns.str.strip()

epochs = df['epoch']
train_box_loss = df['train/obj_loss']
val_box_loss = df['val/obj_loss']

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_box_loss, label='Train Object Loss', color='blue')
plt.plot(epochs, val_box_loss, label='Validation Object Loss', color='orange')

plt.title('Training and Validation Object Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Object Loss')

plt.legend()
plt.show()

"""# Create test dataset"""

import os
import shutil

source_folder = '/content/data/DATASET/rotation'
target_folder = '/content/rotation_test'

os.makedirs(target_folder, exist_ok=True)

subfolders = ['Spurious_copper_rotation', 'Mouse_bite_rotation', 'Open_circuit_rotation', 'Missing_hole_rotation']
for subfolder in subfolders:
    subfolder_path = os.path.join(source_folder, subfolder)

    for filename in os.listdir(subfolder_path):
        if filename.endswith('.jpg'):
            source_filepath = os.path.join(subfolder_path, filename)
            target_filepath = os.path.join(target_folder, filename)
            shutil.copy2(source_filepath, target_filepath)

print("File copying completed!)

from PIL import Image
import os

source_folder = '/content/rotation_test'
target_folder = '/content/rotation_test_resized'

os.makedirs(target_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith('.jpg'):
        source_filepath = os.path.join(source_folder, filename)
        img = Image.open(source_filepath)
        img_resized = img.resize((640, 640))

        target_filepath = os.path.join(target_folder, filename)
        img_resized.save(target_filepath)

print("File resizing completed!")


#"Send the JPG files in /content/rotation_test_resized to the previous model for testing.

# Feed into the previous model training --> using best.pt"

!python /content/yolov5/detect.py --weights /content/yolov5/runs/my_experiment/weights/best.pt --img-size 640 --conf 0.5 --source /content/rotation_test_resized --save-txt --save-conf --project /content/yolov5/runs/detect/

#"""Random visual inspection from test dataset"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

original_folder = '/content/rotation_test_resized'
result_folder = '/content/yolov5/runs/detect/exp'

original_files = [f for f in os.listdir(original_folder) if f.endswith('.jpg')]

selected_file = random.choice(original_files)
selected_original_filepath = os.path.join(original_folder, selected_file)

original_img = Image.open(selected_original_filepath)
result_file = os.path.join(result_folder, selected_file)
result_img = Image.open(result_file)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_img)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(result_img)
axes[1].set_title('Result Image')
axes[1].axis('off')
plt.show()
print("File name:", selected_file)

label_filepath = os.path.join(result_folder, 'labels', os.path.splitext(selected_file)[0] + '.txt')
if os.path.exists(label_filepath):
    with open(label_filepath, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split()[1:])
            print(f"Category: {int(class_id)}, Position: [{x_center:.2f}, {y_center:.2f}, {width:.2f}, {height:.2f}]")
else:
    print(f"Label file not found. {label_filepath}")