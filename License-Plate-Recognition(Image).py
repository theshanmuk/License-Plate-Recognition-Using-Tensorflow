import os
import tensorflow as tf
import numpy as np
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util

# Define the image stream
img = cv2.imread('car.jpg')  # Change only if you have more than one webcams

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'model'
MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('model', 'label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 1

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        image_np_expanded = np.expand_dims(img, axis=0)

        # Extract image tensor
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Extract detection boxes
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Extract detection scores
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # Extract detection classes
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Extract number of detectionsd
        num_detections = detection_graph.get_tensor_by_name(
            'num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        keywords = ""
        i = 0
        while i < len(np.squeeze(scores)):
            currentScore = np.squeeze(scores)[i]
            if currentScore >= 0.75:
                currentClasses = np.squeeze(classes).astype(np.int32)[i]
                keywords += category_index[currentClasses]["name"] + ", "
                print(category_index[currentClasses]["name"])  # prints the class
                print(category_index[currentClasses])  # prints the id and name
                cv2.imwrite('C:/Users/shanmukmichael/Desktop/license-plate.jpg', img)
            i = i + 1

            # Display output
            cv2.imshow('License-Plate-Recognition', cv2.resize(img, (800, 600)))

        if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()











