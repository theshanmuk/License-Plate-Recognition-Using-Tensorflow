# License-Plate-Recognition Using Tensorflow :mag_right:
The project developed using TensorFlow to detect the **License-Plate** with some accuracy.

# Dependencies
- TensorFlow 1.0
- OpenCV
- NumPy

# Training Custom Object Detector
- Setting up the [Object_Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) directory structure in an Environment
- Gathering Data
- **label** pictures using [labelImg](https://github.com/tzutalin/labelImg)
- Creating a **labelmap** 
- Making **xml** Data
- Conveting **xml to CSV** & **CSV to TFRecord** format
- **Train** by using Pre-trained models provided by **TensorFlow**
- Monitor Training Job Progress using **TensorBoard**
- Exporting the **inference graph**
- Using the two Important **frozen_inference_graph.pb** & **label_map.pbtxt** files we can Detect Object

# Detecting Object Using 
- Image
- Camera
- Video Clip

# Demo - Using Image

![Detection - 99%](https://user-images.githubusercontent.com/55943851/78575099-82767080-7848-11ea-8eb5-4b47a47fa89a.jpeg)

# Demo - Using Camera

![Detection - 96%](https://user-images.githubusercontent.com/55943851/78575120-8bffd880-7848-11ea-90d2-dacd690f6579.jpeg)




  
