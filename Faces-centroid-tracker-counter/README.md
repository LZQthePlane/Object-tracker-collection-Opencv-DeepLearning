# Faces-centroid-tracker-counter
A project tracking and counting faces in real time, a paramount to build a ***Person-Counter***. 
(The code comments using here is chinese)

-------------------------------------------------
## ***Process***
1. Take an initial set of detections as input, like *bounding box* generated from object-detection algorithms (R-CNN series, YOLO, SSD);

2. Give each face-detection a unique identity, and compute the centroids coordinates of them;

3. Track face-centroids by measuring the distance between the coordinates of two neighbouring frames, assume the nearest two centroids the same face and associte these two;

4. if there's new face detected during shooting, register and identify it; and of course deregister the id if vanishing.

-------------------------------------------------
## ***Files Intro***

### —face_tracker_counter.py
Main script: detecting, tracking and visualizing.

### —centroid_tracker.py
This script definites several *tracking functions* which is important for running *face_tracker_counter.py*.

### —Parse_tracking_distance_relation.py
A detaching programm just to explain a complicate code block.

### —ResNetSSD
Storing the *.prototxt* and *.caffemodel* files which are necessary for face detection.

--------------------------------------------------
## ***Result***
![result](https://github.com/LZQthePlane/Object-tracker-collection-Opencv-DeepLearning/blob/master/Faces-centroid-tracker-counter/test_out/example.gif)

