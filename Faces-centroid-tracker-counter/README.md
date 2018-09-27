# Faces-centroid-tracker-counter
A project tracking and counting faces in real time, a paramount to build a **Person-Counter**. 
(The code comments using here is chinese)

-------------------------------------------------
## ***Process***
1. Take an initial set of detections as input, like *bounding box* generated from object-detection algorithms (R-CNN series, YOLO, SSD);
2. Give each face-detection a unique identity, and compute the centroids coordinates of them;
3. Track face-centroids by measuring the distance between the coordinates of two neighbouring frames, assume the nearest two centroids the same face and associte these two;
4. if there's new face detected during video-stream, register and identify it; and of course deregister the id if vanishing.

-------------------------------------------------
## ***Files Intro***
### —centroid_tracker.py
The script of the object-movemnt-tracking, you can run it with the command *python obj_tracker*.

### —face_tracker_counter.py
A *.dll* required to run on video-mode and to save it as *.mp4* if using **Windows** platform.

### —Parse_tracking_distance_relation.py
A simple video to test video-mode recorded by myself.

### —Parse_tracking_distance_relation.py
A simple video to test video-mode recorded by myself.

--------------------------------------------------
## ***Result***
![result](https://github.com/LZQthePlane/Object-tracker-with-opencv/blob/master/object-tracker-based-on-color/test_gif.gif)

