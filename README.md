# Object-tracker-collection-Opencv-DeepLearning
Collection of object-tracker projects of hot topics, with opencv and deep learning (on some projects).   
(The code comments using here is chinese)

----------------------------------------------------
## requirements
- ***python3***    
- ***opencv***  (version 3.4 or above is better):  `pip install opencv-python`   
- ***imutils***:  `pip install imutils`   
- ***opencv-contrib***: (for *Faster-object-tracker-with-opencv-constructor*, [see installation here](https://github.com/LZQthePlane/Object-tracker-collection-Opencv-DeepLearning/blob/master/Faster-object-tracker-with-opencv-constructor/README.md))
- ***scipy***   

----------------------------------------------------
## object-tracker-based-on-color
The implemetation of object-movemnt-tracking based on color (green in this example).

![result](https://github.com/LZQthePlane/Object-tracker-with-opencv/blob/master/object-tracker-based-on-color/test_gif.gif)

----------------------------------------------------
## Face-centroids-track-and-counter
A project can tracking and counting faces in real time, can be seen as a paramount to build a ***Person-Counter***. .

![result](https://github.com/LZQthePlane/Object-tracker-collection-Opencv-DeepLearning/blob/master/Faces-centroid-tracker-counter/test_out/example.gif)

----------------------------------------------------
## Faster-object-tracker-with-opencv-constructor
Apply detection only once and then track it in every subsequent frame, leading to faster tracking pipline.  
(Using functions provided by OpenCV)

![result](https://github.com/LZQthePlane/Object-tracker-collection-Opencv-DeepLearning/blob/master/Faster-object-tracker-with-opencv-constructor/test_out/example.gif)
