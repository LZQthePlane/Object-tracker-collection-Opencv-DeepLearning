# Faster-object-tracker-with-opencv-constructor
Apply detection only once and then track it in every subsequent frame, leading to faster tracking pipline.    
(The code comments using here is chinese)

-------------------------------------------------
## ***Features***
To track object in video-stream or in real-time, we usually run our detector on each frame, but it takes much **computing resouces** and potentially lead to **slower FPS**.  

Instead, the algorithms provided by opencv **apply object detection only once and then have the object tracker be able to handle every subsequent frame**, leading to a **faster, more efficient** object tracking pipeline.   

OpenCV includes eight separate object tracking implementations that you can use in your own computer vision applications.
1. ***BOOSTING Tracker***: Based on the same algorithm used to power the machine learning behind Haar cascades (AdaBoost), but like Haar cascades, is over a decade old. This tracker is slow and doesn’t work very well. Interesting only for legacy reasons and comparing other algorithms. (minimum OpenCV 3.0.0)
2. ***MIL Tracker***: Better accuracy than BOOSTING tracker but does a poor job of reporting failure. (minimum OpenCV 3.0.0)
3. ***KCF Tracker***: Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well. (minimum OpenCV 3.1.0)
4. ***CSRT Tracker***: Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than KCF but slightly slower. (minimum OpenCV 3.4.2)
5. ***MedianFlow Tracker***: Does a nice job reporting failures; however, if there is too large of a jump in motion, such as fast moving objects, or objects that change quickly in their appearance, the model will fail. (minimum OpenCV 3.0.0)
6. ***TLD Tracker***: I’m not sure if there is a problem with the OpenCV implementation of the TLD tracker or the actual algorithm itself, but the TLD tracker was incredibly prone to false-positives. I do not recommend using this OpenCV object tracker. (minimum OpenCV 3.0.0)
7. ***MOSSE Tracker***: Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed. (minimum OpenCV 3.4.1)
8. ***GOTURN Tracker***: The only deep learning-based object detector included in OpenCV. It requires additional model files to run (will not be covered in this post). My initial experiments showed it was a bit of a pain to use even though it reportedly handles viewing changes well (my initial experiments didn’t confirm this though). I’ll try to cover it in a future post, but in the meantime, take a look at Satya’s writeup. (minimum OpenCV 3.2.0)   


Suggestion from ***Adrian Rosebrock*** is to:
   - Use **CSRT** when you need **higher accuracy** and can tolerate slower FPS throughput;   
   - Use **KCF** when you need **faster FPS** throughput but can handle slightly lower object tracking accuracy;   
   - Use **MOSSE** when you need **pure speed**;

-------------------------------------------------
## ***Preparetion***
### - OpenCV minimum 3.4.1
So you can test all 8 algorithms constructed by opencv.

### - OpenCV-contrib (same version with opencv)
If you get error as following: "AttributeError: module ‘cv2.cv2’ has no attribute ‘TrackerCSRT_create’", it's may because your opencv version is offcial while some functions are changed and moved to their **opencv_contrib module** for some reasons.   

You can install it just by run `pip install opencv-contrib-python` in pip command or Anaconda command      
**Make sure the version is same with opencv you installed**.   

If it still doesn't work or you have problems when installing the same version, try this:   
**uninstall opencv --> install opencv-contrib --> install opencv again**.

--------------------------------------------------
## ***Result***
![result](https://github.com/LZQthePlane/Object-tracker-collection-Opencv-DeepLearning/blob/master/Faster-object-tracker-with-opencv-constructor/test_out/example.gif)

