import cv2 as cv
import numpy as np
import os
import time
from centroid_tracker import CentroidTracker

file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
threshold = 0.5  # objects' confidence threshold

# 加载预训练后的ResNetSSD的caffe模型
prototxt = file_path +'ResNetSSD'+os.sep+ 'Resnet_SSD_deploy.prototxt'
caffemodel = file_path +'ResNetSSD'+os.sep+ 'Res10_300x300_SSD_iter_140000.caffemodel'

net = cv.dnn.readNetFromCaffe(prototxt, caffeModel=caffemodel)
print('ResNetSSD caffe model loaded successfully')

# 获取摄像头
# 这里使用的是opencv的API，而非imutils中的VideoStream，cap.read()返回值有所不同
cap = cv.VideoCapture(0)
time.sleep(1.0)

# 显示实时的FPS
start_time = time.time()
interval = 1  # 每隔1秒重新计算帧数
counter = 0  # 统计每一秒的帧数
realtime_fps = 'Starting'

# 输出视频的相关参数
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
out_fps = 20  # 输出视频的帧数
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 输出视频的格式
writer = cv.VideoWriter()
out_path = file_path+'test_out'+os.sep+'example.mp4'
writer.open(out_path, fourcc, out_fps, size, True)

ct = CentroidTracker()

while True:
    _, frame = cap.read()
    origin_h, origin_w = frame.shape[:2]
    # 将每一帧图像送入深度学习模型中，通过前馈计算，得到detection结果
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    counter += 1  # 帧数+1

    bboxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
            bboxes.append(bounding_box.astype(int))
            x_start, y_start, x_end, y_end = bounding_box.astype(int)
            # 显示置信度
            label = '{0:.2f}%'.format(confidence * 100)
            # 画bounding box
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            # 画文字的填充矿底色
            cv.rectangle(frame, (x_start, y_start - 18), (x_end, y_start), (0, 0, 255), -1)
            # detection result的文字显示
            cv.putText(frame, label, (x_start + 2, y_start - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS的实时显示
        fps_show = 'FPS:{0:.4}'.format(realtime_fps)
        cv.putText(frame, fps_show, (int(origin_w * 0.75), int(origin_h * 0.95)), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0), 2)
        if (time.time() - start_time) > interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            realtime_fps = counter / (time.time() - start_time)
            counter = 0  # 帧数清零
            start_time = time.time()

    # 根据bounding box找到质心，并进行追踪更新
    faces = ct.update(bboxes)
    # 给每个质心进行文字和图像标记
    for (objID, centroid) in faces.items():
        text = 'ID {}'.format(objID)
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), thickness=-1)  # thickness=-1表示填充

    # 保存成视频
    writer.write(frame)
    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord("q"):  # 退出键
        break

# 释放摄像头，释放保存好的视频
writer.release()
cap.release()
cv.destroyAllWindows()