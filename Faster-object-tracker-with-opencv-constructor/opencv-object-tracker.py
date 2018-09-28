import cv2 as cv
import time
import os
from imutils.video import FPS

file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep

# opencv内置的八种目标跟踪算法
# 确保安装了opencv-contrib模块
OPENCV_OBJECT_TRACKERS = {
        "csrt": cv.TrackerCSRT_create(),
		"kcf": cv.TrackerKCF_create(),
		"boosting": cv.TrackerBoosting_create(),
		"mil": cv.TrackerMIL_create(),
		"tld": cv.TrackerTLD_create(),
		"medianflow": cv.TrackerMedianFlow_create(),
		"mosse": cv.TrackerMOSSE_create()
}
# 选择算法
tracker_str = "mosse"
tracker = OPENCV_OBJECT_TRACKERS[tracker_str]
# 初始化追踪目标的bounding box
bounding_box = None

# 获取摄像头, 这里使用的是opencv的API，而非imutils中的VideoStream，cap.read()返回值有所不同
cap = cv.VideoCapture(0)
time.sleep(1.0)
fps = None

# 输出视频的相关参数
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
out_fps = 20  # 输出视频的帧数
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 输出视频的格式
writer = cv.VideoWriter()
out_path = file_path+'test_out'+os.sep+'example.mp4'
writer.open(out_path, fourcc, out_fps, size, True)

while True:
    _, frame = cap.read()
    origin_h, origin_w = frame.shape[:2]
    if bounding_box is not None:
        (success, bbox) = tracker.update(frame)
        if success:
            # 获取所有的bounding box，并可视化
            (x, y, w, h) = [int(v) for v in bbox]
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 更新fps，定义在选定bounding box后
        fps.update()
        fps.stop()

        # 显示在画面中的信息
        info = [
            ("Tracker", tracker_str),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, origin_h - ((i * 20) + 20)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv.imshow('Frame', frame)
    writer.write(frame)

    key = cv.waitKey(1) & 0xff
    # 按s键，为跟踪的目标画bounding box
    if key == ord("s"):
        bounding_box = cv.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, bounding_box)
        fps = FPS().start()
    # 退出键q
    if key == ord("q"):
        break

writer.release()
cap.release()
cv.destroyAllWindows()