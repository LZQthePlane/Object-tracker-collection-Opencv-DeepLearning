from collections import deque  # 用法与list相似，而list自带的append和pop方法（尾部插入和删除）速度慢
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import cv2 as cv
import imutils
import time
import os


video_path = os.path.dirname(os.path.abspath(__file__))
max_buffer = 16  # 所跟踪对象在视频流中保留的帧数
# 追踪的颜色上下限
green_low = (29, 100, 100)
green_up = (64, 255, 255)
# 跟随点的最大数量（小尾巴的点个数），初始值仍是0
points = deque(maxlen=max_buffer)


def get_contours(frame):
    blurred = cv.GaussianBlur(frame, (11, 11), 0)  # 高斯滤波,减少高斯噪声
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)  # 转为HSV格式
    de_background = cv.inRange(hsv, green_low, green_up)  # 去除背景，低于green_low和高于green_up的值均变为0
    erode = cv.erode(de_background, None, iterations=2)  # 先进行腐蚀，再进行膨胀，将小的噪点去除
    mask = cv.dilate(erode, None, iterations=2)
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取目标轮廓
    contours = contours[0] if imutils.is_cv2() else contours[1]  # 针对不同cv版本，取不同的索引以得到值
    return contours


def append_center_pts(contours, frame):
    c = max(contours, key=cv.contourArea)  # 找到最大的轮廓
    m = cv.moments(c)  # 图像矩
    center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))  # 获取轮廓图像的质心，基于面积
    ((x, y), r) = cv.minEnclosingCircle(c)  # 轮廓的最小包络圆及其中心坐标和半径
    if r > 10:
        cv.circle(frame, (int(x), int(y)), int(r), (0, 255, 255), thickness=2)  # 物体的轮廓
        points.appendleft(center)  # 更新所追踪目标质心的集合，注意是从集合左侧添加


def direction_show(contours, frame):
    dx, dy = 0, 0  # 坐标变化值
    direction = ''  # 方向提示文字
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        # 以10帧的变化量作为方向的参考，帧数过少变化过于频繁
        if i == 1 and len(points) >= 10:
            dx = points[-10][0] - points[i][0]
            dy = points[-10][1] - points[i][1]
            dir_x, dir_y = '', ''
            # 少于以50个像素的偏移，视为单向偏移
            if np.abs(dx) > 50:
                dir_x = "East" if np.sign(dx) == 1 else "West"
            if np.abs(dy) > 50:
                dir_y = "North" if np.sign(dy) == 1 else "South"
            # 两个方向均超过50个像素时，则视为二维偏移
            if dir_x != "" and dir_y != "":
                direction = "{}-{}".format(dir_y, dir_x)
            else:
                direction = dir_x if dir_x != "" else dir_y
        # 画跟踪线，thickness随时间变细
        if len(contours) > 0:
            thickness = int(np.sqrt(max_buffer / float(i + 1)) * 2.5)
            cv.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)
    return direction, dx, dy


def obj_track_func(frame):
    # 获取所有与目标颜色一致的objects的轮廓
    contours = get_contours(frame)

    if len(contours) > 0:
        # 获取目标object的轮廓并显示，并找到连续16帧的目标质心位置
        append_center_pts(contours, frame)

    # 显示质心的移动方向及偏移情况， 画跟随线
    direction, dx, dy = direction_show(contours, frame)

    # 添加左上角的移动方向文字，以及左下角的坐标变化文字
    cv.putText(frame, direction, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
    cv.putText(frame, "dx: {}, dy: {}".format(dx, dy), (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 1.25,
               (0, 0, 255), 2)
    cv.imshow("Green object tracker", frame)
    return frame


def track_with_cam():
    vs = cv.VideoCapture(0)  # VideoStream以线程方式处理相机帧，效率更高
    time.sleep(1.0)  # 让摄像头或视频预热
    fps = FPS().start()

    width = int(vs.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv.CAP_PROP_FRAME_HEIGHT))
    out_fps = 20
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    writer = cv.VideoWriter()
    out_path = video_path + os.sep + 'cam_out.mp4'
    writer.open(out_path, fourcc, out_fps, (width, height), True)

    while True:
        _, frame = vs.read()  # 获取当前帧
        # if frame is None:
        #     break
        # frame = imutils.resize(frame, height=800, width=1000)  # 缩小frame大小可以加快FPS
        frame = obj_track_func(frame)

        fps.update()
        fps.stop()
        writer.write(frame)

        if cv.waitKey(1) & 0xFF == ord("q"):  # 退出键
            break

    print('Elapsed time: {0:.2f}'.format(fps.elapsed()))  # webcam运行的时间
    print('Approx. FPS: {0:.2f}'.format(fps.fps()))  # 每秒帧数
    writer.release()
    vs.release()
    cv.destroyAllWindows()


def track_in_video():
    #  如果是视频文件的话 保存处理后的视频
    cap = cv.VideoCapture(video_path + os.sep + 'test_video.mp4')
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = int(cap.get(cv.CAP_PROP_FOURCC))  # 视频编码格式
    writer = cv.VideoWriter(video_path + os.sep + 'test_out.mp4', fourcc, fps, (width, height))
    time.sleep(1.0)
    have_more_frame = True
    while have_more_frame:
        have_more_frame, frame = cap.read()  # 获取当前帧
        if have_more_frame:
            frame = obj_track_func(frame=frame)
            writer.write(frame)
        if cv.waitKey(1) & 0xFF == ord("q"):  # 退出键
            break
    cap.release()
    writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # track_in_video()
    track_with_cam()
