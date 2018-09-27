from scipy.spatial.distance import cdist
from collections import OrderedDict
import numpy as np


class CentroidTracker():
    def __init__(self, max_fra_disap=20):
        # 当前检测到的face ID的序号
        self.cur_faceID = 1
        # 当前存在的faces，ID为key，坐标(x, y)为value
        self.faces = OrderedDict()
        # 当前帧中不存在， 但依然未注销的faces,ID为key，帧数为value
        self.pre_disap = OrderedDict()
        # 当pre_disap大于max_fra_disap时，注销其信息
        self.max_fra_disap = max_fra_disap

    def register(self, centroid):
        # 在faces字典集合中添加一个item: (ID, centroid)
        self.faces[self.cur_faceID] = centroid
        # 将该face添加到“即将注销”集合中，将“已消失帧数”初始化为0，
        # 消失后开始随frame增加，直到达到max_fra_disap，注销
        self.pre_disap[self.cur_faceID] = 0
        # 注册一个face后，下一个faceID号+1
        self.cur_faceID += 1

    # face消失超过max_fra_disap帧后，彻底注销其信息
    def deregister(self, face_id):
        # 在当前faces集合和“即将注销”集合中，删除此item
        del self.faces[face_id]
        del self.pre_disap[face_id]

    def update(self, bboxs):
        # 若当前帧检测无face对象
        if len(bboxs) == 0:
            # 遍历当前所有在“即将注销”字典集合中的faces
            for id in self.pre_disap.keys():
                # 给这些face保存的帧数+1（update函数每一帧调用一次）
                self.pre_disap[id] += 1
                # 当pre_disap大于max_fra_disap时，注销其信息
                if self.pre_disap[id] > self.max_fra_disap:
                    self.deregister(id)
            # 返回处理过后的faces集合，可能有被注销的face
            return self.faces

        # 若当前帧有face对象，则初始化一个二维数组集合用于存放faces的质心，
        # 由于后面需要带入cdist()函数中，需要定义为np.array的形式
        input_centroids = np.zeros((len(bboxs), 2), dtype=np.int)
        # 通过bounding box将每个face的质心求出，并保存到集合中
        # bounding box可由R-CNN/SSD/YOLO等各种目标检测算法得到
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(bboxs):
            x = int((start_x + end_x) / 2.0)
            y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (x, y)

        # 若当前没有正在进行track的face，则依次对其注册
        # 一般发生在frame重新出现bounding box时，此时这些face还未进行注册
        if len(self.faces) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])

        # 否则，对两帧之间相距最近的质心，分别进行关联
        # 这里假设：在两帧之间，任一个object在所移动的距离，小于其与其他object之间的距离
        else:
            # 获取当前保存着的的face的所有item的信息，即上一帧的信息
            # 不包括当前帧中可能出现的新的face
            face_ids = list(self.faces.keys())
            face_centroids = list(self.faces.values())

            # --------------此代码块详见Parse_tracking_distance_relation------------------
            # 该函数用于计算两个输入集合的距离，通过metric参数指定计算距离的不同方式
            # 结果数组中的第一行数据表示的是x1数组中第一个元素点与x2数组中各个元素点的距离
            # 这里使用欧式距离
            D = cdist(np.array(face_centroids), input_centroids, metric='euclidean')

            # 找到每行中的最小值，根据最小值对行索引进行排序,得到上一帧中待关联的质心
            rows = D.min(axis=1).argsort()
            # 找到每列中的最小值，然后根据排序的行对它们进行排序，
            # 得到这一帧中待关联的质心，且与上一帧的待关联质心一一对应
            cols = D.argmin(axis=1)[rows]
            # ----------------------------------------------------------------------------

            # 初始化俩个set集合用于存储我们已使用的行索引/列索引，用于去重
            # 当前帧与上一帧中的centroids需要一一对应，不可以出现重复关联的情况
            use_in_pre, use_in_cur = set(), set()
            # 遍历所有关联好的质心的索引（associated centroids index）
            for (pre_asso_index, cur_asso_index) in zip(rows, cols):
                # 如果已经使用过，则忽略，防止重复关联
                if pre_asso_index in use_in_pre or cur_asso_index in use_in_cur:
                    continue
                # 正式进行关联，将id匹配的 上一帧中质心与当前帧质心进行关联
                # 并将其“即将注销”属性重新清零
                id = face_ids[pre_asso_index]
                self.faces[id] = input_centroids[cur_asso_index]
                self.pre_disap[id] = 0
                # 将已进行过匹配的index归到“已使用”的集合
                use_in_cur.add(cur_asso_index)
                use_in_pre.add(pre_asso_index)

            # 获取未使用，即未进行关联的的质心集合，这些质心要么是下一帧就不存在，要么就是新出现
            # D.shape[0]表示上一帧的质心数量，[1]表示当前帧的质心数量
            unused_in_pre = set(range(D.shape[0])).difference(use_in_pre)
            unused_in_cur = set(range(D.shape[1])).difference(use_in_cur)
            # 如果上一帧的质心数量大于当前帧，需要对上一帧中未匹配关联到的质心的“即将注销”属性+1
            if D.shape[0] >= D.shape[1]:
                for index in unused_in_pre:
                    id = face_ids[index]
                    self.pre_disap[id] += 1
                    # 若超过max_fra_disap值，则进行注销
                    if self.pre_disap[id] > self.max_fra_disap:
                        self.deregister(id)
            # 若当前帧的质心多于上一帧，将多出来的依次进行注册
            else:
                for index in unused_in_cur:
                    self.register(input_centroids[index])
        return self.faces
