from scipy.spatial import distance as dist
import numpy as np

np.random.seed(42)
objectCentroids = np.random.uniform(size=(2, 2))
print(objectCentroids)
# [[0.37454012 0.95071431]
 # [0.73199394 0.59865848]]
centroids = np.random.uniform(size=(3, 2))
print(centroids)
# [[0.15601864 0.15599452]
#  [0.05808361 0.86617615]
#  [0.60111501 0.70807258]]
D = dist.cdist(objectCentroids, centroids)
# [[0.82421549 0.32755369 0.33198071]
#  [0.72642889 0.72506609 0.17058938]]

print(D.min(axis=1))
# [0.32755369 0.17058938]  找到每一行的最小值
rows = D.min(axis=1).argsort()
print(rows)
# [1 0]  将每一行的最小值所在的行索引，按照最小值的大小 从小到大进行排序

print(D.argmin(axis=1))
# [1 2]  找到每一行最小值所在的列
cols = D.argmin(axis=1)[rows]
print(cols)
# [2 1]  将列索引 按照最小值的大小 从小到大进行排序

print(list(zip(rows, cols)))
# [(1, 2), (0, 1)]
# 结果表明：objectCentroids中的第2列代表的质点A与centroids中第3列代表的质点a相配
#           objectCentroids中的第1列代表的质点B与centroids中第2列代表的质点b相配
# 即认为，在相邻的两帧之间，A与a为同一个object的质点，B与b为同一个object的质点