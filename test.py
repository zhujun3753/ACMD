# 综合分类数据集
from numpy import where
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
# 定义数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# 为每个类的样本创建散点图
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.figure()
for class_value in range(2):
# 获取此类的示例的行索引
    row_ix = where(y == class_value)
    # 创建这些样本的散布
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
plt.show()