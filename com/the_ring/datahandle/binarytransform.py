# 引用相应的库，Numpy 用于生成随机值，引用 sklearn.preprocessing 库，其中绝大部分的数据预处理方法
from sklearn import preprocessing
import numpy as np

if __name__=='__main__':
    # 原始数据 X
    X = np.asarray([[3., -2, 2], [2., 0., 0.], [-1, 1., 3.]])
    # 初始化数据预处理器，本例中为二进制变换器，其中阈值为 1，当大于阈值时转化值为1，反之则为0
    binnarizer = preprocessing.Binarizer(threshold=1)
    # fit_transform 是将数据拟合与转化在同一个步骤实现
    X_binarizer = binnarizer.fit_transform(X)
    print(X)
    print()
    print(X_binarizer)
    

