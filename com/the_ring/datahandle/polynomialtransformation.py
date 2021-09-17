import numpy as np
from sklearn.preprocessing import PolynomialFeatures

if __name__=='__main__':
    # 生成初始数据
    X = np.arange(6).reshape(3, 2)
    print(X)

    # 初始化数据预处理器，其中默认的次数项参数 degree 为 2
    poly = PolynomialFeatures()
    # 交互项参数 interaction_only 默认为 false，当其为 True 时，各个特征将会相乘，形如 xi^n*xj^m
    # 常数项 include_bias 默认为 True，转化之后会多出一个常数列
    print(u'二次转化之后的数据如下：')
    # 从原有的 (x1, x2) 转化为 (1, x1, x2, x1^2, x1*x2, x2^2)
    print(poly.fit_transform(X))

    print(u'三次转化且保留交互项之后的数据如下：')
    # 从原有的 (x1, x2) 转化为 (1, x1, x2, x1*x2)
    poly = PolynomialFeatures(degree=3, interaction_only=True)
    print(poly.fit_transform(X))