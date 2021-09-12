# 引用相应的库，Numpy 用于生成缺失值，引用 sklearn.processing 库，其中包含绝大部分的数据预处理方法
from sklearn import preprocessing
import numpy as np

# 最小值最大值缩放
def min_max_scale(X):
    # 初始化数据预处理器，本例中使用最小值-最大值缩放
    min_max_scaler = preprocessing.MinMaxScaler()
    # 数据转换并打印
    X_minmax = min_max_scaler.fit_transform(X)
    print("原始数据：\n")
    print(X)
    print("缩放规范化结果如下:\n")
    print(X_minmax)
    # 输出其缩放倍数
    print("输出其缩放倍数:\n")
    print(min_max_scaler.scale_)
    # 输出其最小值
    print("输出每一列的最小调整：\n")
    print(min_max_scaler.min_)
    print("输出每一列的最小值：\n")
    print(min_max_scaler.data_min_)

# 最大绝对值缩放
def max_absolute_scale(X):
    print("原始数据：\n")
    print(X)
    # 初始化数据预处理器，本例中为最大值缩放
    max_abs_scaler = preprocessing.MaxAbsScaler()
    # 数据转换并打印
    X_maxabs = max_abs_scaler.fit_transform(X)
    print("最大绝对值缩放结果如下：\n")
    print(X_maxabs)

# 自定义缩放
def define_scale(X):
    # 初始化数据预处理器，本例中为最小值-最大值缩放，需要配置 feature_range 可以转化之后的输出范围
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-2, 6))
    # 数据转换并打印
    X_minmax = min_max_scaler.fit_transform(X)
    print("原始数据：\n")
    print(X)
    print("自定义最大值和最小值之后，缩放规范化结果如下：\n")
    print(X_minmax)

# 标准化
def standard(X):
    print("原始数据：\n")
    print(X)
    # 初始化数据预处理器，本例中为标准化缩放
    standard_scaler = preprocessing.StandardScaler()
    # 数据转换并打印
    X_standard = standard_scaler.fit_transform(X)
    print("标准变换的结果如下：\n")
    print(X_standard)
    # 输出其均值
    print(u"输出标准化变换之后的均值：\n")
    print(X_standard.mean(axis=0))
    # 输出其标准差
    print(u"输出标准化之后的标准差：\n")
    print(X_standard.std(axis=0))


if __name__ == '__main__' :
    # 原始数据
    X = np.array([[3, -2., 2.], [2., 0., 0.], [1, 2., 3.]])

    # # 最小值最大值缩放
    # min_max_scale(X)
    #
    # # 最大绝对值缩放
    # max_absolute_scale(X)
    #
    # # 自定义最大值和最小值之后，缩放规范化结果如下
    # define_scale(X)
    #
    # # 标准化变换
    # standard(X)
    #
    # # 标准发缩放的函数化实现
    # X_standard = preprocessing.scale(X)
    # print(X_standard)
    #
    # # 范数规范化的函数化实现，其中规范化系数 norm 为 L2,还有其他形式，诸如 L1 和 max(无穷范数）
    # print(X)
    # X_norm = preprocessing.normalize(X, norm='l2')
    # print(u"范数L2规范化变换之后的输出：\n")
    # print(X_norm)

    # 初始化数据预处理器，本例为二值化变换器，其中阈值设置为 1，当大于阈值时值转化为 1，反之则为 0
    print(X)
    binarizer = preprocessing.Binarizer(threshold=1)
    # fit_transform 是将数据拟合转化在同一个步骤实现
    X_binarizer = binarizer.fit_transform(X)
    print("二值化变换结果：")
    print(X_binarizer)
