# 引入相应的库，Numpy 用于生成缺失值，sklearm.impute 库中 SimpleImputer 方法用于处理缺失值
import numpy as np
from sklearn.impute import SimpleImputer

def nanhandle():
    # 初始化缺失值处理器，指定缺失值参数 missing_values，默认 np.nan
    # 以及缺失值补全策略参数 strategy，本例中采用的是均值补全，此外还包括如下策略
    # 中位数（median）
    # 常数（constant）
    # 最高频数（most_frequent）
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant')
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X = [[13, 22], [5, 3], [7, np.nan], [np.nan, 5], [3, 7]]
    # 执行缺失值处理的步骤，在 fit 原始数据之后对其进行转换
    imp_mean.fit(X)
    imp_median.fit(X)
    imp_constant.fit(X)
    imp_most_frequent.fit(X)
    print(u"原始数据：")
    print(X)
    print(u"均值处理，结果如下：")
    print(imp_mean.transform(X))
    print(u"中位数处理，结果如下：")
    print(imp_median.transform(X))
    print(u"常数处理，结果如下：")
    print(imp_constant.transform(X))
    print(u"最高频处理，结果如下：")
    print(imp_most_frequent.transform(X))


if __name__ == "__main__":
    nanhandle()