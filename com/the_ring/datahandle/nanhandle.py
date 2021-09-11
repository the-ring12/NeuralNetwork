# 引入相应的库，Numpy 用于生成缺失值，sklearm.impute 库中 SimpleImputer 方法用于处理缺失值
import numpy as np
from sklearn.impute import SimpleImputer

# 初始化缺失值处理器