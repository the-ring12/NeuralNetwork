import numpy as np
from sklearn.preprocessing import FunctionTransformer

# 定义自定义函数
def customer_function(x):
    return x*x-2*x+1

if __name__=='__main__':
    # 输入用于数据转换的自定义函数，数据验证项 validate 默认为 True，输入数据将被转换为 NumPy 矩阵
    transformer = FunctionTransformer(customer_function, validate=True)
    X = np.array([[0, 1], [2, 3]])
    print(u'自定义转换之后的数据如下：')
    print(transformer.transform(X))