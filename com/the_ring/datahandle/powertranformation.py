# 引入相应的库
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

def powertransformation():
    # 设置特定参数
    N_SAMPLES = 1000
    size = (N_SAMPLES)

    # 初始化随机数发生器
    rng = np.random.RandomState(304)

    # 生成 lognormal 分布
    X_lognormal = rng.lognormal(size=size)

    # 生成高斯分布
    loc = 100
    X_gaussian = rng.normal(loc=loc, size=size)

    # 生成均匀分布
    X_uniform = rng.uniform(low=0, high=1, size=size)

    # 将要展示的分布数据
    distributions = [
        ('lognormal', X_lognormal),
        ('Gaussian', X_gaussian),
        ('Uniform', X_uniform),
        ('lognormal after yeo-johnson', []),
        ('Gaussian after yeo-johnson', []),
        ('Uniform after yeo-johnson', []),
        ('lognormal after box-cox', []),
        ('Gaussian after box-cox', []),
        ('Uniform after box-cox', []),
    ]

    # 图标初始化
    f, ax = plt.subplots(ncols=3, nrows=3, figsize=(8, 6))

    # 开始画图
    for i, d in enumerate(distributions):
        # 配置图位置坐标信息
        a = int(np.floor(i / 3))  # floor 大于的最小整数
        b = np.mod(i, 3)  # i 除以 3 的余数
        title, data = d

        # 绘制原始数据图表
        if data != []:
            sns.distplot(data, kde=False, rug=True, ax=ax[a, b], color='r')
            ax[a, b].set_title(title)
        # 绘制经过分位数处理之后的图表
        else:
            _, data = distributions[b]
            X_train, X_test = train_test_split(data, test_size=.5)
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

            # 在分位数分布中，可以采用两种策略，即Yeo-Johnson 方法及 Box-cox 方法，其配置项为method
            strategy = title.split()[2]
            show_data = PowerTransformer(method=strategy).fit(X_train).transform(X_test)
            sns.distplot(show_data, kde=False, rug=True, ax=ax[a, b])
            ax[a, b].set_title(title)

    # 图标展示
    plt.show()

if __name__ == '__main__':
    powertransformation()