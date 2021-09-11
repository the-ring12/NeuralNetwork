# 引入团点生成器
from sklearn.datasets._samples_generator import make_blobs

def dataset():
    # 生成 10 个样本， 3 个团点，数据特征为 2 个，随即状态为 0
    x, y = make_blobs(n_samples=10, centers=3, n_features=2, random_state=0)
    print(x, y)


if __name__ == '__main__':
    dataset()