import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher # FeatureHasher 为特征哈希类
from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer 为词频统计类
from sklearn.metrics.pairwise import euclidean_distances # euclidean_distances 为计算欧式聚类的方法

# 分词技术
def word_split():
    seg_list = jieba.cut(u"机器学习从入门到入职", cut_all=True)
    print(u'全模式分词输出如下：')
    print(", ".join(seg_list))
    # 精确模式，其默认是精确模式
    seg_list = jieba.cut("机器学习从入门到入职", cut_all=False)
    print("默认模式为精确模式：")
    print(",".join(seg_list))
    print(u'切割最后生成 list，如下：')
    list = jieba.lcut(u'机器学习从入门到入职')
    print(list)

# 独热编码
def one_hot_encoding():
    # 在字典中有 3 个不同的词汇
    city_dict = [{'city': 'beijing'}, {"city": "wuhan"}, {'city': 'shenzhen'}]
    # 初始化字典向量化器
    vec = DictVectorizer()
    print(u'将字典拟合再向量化之后。输出结果如下：')
    print(vec.fit_transform(city_dict).toarray())
    print(u'将新的字典用之前生成的向量化器转化，输出结果如下：')
    city_dict_new = [{'city': 'wuhan'}, {'city': 'shanghai'}, {'city': 'beijing'}]
    print(vec.fit_transform(city_dict_new).toarray())

# 哈希技巧
def feature_hash():
    # 文本数据
    text_all = ['The HongKong journalist is running faster than anyone else, it is excited, I have been to all '
                'western countries, I know pretty much what us']
    text1 = ['The HongKong journalist is running faster than anyone else, it is excited']
    text2 = ['I have been to all western countries, I know pretty much what it is']
    fake_text = ['The HK journalist !$ running f@ster th@n any1 else, I I to']

    # 初始化转换器
    vectorizer = CountVectorizer()
    hv_position = FeatureHasher(n_features=6, input_type='string')
    hv = FeatureHasher(n_features=6, input_type='string')
    text_name = ['text1', 'text2', u'伪造文件']

    # 文本数据转换的方法
    def show_transform_code(transformer, text_list=[text1, text2, fake_text]):
        # 拟合总文档 text_all
        transformer_new = transformer.fit(text_all)
        # 将转换之后的编码写入变量 dist
        dist = [transformer_new.transform(x).toarray()[0] for x in text_list]
        print(u'======================================================')
        for j,i in enumerate(dist):
            # print(i)
            if j == i:
                print(u'text2转化之后的编码如下，')
                print(dist[0])
                continue
            # 计算不同编码之间的欧氏距离
            ed = euclidean_distances([dist[1]], [i])
            print(u'{0}转化之后的编码如下，'.format(text_name[j]))
            print(i)
            print(u'{0}与text2的欧氏距离{1：.3f'.format(text_name[j], ed[0][0]))
            print(u'---------------------------------------------------------')
        print(u'============================================================')

    print(u'词频统计变换')
    show_transform_code(vectorizer)
    print(u'无符号特征哈希变换')
    show_transform_code(hv_position)
    print(u'有符号特征哈希变换')
    show_transform_code(hv)


if __name__=='__main__':
    # # 分词技术
    # word_split()
    #
    # # 独热编码
    # one_hot_encoding()
    #
    # 哈希技巧
    feature_hash()