import numpy
import scipy.special

# neural network class of 3 layer
class neuralNetwork :

    # initialiation
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) :
        # 输入、隐藏和输出节点的个数（本类只适用于 3 层神经网络）
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate
        # 输入和隐藏以及隐藏和输出的两层连接权重矩阵
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)) # 正态分布方式初始化权重
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 激活函数 sigmoid()
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    # training
    def train(self, input_list, target_list):
        # <1>正向计算输出
        inputs = numpy.array(input_list, ndmin = 2).T
        targets = numpy.array(target_list, ndmin = 2).T
        # 计算隐藏层节点的输入和输出
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层节点的输入和输出
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # <2>反向计算误差
        # 计算输出层误差
        output_errors = targets - final_outputs
        # 计算隐藏层误差
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # <3>利用误差更新权重
        # 更新隐藏层和输出层之间的权重
        self.who = self.who + self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                  numpy.transpose(hidden_outputs))

        # 更新输入层和隐藏层之间的权重
        self.wih = self.wih + self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                  numpy.transpose(inputs))
        pass

    # quering
    def query(self, input_list):
        # 将输入的一行数据转化为二维数组，并且将行转置成列
        inputs = numpy.array(input_list, ndmin = 2).T
        # 计算隐藏层节点的输入和输出
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层节点的输入和输出
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 返回
        return final_outputs