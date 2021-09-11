
if __name__ == '__main__':
    # 创建神经网络
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.09
    N = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 训练神经网络
    training_data_file = open("C:/User/tan/Desktop/", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    epochs = 5  # 训练 5 个世代
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 输入数据取值范围 0.01~1.0
            targets = numpy.zeros(output_nodes) + 0.01  # 输出数据 0.01 和 0.99
            targets[int(all_values[0])] = 0.99
            N.train(inputs, targets)
            pass
        pass

    # 用 mnist 测试数据集来测试神经网络
    test_data_file = open("C:/Users/Tan/Desktop", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    # 计算正确率
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])  # 正确的数字
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = N.query(inputs)
        label = numpy.argmax(outputs)  # 找出一列中最大的数的位置，从 0 开始编号，也就是输出的数字
        if (correct_label == label):
            scorecard.append(1)
        else:
            scorecard.append(0)
        pass
    scorecard_array = numpy.asarray(scorecard)
    print("mnist 测试数据集中 10000 条手写数据的识别正确率为：", scorecard_array.sum() / scorecard_array.size)

    # 用自己手写数字来测试神经网络
    from PIL import Image # 导入图像处理工具
    image = Image.open("E:/MLP/9.png").convert('F') # 打开图像
    # image = image.resize((28, 28)) # 调整图像大小
    arr = [] # 将图像中的像素作为预测数据点的特征
    for i in range(28):
        for j in range(28):
            pixel = 255.0 - float(image.getpixel((j, i)))
            pixel2 = (pixel / 255.0 * 0.09) + 0.01
            arr.append(pixel2)
    arr1 = numpy.array(arr.reshape(1, -1)) # 只有一个样本，需要进行 reshape 操作
    # 画出将要识别的手写数字
    import matplotlib.pyplot
    image_array = numpy.asfarray(arr1).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap = 'Greys', interpolation = 'None')
    # 神经网络识别出手写数字
    label2 = numpy.argmax(N.query(arr1)) # 找出一列中最大的数的位置，从 0 开始编号，也就是输出的数字
    print("识别的数字为：", label2)
    # N.query(arrl1)

    # 画出 mnist 数据集中任何一行的数字
    import matplotlib.pyplot
    all_values = test_data_list[12].split(',')
    image_array = numpy.asfarray(all_values[1:].reshape(28, 28))
    matplotlib.pyplot.imshow(image_array, cmap = 'Greys', interpolation = 'None')
    N.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)