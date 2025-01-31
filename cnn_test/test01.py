import keras
# import keras：导入 Keras 深度学习库，Keras 是一个高级神经网络 API，能帮助我们快速搭建和训练神经网络。
from keras.datasets import mnist
# from keras.datasets import mnist：从 Keras 的内置数据集模块中导入 MNIST 手写数字数据集。
# MNIST 数据集包含 60,000 张训练图像和 10,000 张测试图像，每张图像都是 28x28 像素的灰度手写数字（0 - 9）。
from keras.models import Sequential
# from keras.models import Sequential：导入Sequential模型类，
# Sequential模型是 Keras 中最简单的模型类型，它是一个线性堆叠的层序列，你可以按照顺序依次添加各个层来构建神经网络。
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense：从 Keras 的层模块中导入构建卷积神经网络所需的几种重要层。
# Conv2D：二维卷积层，用于在图像数据上进行卷积操作，提取图像的特征。
# MaxPooling2D：二维最大池化层，用于对卷积层的输出进行下采样，减少数据的维度，同时保留重要的特征信息。
# Flatten：展平层，用于将多维的输入数据展平成一维向量，以便将卷积层和池化层的输出连接到全连接层。
# Dense：全连接层，也称为密集层，层中的每个神经元都与前一层的所有神经元相连，用于进行分类等任务。



import matplotlib.pyplot as plt
import numpy as np


# 加载MNIST数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#mnist.load_data()：这是 Keras 提供的一个函数，用于自动下载并加载 MNIST 数据集。
# 它返回两个元组，第一个元组包含训练图像train_images和对应的训练标签train_labels，
# 第二个元组包含测试图像test_images和对应的测试标签test_labels。

# 重塑数据形状
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
# 原始的 MNIST 图像数据是二维数组（28x28），但 Keras 的卷积层要求输入数据是四维的，格式为(样本数量, 高度, 宽度, 通道数)。
# 对于灰度图像，通道数为 1。
# reshape方法用于将二维的训练图像和测试图像数组转换为四维数组，分别得到形状为(60000, 28, 28, 1)和(10000, 28, 28, 1)的数组。


# 创建模型model = Sequential([...])：使用Sequential类创建一个新的顺序模型，并通过列表的方式依次添加各个层
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #添加第一个卷积层，包含 32 个卷积核，每个卷积核的大小是 3x3，激活函数使用 ReLU（修正线性单元）。
    # input_shape=(28, 28, 1)指定了输入数据的形状，这是模型的第一层，必须指定输入形状。
    #MaxPooling2D((2, 2)),
    AveragePooling2D((2, 2)),
    #添加第一个最大池化层，池化窗口大小为 2x2，用于对卷积层的输出进行下采样。
    Conv2D(64, (3, 3), activation='relu'),
    #MaxPooling2D((2, 2)),
    AveragePooling2D((2, 2)),

    Flatten(),
    #展平层的作用是将前面卷积层和池化层输出的多维特征图转换为一维向量。
    #因为全连接层要求输入是一维数据，所以需要通过 Flatten 层将特征图展平，以便后续连接到全连接层进行分类。
    Dense(128, activation='relu'),
    #该全连接层包含 128 个神经元。
    # 这些神经元会接收展平后的一维向量作为输入，并对其进行线性变换和非线性激活（使用 ReLU 激活函数），进一步提取特征的组合信息。
    Dense(10, activation='softmax')
    #使用 Softmax 激活函数将输出层的输出转换为概率分布，即每个神经元的输出表示输入图像属于对应类别的概率
    # 所有神经元的输出概率之和为 1。
    # 这样可以方便地进行分类预测，选择概率最大的类别作为最终的预测结果。
])

# 编译模型.Keras中用于配置模型训练过程的重要方法。它需要指定优化器、损失函数和评估指标。
model.compile(optimizer='adam',#指定了 Adam 优化器,调整参数
              loss='sparse_categorical_crossentropy',
              #选择了稀疏分类交叉熵作为损失函数。
              # 当标签是整数编码（例如手写数字识别里的 0 - 9 这样的类别标签）时，就可以使用这个损失函数。
              # 它会衡量模型预测的概率分布与真实标签之间的差异，模型训练的目标就是最小化这个损失值。
              metrics=['accuracy'])#指定了评估指标为准确率

# 训练模型
history = model.fit(train_images, train_labels, epochs=5, batch_size=64)
#epochs指定训练的轮数为 5。一轮表示模型对整个训练数据集完整地进行一次前向传播和反向传播的过程。
# 增加轮数可以让模型有更多机会学习数据中的特征，但也可能导致过拟合。

# 在测试集上评估模型
# model.evaluate()：此方法用于在测试集上评估模型的性能。
# test_images：测试数据的输入，同样是手写数字的图像数据。
# test_labels：测试数据对应的真实标签。
# 该方法会返回两个值，test_loss 是模型在测试集上的损失值，test_acc 是模型在测试集上的准确率。
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


plt.figure(figsize=(12, 4))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制训练准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()

# 随机选择一些测试图像进行可视化预测结果
num_images = 5
random_indices = np.random.choice(len(test_images), num_images)

plt.figure(figsize=(15, 3))
for i, index in enumerate(random_indices):
    img = test_images[index].reshape(28, 28)
    true_label = test_labels[index]
    predictions = model.predict(np.expand_dims(test_images[index], axis=0))
    predicted_label = np.argmax(predictions)

    plt.subplot(1, num_images, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.axis('off')

plt.show()