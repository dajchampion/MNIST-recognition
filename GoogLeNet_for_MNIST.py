import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, concatenate, Dropout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def inception_block(x, filters):
    """定义 Inception 模块"""
    f1, f2_reduce, f2, f3_reduce, f3, f4 = filters  # 各分支的通道数

    # 分支 1：1×1 卷积
    branch1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

    # 分支 2：1×1 降维 + 3×3 卷积
    branch2 = Conv2D(f2_reduce, (1, 1), padding='same', activation='relu')(x)
    branch2 = Conv2D(f2, (3, 3), padding='same', activation='relu')(branch2)

    # 分支 3：1×1 降维 + 5×5 卷积
    branch3 = Conv2D(f3_reduce, (1, 1), padding='same', activation='relu')(x)
    branch3 = Conv2D(f3, (5, 5), padding='same', activation='relu')(branch3)

    # 分支 4：3×3 最大池化 + 1×1 卷积
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch4 = Conv2D(f4, (1, 1), padding='same', activation='relu')(branch4)

    # 拼接所有分支
    output = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    return output

def build_googlenet(input_shape, num_classes):
    """构建 GoogLeNet 网络"""
    inputs = Input(shape=input_shape)

    # 初始卷积层
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    # 添加多个 Inception 模块
    x = inception_block(x, [32, 48, 64, 8, 16, 16])  # Inception 1
    # x = inception_block(x, [128, 128, 192, 32, 96, 64]) # Inception 2
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    # x = inception_block(x, [192, 96, 208, 16, 48, 64])  # Inception 3
    # x = inception_block(x, [160, 112, 224, 24, 64, 64]) # Inception 4
    # x = inception_block(x, [128, 128, 256, 24, 64, 64]) # Inception 5
    # x = inception_block(x, [112, 144, 288, 32, 64, 64]) # Inception 6
    # x = inception_block(x, [256, 160, 320, 32, 128, 128]) # Inception 7
    # x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    # x = inception_block(x, [256, 160, 320, 32, 128, 128]) # Inception 8
    # x = inception_block(x, [384, 192, 384, 48, 128, 128]) # Inception 9
    print(x.shape)
    # 全局平均池化
    x = AveragePooling2D(pool_size=(4,4))(x)
    print(x.shape)
    x = Flatten()(x)
    print(x.shape)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    # 构建模型
    model = Model(inputs, x)
    return model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0  # y_train有60000个,y_test有10000个

# 查看数据集的形状
print("x_train shape:", x_train.shape)  # 训练数据大小 (60000,28,28)
print("y_train shape:", y_train.shape)  # 训练标签大小
print("x_test shape:", x_test.shape)    # 测试数据大小
print("y_test shape:", y_test.shape)    # 测试标签大小

# MNIST 是灰度图 (28,28,1)，但 GoogLeNet 需要 (28,28,3)，所以转换为 3 通道
x_train = np.stack([x_train] * 3, axis=-1)  # 变成 (60000, 28, 28, 3)
x_test = np.stack([x_test] * 3, axis=-1)    # 变成 (10000, 28, 28, 3)

# 构建 GoogLeNet 模型
model = build_googlenet(input_shape=(28, 28, 3), num_classes=10)

# 显示模型摘要，包括每一层的参数数量和总参数数量
model.summary()

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test))
# model.save("googlenet_mnist_1incept_modified.h5")  # 保存整个模型（结构 + 权重）

# # 直接加载整个模型（包括结构和权重）
# model = load_model("googlenet_mnist_1incept.h5")

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 可视化多个识别结果
fig, axes = plt.subplots(3, 5, figsize=(10, 6))

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, x_test.shape[0])
    img = x_test[idx]
    true_label = y_test[idx]
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_label = np.argmax(pred)

    ax.imshow(img[:, :, 0], cmap='gray')
    ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()
