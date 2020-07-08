使用迁移学习再挑战 - CIFAR-10
===========================

## 知识点

* 迁移学习
* CIFAR-10再挑战

## 迁移学习

Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing 
knowledge gained while solving one problem and applying it to a different but related problem.
For example, knowledge gained while learning to recognize cars could apply when trying to 
recognize trucks. This area of research bears some relation to the long history of psychological 
literature on transfer of learning, although formal ties between the two fields are limited. 
From the practical standpoint, reusing or transferring information from previously learned tasks 
for the learning of new tasks has the potential to significantly improve the sample efficiency 
of a reinforcement learning agent.

### 一句话翻译

把别人训练好的模型拷贝过来自己用。

## 实战演习

```python
import numpy as np
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def run():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    ##############################################
    # 数据前处理
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    # 问题：两种方式对结果有什么影响？
    # X_train = normalize(X_train)
    # X_test = normalize(X_test)
    ##############################################

    num_classes = 10 
    y_train = to_categorical(y_train, num_classes = num_classes)
    y_test = to_categorical(y_test, num_classes = num_classes)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    from tensorflow.keras.applications.resnet50 import ResNet50
    base_model = ResNet50(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )

    # from tensorflow.keras.applications.xception import Xception
    # base_model = Xception(
    #     include_top = False,
    #     weights = "imagenet",
    #     input_shape = None
    # )

    # from tensorflow.keras.applications.vgg16 import VGG16
    # base_model = VGG16(
    #     include_top = False,
    #     weights = "imagenet",
    #     input_shape = None
    # )

    # from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    # base_model = MobileNetV2(
    #     include_top = False,
    #     weights = "imagenet",
    #     input_shape = None
    # )

    # from tensorflow.keras.applications.inception_v3 import InceptionV3
    # base_model = InceptionV3(
    #     include_top = False,
    #     weights = "imagenet",
    #     input_shape = None
    # )

    # from tensorflow.keras.applications.densenet import DenseNet121
    # base_model = DenseNet121(
    #     include_top = False,
    #     weights = "imagenet",
    #     input_shape = None
    # )

    ###################################################################
    # 全连接层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    # 模型网络定义
    model = Model(inputs = base_model.input, outputs = predictions)

    model.compile(
        optimizer = Adam(),
        loss = 'categorical_crossentropy',
        metrics = ["acc"]
    )

    # model.summary()
    print("模型网络定义：{}层".format(len(model.layers)))

    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    )

    # Reduce Learning Rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1
    )

    ###################################################################
    # 训练参数
    p_batch_size = 128
    p_epochs = 1

    ###################################################################
    # 图片掺水训练
    # 准备图片：ImageDataGenerator
    train_gen  = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True,
        width_shift_range=0.125, 
        height_shift_range=0.125, 
        horizontal_flip=True)
    test_gen = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True)

    # 数据集前计算
    for data in (train_gen, test_gen):
        data.fit(X_train)

    history = model.fit(
        train_gen.flow(X_train, y_train, batch_size=p_batch_size),
        epochs=p_epochs,
        steps_per_epoch=X_train.shape[0] // p_batch_size,
        validation_data=test_gen.flow(X_test, y_test, batch_size=p_batch_size),
        validation_steps=X_test.shape[0] // p_batch_size,
        callbacks=[early_stopping, reduce_lr])

    # 显示训练结果
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()

    # 模型保存
    model.save("model{}.h5".format(p_epochs))

    ###################################################################
    # 结果评价
    test_loss, test_acc = model.evaluate(
        test_gen.flow(X_test, y_test, batch_size=p_batch_size),
        steps=10)
    print('val_loss: {:.3f}\nval_acc: {:.3f}'.format(test_loss, test_acc ))

run()
```

## 课程文件

https://github.com/komavideo/CIFAR-10-by-Transfer-Learning

## 小马视频频道

http://komavideo.com
