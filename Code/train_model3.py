import numpy as np
import os,random
from keras.layers import Input,Reshape,ZeroPadding2D,Conv2D,Dropout,Flatten,Dense,Activation,MaxPooling2D,AlphaDropout
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
plt.rc('font', family='Times New Roman')  # 设置全局字体
plt.rcParams.update({'font.size': 26})  # 设置全局字体大小
#plt.rcParams.update({'font.size': 24})  # 设置全局字体大小
#exit()
import seaborn as sns

#exit()
#增加代码
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cnns(in_shp, classes):

    model = Sequential()
    model.add(Reshape([1] + in_shp, input_shape=in_shp))

    model.add(Conv2D(kernel_initializer="glorot_uniform",
                     name="conv1",
                     activation="relu",
                     data_format="channels_first",
                     padding="same", filters=32, kernel_size=(3, 1)))
    #exit()
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    #model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    #model.add(MaxPooling2D())


    model.add(Conv2D(kernel_initializer="glorot_uniform",
                     name="conv2",
                     activation="relu",
                     data_format="channels_first",
                     padding="same", filters=64, kernel_size=(3, 1)))
    #model.add(MaxPooling2D((2,2), padding='same'))

    dr = 0.5  # dropout rate (%)

    #model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

    model.add(Conv2D(kernel_initializer="glorot_uniform",
                     name="conv4",
                     activation="relu",
                     data_format="channels_first",
                     padding="same", filters=64, kernel_size=(3, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer="he_normal",
                    activation="relu", name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(len(classes),
                    kernel_initializer="he_normal", name="dense2"))
    model.add(Activation('softmax'))
    model.add(Reshape([len(classes)]))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])
    model.summary()
    return model

#exit()
def getData():
    lb = LabelBinarizer()

    path = 'random_pos_2/'
    x_train = np.load(path + 'gtrain.npy')
    y_train = np.load(path + 'L_train.npy')
    x_test = np.load(path + 'gtest.npy')
    y_test = np.load(path + 'L_test.npy')

    #改变维度
    #
    #
    #2种数据*6个点=12个维度
    x_train = np.reshape(x_train, [-1,2])
    x_test = np.reshape(x_test, [-1,2])

    x_validate = x_test
    y_validate = y_test

    np.random.seed(106)
    np.random.shuffle(x_train)
    np.random.seed(106)
    np.random.shuffle(y_train)

    np.random.seed(122)
    np.random.shuffle(x_test)
    np.random.seed(122)
    np.random.shuffle(y_test)

    np.random.seed(300)
    np.random.shuffle(x_validate)
    np.random.seed(300)
    np.random.shuffle(y_validate)

    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    y_validate = lb.transform(y_validate)

    return x_train.reshape(x_train.shape[0], x_train.shape[1], 1), y_train, \
           x_test.reshape(x_test.shape[0], x_test.shape[1], 1), y_test, \
           x_validate.reshape(x_validate.shape[0], x_validate.shape[1], 1), y_validate
    #exit()
#exit()
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.binary, labels=[]):
    # plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)

#exit()
def train(epoch,  batch_size):
    classes = ['room 1', 'room 2', 'room 3','room 4']
    X_train, Y_train, X_test, Y_test, X_valid, Y_valid = getData()
    print('训练集X维度：', X_train.shape, '训练集Y维度：', Y_train.shape)
    print('验证集X维度：', X_valid.shape, '验证集Y维度：', Y_valid.shape)
    print('测试集X维度：', X_test.shape, '测试集Y维度：', Y_test.shape)
    in_shp = [X_train.shape[1], 1]
    print("in_shp ",in_shp)
    model = cnns(in_shp, classes)
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                        verbose=1)

    #history = model.fit(X_train, Y_train, epochs=epoch, validation_data=(X_test, Y_test), batch_size=batch_size,verbose=1)
    #ax = plt.gca()
    #sns.set(font_scale=1.5)  # 将混淆矩阵中的数字字体变大**
    #sns.set(font_scale=15)  # 将混淆矩阵中的数字字体变大
    #ax.set_ylabel('True', fontsize=15)
    #ax.set_xlabel('Pred', fontsize=15)
    #ax.tick_params(axis='y', labelsize=15, labelrotation=45)  # y轴
    #ax.tick_params(axis='x', labelsize=15)  # x轴

    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    print("len(classes)",len(classes))
    for i in range(0, X_test.shape[0]):
        j = list(Y_test[i, :]).index(1)
        k = int(np.argmax(test_Y_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plot_confusion_matrix(confnorm, classes, labels=classes)
    plt.savefig('6confusion.png')
    plt.show()
    c = 0
    for i in range(len(confnorm)):
        c = c + confnorm[i, i]
        print(classes[i], confnorm[i, i])
    acc = c / len(confnorm)
    print('mean acc = ', acc)
    #绘制精度图
    score = model.evaluate(X_test, Y_test, verbose=1)

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.legend([], loc='upper left')
    plt.ylim((0, 1))
    plt.savefig('6accuracy.png')
    plt.tight_layout()
    plt.show()
    #绘制损失精度图
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.legend([], loc='upper left')
    plt.ylim((0,1))
    # 设置x/y轴尺度
    #plt.xticks(a[::5])
    #plt.yticks([0,0.5,1])
    #保存图片
    plt.savefig('6loss.png')
    plt.tight_layout()
    plt.show()



    #保存模型
    #model_json = model.to_json()
    #open('p_ds3room_architeture.json', 'w').write(model_json)
    #model.save_weights('p_ds3room_weights.h5', overwrite=True)
#exit()
if __name__ == '__main__':
    train(epoch=2, batch_size=20)
    #train(epoch=200, batch_size=20)

