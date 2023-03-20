import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def acc(pred, true):
    c = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            c = c + 1
    acc = c/len(pred)

    return acc


path = 'random_pos_2/'

x_train = np.load(path + 'gtrain.npy')
y_train = np.load(path + 'L_train.npy')
x_test = np.load(path + 'gtest.npy')
y_test = np.load(path + 'L_test.npy')
print('x_train',x_train[0])
print(y_train[0])
print('x_train', len(x_train))
print('L_train', len(y_train))
print('x_test', len(x_test))
print('x_test[0]', x_test[0])
print('y_test', len(y_test))
print('x_train',x_train[0:4])
print(y_train[0:4])
#exit()
np.random.seed(106)
np.random.shuffle(x_train)
np.random.seed(106)
np.random.shuffle(y_train)

np.random.seed(1016)
np.random.shuffle(x_test)
np.random.seed(1016)
np.random.shuffle(y_test)

# 指出训练集的标签和特征以及测试集的标签和特征，0.2为参数，对测试集以及训练集按照2:8进行划分

model = KNeighborsClassifier(n_neighbors=9)
#[-1,6]--6-表示有几行数据值
#列如：8个点。3种数据就有24个
x_train = np.reshape(x_train,[-1,2])
x_test = np.reshape(x_test,[-1,2])

model.fit(x_train, y_train)  # 现在只需要传入训练集的数据
pred = model.predict(x_test)
print(pred[:8])
print(y_test[:8])


clf = svm.SVC()  # svm class
clf.fit(x_train, y_train)
pred_svm = clf.predict(x_test)
ACC2 = acc(pred_svm, y_test)
print('SVM ACC = ', ACC2)

















