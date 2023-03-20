

import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import tree

data = loadmat('DS (2).mat')#导入数据
data = data['data']  # belong to the range of (-90.2504 -14.0318)
#nrows = data.nrows
print('data',data.shape)
print('data.shape[1]',data.shape[1])
print('data[0]',data[0])
#exit()

#
#
#
data1 = np.reshape(data, [-1,1,2,4])
#data1 = np.reshape(data, [-1,3,8,4])三种特征，8个采样点，4个房间
data1 = np.swapaxes(data1,0,3)
data1 = np.swapaxes(data1,1,3)

print(data1.shape)
print(data1[0:2])

label_data = []
for i in range(data1.shape[0]):
    for j in range(data1.shape[1]):
        label_data.append(i)
print(len(label_data))

for i in range(data1.shape[3]):
    for j in range(data1.shape[2]):
        data1[:, :, j, i] = (data1[:, :, j, i]-np.min(data1[:, :, j, i]))/(np.max(data1[:, :, j, i])-np.min(data1[:, :, j, i]))
print(np.max(data1), np.min(data1))
print(data1.shape)

#
#
#
data1 = np.reshape(data1,[-1,2,1])
#data1 = np.reshape(data1,[-1,8,3])8个采样点，三种数据
print(data1.shape)

#data_minMax =data
#print('data_minMax',data_minMax[0:4])
from sklearn.model_selection import train_test_split
power_train, power_test, L_train, L_test = train_test_split(data1, label_data,test_size=0.2, shuffle=True)

# power_train, power_test, L_train, L_test = getData(data_minMax, index)

print('power_train',power_train[0:4])
print('L_train',L_train[0:4])
print('power_test',power_test[0:4])
print('L_test',L_test[0:4])
#exit()
path = 'random_pos_2/'
np.save(path + 'gtrain', power_train)
np.save(path + 'gtest', power_test)
np.save(path + 'L_train', L_train)
np.save(path + 'L_test', L_test)


































