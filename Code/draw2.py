import numpy as np
import matplotlib.pyplot as plt

x1 = [7,6,5,4,3,2,1]
#x1=np.array(x)
#x=[1,2,3,4,5,6,7]
y1=[1,0.993,0.99,0.98,0.99,0.91,0.32]
y2=[0.97,0.97,0.93,0.93,0.41,0.38,0.375]
y3=[0.83,0.83,0.78,0.78,0.54,0.43,0.343]
y1.reverse()
y2.reverse()
y3.reverse()
x1.reverse()
#plt.xticks(x1)
#plt.plot(x, y1, 'y*:', ms=10,label='CNN')

plt.xticks(x1)
plt.plot(x1, y1, 'g*:', ms=10,label='CNN')
plt.plot(x1, y2, 'yo:', ms=10,label='KNN')
plt.plot(x1, y3, 'r^:', ms=10,label='SVM')
plt.xlabel('Kind of data')
ax = plt.gca()
#ax.yaxis.set_ticks_position('right') #将y轴的位置设置到右端
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.show()














