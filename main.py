import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from matplotlib.pyplot import figure
from prediction import predict_y_value
from input_dataset import Dataset


dataset_1 = Dataset('img_data_2.csv',0.8,-1,(20,20),(400,400))
X_test, Y_test = dataset_1.Test_set()
dataset_1.plot_instances((5,10))


'''
img = cv2.imread('pic1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = 255 - img
#print(img.max(),img.min())
unique, counts = np.unique(img,return_counts=True)
unique = unique.reshape((-1,1))
counts = counts.reshape((-1,1))
set_ = (img.min()+30<img).astype(int).astype(float)
img = img*set_
shp = img.shape
img = img.flatten()
for i,value in np.ndenumerate(img):
    if value == 0:
        img[i] = 27
img = img.reshape(shp)

x = cv2.resize(img,(20,20))
x = x/255*1.2596515293574126 - 0.2626782601596817
x = x.reshape(1,-1)

predicted_y = predict_y_value(x)  #[index]
print(predicted_y)

#img = x[index,:].reshape(20,20)
#resized_img = cv2.resize(img,(400,400))

plt.imshow(img,cmap='binary')
plt.show()
'''







'''
x = 1-array_set[:,:-1]
y = array_set[:,-1].reshape(-1,1)
new = np.hstack((x,y))
df = pd.DataFrame(new,columns=columns)
df.to_csv('img_data_1.csv',header=False,index=False)

new_images = []
for x in range(len(array_set)):
 new_images.append(array_set[x,:-1].reshape(20,20).T.reshape(1,-1))
x = np.squeeze(np.array(new_images))
y = array_set[:,-1].reshape(-1,1)
new_array = np.hstack((x,y))
df = pd.DataFrame(new_array,columns=columns)
df.to_csv('img_data_2.csv',header=False,index=False)

img = array_set[0,:-1].reshape(20,20)
print(array_set[0,-1])
resized_img = cv2.resize(img,(400,400))
plt.imshow(resized_img)
plt.show()

fig_1 = figure()
ax_1 = fig_1.add_axes([0,0,1,1])
#ax_1.plot(*costs)
plt.imshow(X[300,:].reshape(20,20).T)
plt.show()
'''


