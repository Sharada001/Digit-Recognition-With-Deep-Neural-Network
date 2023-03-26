import cv2
import numpy as np
import matplotlib.pyplot as plt
from input_dataset import Dataset



dataset_1 = Dataset('img_data_2.csv',0.8,-1,(20,20),(400,400))
#X_test, Y_test = dataset_1.Test_set()
#dataset_1.plot_instances((5,10))

#print(dataset_1.x_train.min(),dataset_1.x_train.max())
#print(dataset_1.x_test.min(),dataset_1.x_test.max())
max = 1.127688299158888
min = -0.1319632301985248
range_ = max-min
new_x_val = (dataset_1.x_train[0]-min)/range_*255
# new_image = new_x_val.reshape(20,20)
# new_image = cv2.resize(new_image,(400,400))
# plt.imshow(new_image,cmap='binary')
# plt.show()

new_set = dataset_1.array_set[:,:-1]
de_normalized = new_set + np.mean(new_set)
range = de_normalized.max()-de_normalized.min()
de_scaled = np.round(de_normalized/range_*255).astype(int)
print(de_scaled.min(),de_scaled.max())
print(de_scaled)
new_image = de_scaled[0].reshape(20,20)
#new_image = cv2.resize(new_image,(400,400))
plt.imshow(new_image,cmap='binary')
plt.show()

