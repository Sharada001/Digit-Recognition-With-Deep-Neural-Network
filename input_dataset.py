import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from prediction import predict_y_value

class Dataset:

    def __init__(self,file_name,ration,y_position,input_image_size,output_image_size):
        self.input_image_size = input_image_size
        self.output_image_size = output_image_size

        self.total_n_colimns = np.array(pd.read_csv(file_name)).shape[1]
        self.all_columns = [str(x) for x in range(self.total_n_colimns)]
        self.df = pd.read_csv(file_name, names=self.all_columns)
        self.array_set = np.array(self.df)
        np.random.shuffle(self.array_set)

        self.division_number = int(np.ceil(len(self.array_set)*ration))
        self.train_set = self.array_set[:self.division_number]
        self.test_set = self.array_set[self.division_number:]

        if y_position==0:
            self.x_train, self.y_train = self.train_set[:,1:], self.train_set[:,0].reshape(-1, 1).astype(int)
            self.x_test, self.y_test = self.test_set[:, 1:], self.test_set[:, 0].reshape(-1, 1).astype(int)
        if y_position==-1:
            self.x_train, self.y_train = self.train_set[:,:-1], self.train_set[:,-1].reshape(-1, 1).astype(int)
            self.x_test, self.y_test = self.test_set[:,:-1], self.test_set[:,-1].reshape(-1, 1).astype(int)

    def Train_set(self):
        return self.x_train, self.y_train

    def Test_set(self):
        return self.x_test, self.y_test

    def single_instance_gen(self,plot_n_print=1):
        for index in range(len(self.train_set)):
            predicted_y = predict_y_value(self.x_test[index])
            original_img = self.x_test[index].reshape(*self.input_image_size)
            resized_image = cv2.resize(original_img,self.output_image_size)
            if plot_n_print==1:
                print(f'predicted: {predicted_y},\ttrue_value: {self.y_test[index, 0]}')
                plt.imshow(resized_image)
                plt.show()
            yield resized_image, predicted_y

    def plot_instances(self,n_rows_cols):
        rows = n_rows_cols[0]
        columns = n_rows_cols[1]
        images = []
        iter_obj = self.single_instance_gen(0)
        for x in range(rows*columns):
            images.append(next(iter_obj))
        fig, ax = plt.subplots(rows,columns)
        index = 0
        for i in range(rows):
            for j in range(columns):
                ax[i][j].imshow(images[index][0])#,cmap='binary')
                ax[i][j].set_title(f'{images[index][1]}',y=0,pad=-15)
                ax[i][j].axis('off')
                index+=1
        plt.show()